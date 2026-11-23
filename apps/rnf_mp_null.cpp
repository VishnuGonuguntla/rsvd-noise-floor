// Phase 2 deliverable: calibrate the noise floor against the i.i.d. null.
//
// Generate returns that are pure noise -- the true correlation IS the identity
// -- then compute the full spectrum of the sample correlation matrix and
// compare it against the Marchenko-Pastur law. The point is that the
// eigenvalues do NOT cluster at 1: they spread over a wide bulk whose edges
// are predicted exactly. Everything later in the project cuts at that upper
// edge, so this is where the cut earns its credibility.
//
// The sweep over N at fixed q is the real test: MP is an asymptotic law, so
// the KS distance between the empirical and theoretical CDFs must shrink as N
// grows. A single N proves nothing.

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "rnf/cli.hpp"
#include "rnf/csv.hpp"
#include "rnf/matrix.hpp"
#include "rnf/mp.hpp"
#include "rnf/rng.hpp"
#include "rnf/stats.hpp"
#include "rnf/syev.hpp"
#include "rnf/timer.hpp"

using namespace rnf;

namespace {

struct NullRun {
    std::size_t         n = 0;
    std::size_t         t = 0;
    double              q = 0.0;
    double              ks = 0.0;
    double              pvalue = 0.0;
    double              lambda_max = 0.0;
    double              lambda_min = 0.0;
    double              trace = 0.0;
    double              build_ms = 0.0;
    double              syev_ms = 0.0;
    std::vector<double> spectrum;  // descending
};

// One null realization: i.i.d. Gaussian returns -> correlation -> spectrum.
NullRun run_null(std::size_t n, std::size_t t, std::uint64_t seed) {
    NullRun r;
    r.n = n;
    r.t = t;
    r.q = aspect_ratio(t, n);

    Stopwatch sw;
    Rng       rng(seed);
    Matrix    x = gaussian_matrix(t, n, rng);
    Matrix    c = correlation_from_returns(x);
    r.build_ms = sw.elapsed_ms();
    r.trace    = trace(c);

    sw.reset();
    r.spectrum = syev_values(c);  // values only: the O(N^3) vector accumulation is pure waste here
    r.syev_ms  = sw.elapsed_ms();

    r.lambda_max = r.spectrum.front();
    r.lambda_min = r.spectrum.back();

    const MpParams p{r.q, 1.0};
    r.ks     = ks_statistic(r.spectrum, p);
    r.pvalue = ks_pvalue(r.ks, n);
    return r;
}

void usage() {
    std::printf(
        "rnf_mp_null -- validate the sample-correlation spectrum against Marchenko-Pastur\n"
        "\n"
        "  --N <int>            assets in the detailed run      (default 1000)\n"
        "  --T <int>            observations in the detailed run (default 4000)\n"
        "  --seed <int>         RNG seed                         (default 42)\n"
        "  --bins <int>         histogram bins                   (default 80)\n"
        "  --sweep <list>       N values at fixed q, comma-separated\n"
        "                                                        (default 250,500,1000,2000)\n"
        "  --out <path>         histogram overlay CSV    (default results/mp_null.csv)\n"
        "  --sweep-out <path>   KS-vs-N CSV              (default results/mp_ks_sweep.csv)\n"
        "  --spectrum-out <path>  full sorted spectrum CSV (optional)\n");
}

}  // namespace

int main(int argc, char** argv) try {
    const Args args(argc, argv);
    if (args.has("help")) { usage(); return 0; }

    const std::size_t   n    = args.count("N", 1000);
    const std::size_t   t    = args.count("T", 4000);
    const std::uint64_t seed = args.seed("seed", 42);
    const std::size_t   bins = args.count("bins", 80);
    const std::string   out  = args.str("out", "results/mp_null.csv");
    const std::string   sweep_out = args.str("sweep-out", "results/mp_ks_sweep.csv");
    const auto sweep_ns = args.counts("sweep", {250, 500, 1000, 2000});

    if (n == 0 || t < 2) throw std::invalid_argument("need N >= 1 and T >= 2");

    const double    q = aspect_ratio(t, n);
    const MpParams  p{q, 1.0};
    const MpSupport sup = mp_support(p);

    std::printf("Marchenko-Pastur null test\n");
    std::printf("  N = %zu assets, T = %zu observations, q = N/T = %.6f\n", n, t, q);
    std::printf("  predicted bulk [lambda-, lambda+] = [%.6f, %.6f]\n", sup.lo, sup.hi);
    std::printf("  BBP detectability threshold 1+sqrt(q) = %.6f\n\n", bbp_threshold(q));

    // ---- detailed run -----------------------------------------------------
    const NullRun r = run_null(n, t, seed);

    std::printf("Detailed run (seed %llu)\n", static_cast<unsigned long long>(seed));
    std::printf("  trace(C)            = %.10f   (exactly N = %zu)\n", r.trace, n);
    std::printf("  empirical bulk      = [%.6f, %.6f]\n", r.lambda_min, r.lambda_max);
    std::printf("  lambda_max / lambda+= %.6f   (finite-size edge fluctuation)\n",
                r.lambda_max / sup.hi);
    std::printf("  KS distance         = %.6f\n", r.ks);
    std::printf("  KS p-value          = %.4f%s\n", r.pvalue,
                r.pvalue > 0.05 ? "   (consistent with MP)" : "   (REJECTS MP)");
    std::printf("  build %.1f ms, spectrum %.1f ms\n\n", r.build_ms, r.syev_ms);

    // How much of the spectrum is inside the predicted bulk -- the headline
    // number: almost all of it, for pure noise.
    std::size_t inside = 0;
    for (double v : r.spectrum) {
        if (v >= sup.lo && v <= sup.hi) ++inside;
    }
    std::printf("  %zu of %zu eigenvalues (%.2f%%) lie inside the predicted bulk\n\n", inside, n,
                100.0 * static_cast<double>(inside) / static_cast<double>(n));

    // Histogram spans a little beyond the support so the edges are visible.
    const double lo_plot = 0.0;
    const double hi_plot = sup.hi * 1.15;
    const SpectrumHistogram h = histogram_vs_mp(r.spectrum, p, bins, lo_plot, hi_plot);
    write_table(out, {"bin_centre", "empirical_density", "mp_density"},
                {h.bin_centre, h.empirical_density, h.mp_density});
    std::printf("  wrote %s (%zu bins)\n", out.c_str(), bins);

    if (args.has("spectrum-out")) {
        const std::string sp = args.str("spectrum-out", "");
        std::vector<double> index(r.spectrum.size());
        std::vector<double> cdf(r.spectrum.size());
        for (std::size_t i = 0; i < r.spectrum.size(); ++i) {
            index[i] = static_cast<double>(i + 1);
            cdf[i]   = mp_cdf(r.spectrum[i], p);
        }
        write_table(sp, {"rank", "eigenvalue", "mp_cdf"}, {index, r.spectrum, cdf});
        std::printf("  wrote %s\n", sp.c_str());
    }

    // ---- convergence sweep ------------------------------------------------
    // q is held fixed while N grows, which is the limit MP describes.
    std::printf("\nConvergence in N at fixed q = %.4f (MP is an asymptotic law)\n", q);
    std::printf("  %6s %8s %12s %10s %12s %12s %10s\n", "N", "T", "KS", "p-value", "lambda_max",
                "lambda+", "syev_ms");

    std::vector<double> sw_n, sw_t, sw_q, sw_ks, sw_p, sw_lmax, sw_lmin, sw_lplus, sw_ms;
    for (std::size_t i = 0; i < sweep_ns.size(); ++i) {
        const std::size_t ni = sweep_ns[i];
        const std::size_t ti = static_cast<std::size_t>(static_cast<double>(ni) / q + 0.5);
        if (ni == 0 || ti < 2) continue;

        // Vary the seed per size so the sweep is not one lucky draw repeated.
        const NullRun ri = run_null(ni, ti, seed + 1000 * (i + 1));
        std::printf("  %6zu %8zu %12.6f %10.4f %12.6f %12.6f %10.1f\n", ri.n, ri.t, ri.ks,
                    ri.pvalue, ri.lambda_max, mp_support(MpParams{ri.q, 1.0}).hi, ri.syev_ms);

        sw_n.push_back(static_cast<double>(ri.n));
        sw_t.push_back(static_cast<double>(ri.t));
        sw_q.push_back(ri.q);
        sw_ks.push_back(ri.ks);
        sw_p.push_back(ri.pvalue);
        sw_lmax.push_back(ri.lambda_max);
        sw_lmin.push_back(ri.lambda_min);
        sw_lplus.push_back(mp_support(MpParams{ri.q, 1.0}).hi);
        sw_ms.push_back(ri.syev_ms);
    }

    write_table(sweep_out,
                {"N", "T", "q", "ks", "pvalue", "lambda_max", "lambda_min", "lambda_plus",
                 "syev_ms"},
                {sw_n, sw_t, sw_q, sw_ks, sw_p, sw_lmax, sw_lmin, sw_lplus, sw_ms});
    std::printf("\n  wrote %s\n", sweep_out.c_str());

    if (sw_ks.size() >= 2) {
        std::printf("\n  KS shrank from %.6f (N=%.0f) to %.6f (N=%.0f): ratio %.3f\n", sw_ks.front(),
                    sw_n.front(), sw_ks.back(), sw_n.back(), sw_ks.back() / sw_ks.front());
    }
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "rnf_mp_null: %s\n", e.what());
    return 1;
}
