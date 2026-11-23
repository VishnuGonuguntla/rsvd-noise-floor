// Phase 4 deliverable: plant known factors, recover them, and map the BBP
// detectability threshold.
//
// Two experiments:
//
//  1. Recovery. Plant a few factors of known strength above the noise floor,
//     then check that RSVD finds exactly those, that the Marchenko-Pastur cut
//     selects the right rank without being told it, and that the discarded
//     mass is consistent with MP noise (trace conservation) -- the last
//     without ever computing the bulk.
//
//  2. The BBP sweep. Walk a single spike's strength down toward 1 + sqrt(q)
//     and watch it vanish: the eigenvalue sinks to the bulk edge and the
//     eigenvector decorrelates from the planted direction. Below that
//     threshold the factor is unrecoverable by any method, which is what
//     makes the lambda_plus cut the right place to stop.
//
// The sweep is also run at a deliberately low iteration count, because the
// two effects are easy to confuse: a spike near the threshold is BOTH
// statistically marginal AND numerically slow for randomized subspace
// iteration, since both follow from the same small spectral gap.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

#include "rnf/cli.hpp"
#include "rnf/csv.hpp"
#include "rnf/diagnostics.hpp"
#include "rnf/mp.hpp"
#include "rnf/rsvd.hpp"
#include "rnf/spiked.hpp"
#include "rnf/stats.hpp"

using namespace rnf;

namespace {

double overlap_sq(const double* u, const double* v, std::size_t n) {
    double d = 0.0, nu = 0.0, nv = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        d  += u[i] * v[i];
        nu += u[i] * u[i];
        nv += v[i] * v[i];
    }
    if (nu <= 0.0 || nv <= 0.0) return 0.0;
    return (d * d) / (nu * nv);
}

void usage() {
    std::printf(
        "rnf_spiked -- recover planted factors and map the BBP threshold\n"
        "\n"
        "  --N <int>             assets                      (default 1000)\n"
        "  --T <int>             observations                (default 4000)\n"
        "  --spikes <list>       planted strengths, comma-separated\n"
        "                                                    (default 25,9,4,2.5)\n"
        "  --sweep-spike lo:hi:step  BBP sweep range         (default 1.0:4.0:0.1)\n"
        "  --trials <int>        draws averaged per sweep point (default 3)\n"
        "  --iters <int>         RSVD power iterations       (default 30)\n"
        "  --cheap-iters <int>   under-iterated comparison   (default 3)\n"
        "  --seed <int>          RNG seed                    (default 7)\n"
        "  --out <path>          recovery CSV      (default results/spiked_recovery.csv)\n"
        "  --sweep-out <path>    BBP sweep CSV     (default results/bbp.csv)\n");
}

}  // namespace

int main(int argc, char** argv) try {
    const Args args(argc, argv);
    if (args.has("help")) { usage(); return 0; }

    const std::size_t   n     = args.count("N", 1000);
    const std::size_t   t     = args.count("T", 4000);
    const std::uint64_t seed  = args.seed("seed", 7);
    const std::size_t   iters = args.count("iters", 30);
    const std::size_t   cheap = args.count("cheap-iters", 3);
    const std::size_t   trials = std::max<std::size_t>(1, args.count("trials", 3));
    const std::string   out       = args.str("out", "results/spiked_recovery.csv");
    const std::string   sweep_out = args.str("sweep-out", "results/bbp.csv");

    std::vector<double> spikes;
    {
        const std::string spec = args.str("spikes", "25,9,4,2.5");
        std::string       item;
        std::stringstream ss(spec);
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) spikes.push_back(std::stod(item));
        }
    }
    const std::vector<double> sweep = args.range("sweep-spike", {});
    std::vector<double> sweep_vals = sweep;
    if (sweep_vals.empty()) {
        for (double s = 1.0; s <= 4.0 + 1e-12; s += 0.1) sweep_vals.push_back(s);
    }

    const double    q   = aspect_ratio(t, n);
    const MpParams  p{q, 1.0};
    const MpSupport sup = mp_support(p);
    const double    s_crit = bbp_threshold(q);

    std::printf("Spiked model: recovery and the BBP threshold\n");
    std::printf("  N = %zu, T = %zu, q = %.6f\n", n, t, q);
    std::printf("  naive bulk edge lambda+        = %.6f\n", sup.hi);
    std::printf("  BBP threshold  1 + sqrt(q)     = %.6f\n\n", s_crit);

    // ---- experiment 1: recovery -------------------------------------------
    std::printf("Planted factors:");
    for (double s : spikes) std::printf(" %.3f", s);
    std::printf("\n");

    const SpikedModel model = generate_spiked_returns(t, n, spikes, seed);
    Matrix            c     = correlation_from_returns(model.returns);

    // Ask for more than were planted, so lambda_{k+1} is available for the
    // trace-conservation check and the MP cut has something to reject.
    RsvdOptions opt;
    opt.rank             = std::min(n - 1, spikes.size() + 6);
    opt.oversampling     = 12;
    opt.power_iterations = iters;
    opt.seed             = seed + 1;

    const RsvdResult    r  = rsvd_symmetric(c, opt);
    const RsvdDiagnostics d = diagnose(c, r);

    std::printf("\nRSVD (rank %zu, oversampling %zu, %zu power iterations) in %.1f ms\n",
                opt.rank, opt.oversampling, opt.power_iterations, r.elapsed_ms);
    std::printf("  orthogonality ||U^T U - I||_F            = %.3e\n", d.orthogonality);
    {
        // Split the residual: the modes above the cut are the ones we claim,
        // and they converge. The requested-but-rejected modes reach into the
        // bulk, where eigenvectors are not well separated and a large
        // residual is the correct answer, not a failure.
        double sig = 0.0, blk = 0.0;
        for (std::size_t i = 0; i < d.residuals.size(); ++i) {
            if (i < spikes.size()) sig = std::max(sig, d.residuals[i]);
            else                   blk = std::max(blk, d.residuals[i]);
        }
        std::printf("  max residual, planted modes              = %.3e\n", sig);
        std::printf("  max residual, modes reaching into bulk   = %.3e"
                    "   (expected: not separated)\n", blk);
    }

    // The noise floor is fitted from the data -- the planted count is never
    // handed to it.
    const NoiseFloor nf = fit_noise_floor(r.values, n, q);
    std::printf("\nSelf-consistent noise floor (fitted, NOT told how many factors exist)\n");
    std::printf("  sigma^2_eff   = %.6f   (naive assumption would be 1.0)\n", nf.sigma2);
    std::printf("  lambda+_eff   = %.6f   (naive would be %.6f)\n", nf.lambda_plus, sup.hi);
    std::printf("  factors found = %zu     (planted %zu)%s\n", nf.k_signal, spikes.size(),
                nf.k_signal == spikes.size() ? "   <- correct" : "   <- MISMATCH");
    std::printf("  converged     = %s\n", nf.converged ? "yes" : "no");

    // A trap worth naming: BBP is stated for the COVARIANCE model, where the
    // bulk sits at 1. Standardizing to a correlation matrix divides each
    // asset by its own standard deviation -- which the spikes themselves
    // inflated -- so every eigenvalue is pulled down by roughly
    // sigma^2_eff = N / tr(Sigma). The sample covariance spectrum matches the
    // raw BBP prediction to three digits; the correlation spectrum does not,
    // and should not be expected to.
    //
    // This is exactly what the self-consistent sigma^2 fit absorbs, which is
    // why the rank selection above is still right.
    std::printf("\n  %5s %12s %12s %12s %12s %10s %10s\n", "rank", "planted_s", "bbp_pred",
                "rescaled", "recovered", "overlap^2", "above_cut");
    std::vector<double> col_rank, col_planted, col_pred, col_scaled, col_got, col_ov, col_above;
    for (std::size_t i = 0; i < r.values.size(); ++i) {
        const bool   planted   = i < spikes.size();
        const double s         = planted ? model.strengths[i] : 0.0;
        const double predicted = planted ? spike_to_eigenvalue(s, q) : 0.0;
        // The same prediction expressed in the correlation metric, where the
        // bulk sits at sigma^2_eff rather than at 1.
        const double rescaled  = planted ? nf.sigma2 * predicted : 0.0;
        const double ov        = planted
                                   ? overlap_sq(r.vectors.col(i), model.directions.col(i), n)
                                   : 0.0;
        const bool   above     = r.values[i] > nf.lambda_plus;

        char buf_s[32], buf_p[32], buf_r[32], buf_o[32];
        // Always format as a double; the "-" for unplanted rows is chosen at
        // print time below. A conditional format string here would be
        // undefined behaviour the moment the "%s" branch were taken.
        std::snprintf(buf_s, sizeof buf_s, "%.4f", s);
        std::snprintf(buf_p, sizeof buf_p, "%.4f", predicted);
        std::snprintf(buf_r, sizeof buf_r, "%.4f", rescaled);
        std::snprintf(buf_o, sizeof buf_o, "%.4f", ov);

        std::printf("  %5zu %12s %12s %12s %12.6f %10s %10s\n", i + 1,
                    planted ? buf_s : "-", planted ? buf_p : "-", planted ? buf_r : "-",
                    r.values[i], planted ? buf_o : "-", above ? "yes" : "no");

        col_rank.push_back(static_cast<double>(i + 1));
        col_planted.push_back(s);
        col_pred.push_back(predicted);
        col_scaled.push_back(rescaled);
        col_got.push_back(r.values[i]);
        col_ov.push_back(ov);
        col_above.push_back(above ? 1.0 : 0.0);
    }
    std::printf("\n  bbp_pred is the covariance-model prediction; rescaled multiplies it by\n"
                "  the fitted sigma^2_eff = %.4f to express it in the correlation metric.\n"
                "  The residual gap is the spread in diag(Sigma): standardization is not a\n"
                "  uniform rescaling when the planted directions load unevenly.\n", nf.sigma2);
    write_table(out,
                {"rank", "planted_strength", "bbp_prediction", "bbp_rescaled",
                 "recovered_eigenvalue", "overlap_sq", "above_cut"},
                {col_rank, col_planted, col_pred, col_scaled, col_got, col_ov, col_above});
    std::printf("\n  wrote %s\n", out.c_str());

    // Trace conservation: the bulk was never computed, yet we can still say
    // whether what we threw away was noise.
    const TraceCheck tc = trace_conservation_check(r.values, nf.k_signal, n, q,
                                                   static_cast<double>(n));
    std::printf("\nTrace conservation at k = %zu (the bulk is never computed)\n", nf.k_signal);
    std::printf("  tr(C)                 = %.6f   (exactly N)\n", static_cast<double>(n));
    std::printf("  retained by top-%zu     = %.6f\n", nf.k_signal,
                static_cast<double>(n) - tc.discarded_mass);
    std::printf("  discarded mass        = %.6f\n", tc.discarded_mass);
    std::printf("  implied sigma^2       = %.6f\n", tc.sigma2_eff);
    std::printf("  implied lambda+       = %.6f\n", tc.lambda_plus_eff);
    std::printf("  lambda_{k+1}          = %.6f\n", tc.next_eigenvalue);
    std::printf("  headroom              = %+.6f  -> %s\n", tc.headroom,
                tc.consistent ? "remainder IS consistent with MP noise"
                              : "remainder still contains signal");

    // ---- experiment 2: the BBP sweep --------------------------------------
    std::printf("\nBBP sweep: one spike, strength walked down to the threshold\n");
    std::printf("  averaging %zu draws per point; %zu power iterations"
                " (and %zu for the under-iterated comparison)\n",
                trials, iters, cheap);
    std::printf("\n  %8s %12s %12s %12s %12s %12s\n", "s", "lambda_th", "lambda_rsvd",
                "ov2_theory", "ov2_rsvd", "ov2_cheap");

    std::vector<double> sw_s, sw_lth, sw_lr, sw_oth, sw_or, sw_oc, sw_edge, sw_det;
    for (double s : sweep_vals) {
        double acc_l = 0.0, acc_o = 0.0, acc_oc = 0.0;
        for (std::size_t trial = 0; trial < trials; ++trial) {
            const SpikedModel m =
                generate_spiked_returns(t, n, {s}, seed + 1013 * (trial + 1));
            Matrix       ci = correlation_from_returns(m.returns);
            RsvdOptions  o;
            o.rank = 1;
            o.oversampling = 12;
            o.seed = seed + 2;

            o.power_iterations = iters;
            const RsvdResult good = rsvd_symmetric(ci, o);
            acc_l += good.values[0];
            acc_o += overlap_sq(good.vectors.col(0), m.directions.col(0), n);

            o.power_iterations = cheap;
            const RsvdResult meh = rsvd_symmetric(ci, o);
            acc_oc += overlap_sq(meh.vectors.col(0), m.directions.col(0), n);
        }
        const double dt  = static_cast<double>(trials);
        const double lr  = acc_l / dt;
        const double orr = acc_o / dt;
        const double oc  = acc_oc / dt;
        const double lth = spike_to_eigenvalue(s, q);
        const double oth = spike_eigenvector_overlap_sq(s, q);

        std::printf("  %8.3f %12.6f %12.6f %12.6f %12.6f %12.6f%s\n", s, lth, lr, oth, orr, oc,
                    std::fabs(s - s_crit) < 0.051 ? "   <- BBP threshold" : "");

        sw_s.push_back(s);
        sw_lth.push_back(lth);
        sw_lr.push_back(lr);
        sw_oth.push_back(oth);
        sw_or.push_back(orr);
        sw_oc.push_back(oc);
        sw_edge.push_back(sup.hi);
        sw_det.push_back(lr > sup.hi ? 1.0 : 0.0);
    }

    write_table(sweep_out,
                {"s", "lambda_theory", "lambda_rsvd", "overlap_theory", "overlap_rsvd",
                 "overlap_rsvd_cheap", "lambda_plus", "detected"},
                {sw_s, sw_lth, sw_lr, sw_oth, sw_or, sw_oc, sw_edge, sw_det});
    std::printf("\n  wrote %s\n", sweep_out.c_str());
    std::printf("\n  Below s = %.3f the planted factor is indistinguishable from noise:\n"
                "  no algorithm recovers it, so cutting at lambda+ discards nothing.\n", s_crit);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "rnf_spiked: %s\n", e.what());
    return 1;
}
