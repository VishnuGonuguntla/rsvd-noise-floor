// Phase 6 deliverable: point the whole pipeline at a real returns file.
//
// Read returns, standardize, extract the leading eigenpairs with RSVD
// (route B -- the correlation matrix is never formed), fit the
// Marchenko-Pastur noise floor from the data, and report only the factors
// that clear it.
//
// The interesting output is not the eigenvalues but the eigenVECTORS. The
// inverse participation ratio separates the two kinds of factor by
// inspection: a market mode loads a little on everything, so 1/IPR is close
// to N, while a sector mode concentrates on a handful of names and 1/IPR
// collapses toward the size of that handful. Everything below the floor is
// discarded without apology -- the BBP result in rnf_spiked is the argument
// that nothing recoverable is being thrown away.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "rnf/cli.hpp"
#include "rnf/csv.hpp"
#include "rnf/diagnostics.hpp"
#include "rnf/mp.hpp"
#include "rnf/rsvd.hpp"
#include "rnf/stats.hpp"

using namespace rnf;

namespace {

void usage() {
    std::printf(
        "rnf_analyze -- find the factors in a returns file that clear the noise floor\n"
        "\n"
        "  --csv <path>        returns CSV (header row, optional leading date column)\n"
        "  --no-dates          the file has no date column\n"
        "  --k <int>           eigenpairs to extract (default: min(N-1, 30))\n"
        "  --iters <int>       RSVD power iterations     (default 20)\n"
        "  --top <int>         names listed per factor   (default 6)\n"
        "  --floor <fitted|naive>  which cut to apply   (default fitted)\n"
        "  --seed <int>        RNG seed                  (default 1)\n"
        "  --out <path>        factor summary CSV (default results/factors.csv)\n"
        "  --loadings <path>   per-name loadings CSV (default results/loadings.csv)\n");
}

}  // namespace

int main(int argc, char** argv) try {
    const Args args(argc, argv);
    if (args.has("help")) { usage(); return 0; }

    const std::string path = args.str("csv", "data/sample_returns.csv");
    const bool        dates = !args.has("no-dates");
    const std::size_t top   = args.count("top", 6);
    const std::size_t iters = args.count("iters", 20);
    const std::uint64_t seed = args.seed("seed", 1);
    const std::string out      = args.str("out", "results/factors.csv");
    const std::string loadings = args.str("loadings", "results/loadings.csv");

    ReturnsData data = read_returns_csv(path, dates);
    const std::size_t t = data.values.rows();
    const std::size_t n = data.values.cols();
    if (n < 2) throw std::runtime_error("need at least 2 assets");
    if (t < n + 2) {
        std::printf("note: T = %zu is not much larger than N = %zu; q = %.3f means the\n"
                    "      noise floor is wide and few factors will clear it.\n\n", t, n,
                    aspect_ratio(t, n));
    }

    const double    q = aspect_ratio(t, n);
    const MpParams  p{q, 1.0};
    const MpSupport sup = mp_support(p);

    std::printf("rnf_analyze: %s\n", path.c_str());
    std::printf("  %zu observations x %zu assets", t, n);
    if (!data.dates.empty()) {
        std::printf("   (%s .. %s)", data.dates.front().c_str(), data.dates.back().c_str());
    }
    std::printf("\n  q = N/T = %.6f\n", q);
    std::printf("  naive bulk edge lambda+ = %.6f, BBP threshold = %.6f\n\n", sup.hi,
                bbp_threshold(q));

    // Standardize in place; route B works directly on Z and never forms C.
    standardize_columns(data.values.view());

    RsvdOptions opt;
    opt.rank             = args.count("k", std::min<std::size_t>(n - 1, 30));
    opt.oversampling     = std::min<std::size_t>(12, n - opt.rank);
    opt.power_iterations = iters;
    opt.seed             = seed;

    const RsvdResult r = rsvd_from_returns(data.values, opt);
    std::printf("RSVD route B (C never formed): rank %zu, %zu power iterations, %.1f ms%s\n",
                opt.rank, opt.power_iterations, r.elapsed_ms,
                r.used_dense_fallback ? "   [dense fallback: sketch >= N]" : "");

    // The noise floor is fitted from the spectrum and the exact trace; the
    // number of real factors is never supplied.
    const NoiseFloor nf = fit_noise_floor(r.values, n, q);

    std::size_t k_naive = 0;
    while (k_naive < r.values.size() && r.values[k_naive] > sup.hi) ++k_naive;

    std::printf("\nNoise floor: two cuts\n");
    std::printf("  naive  (sigma^2 = 1)       lambda+ = %.6f  ->  %zu factors\n", sup.hi,
                k_naive);
    std::printf("  fitted (sigma^2 = %.6f) lambda+ = %.6f  ->  %zu factors%s\n", nf.sigma2,
                nf.lambda_plus, nf.k_signal,
                nf.saturated ? "   [saturated: raise --k]" : "");

    // The two cuts disagreeing is informative, not a bug to paper over.
    //
    // The fitted floor is self-consistent: it estimates the bulk variance
    // from the trace left after removing the signal, which is the only thing
    // available when the bulk is never computed. But that makes it a fixed
    // point with a failure mode -- absorbing one bulk eigenvalue lowers
    // sigma^2, which lowers the cut, which absorbs the next. At small N,
    // where the finite-size edge fluctuation is comparable to the gaps
    // between bulk eigenvalues, it can run away like this.
    //
    // The margin column below is what to read: a genuine factor clears the
    // floor by a wide multiple, a runaway one clears it by a few percent. A
    // Tracy-Widom threshold would replace this judgement call with a p-value;
    // that is the natural next extension and is not implemented here.
    if (nf.k_signal > k_naive + 1) {
        std::printf("\n  WARNING: the fitted cut accepts %zu factors to the naive cut's %zu.\n"
                    "  At N = %zu the edge fluctuation is roughly N^(-2/3) = %.1f%%, comparable\n"
                    "  to the spacing near the edge, so the fitted cut is likely running away.\n"
                    "  Treat small-margin factors below as noise; --floor naive is the\n"
                    "  conservative choice here.\n",
                    nf.k_signal, k_naive, n,
                    100.0 * std::pow(static_cast<double>(n), -2.0 / 3.0));
    }

    const std::string floor_choice = args.str("floor", "fitted");
    double            cut = nf.lambda_plus;
    std::size_t       k_signal = nf.k_signal;
    if (floor_choice == "naive") {
        cut = sup.hi;
        k_signal = k_naive;
    } else if (floor_choice != "fitted") {
        throw std::invalid_argument("--floor must be 'fitted' or 'naive', got '" +
                                    floor_choice + "'");
    }
    std::printf("\n  using the %s cut (lambda+ = %.6f)\n", floor_choice.c_str(), cut);

    if (k_signal == 0) {
        std::printf("\nNothing clears the noise floor: this panel is consistent with\n"
                    "pure noise at q = %.4f.\n", q);
        return 0;
    }

    // ---- per-factor report -------------------------------------------------
    std::printf("\n  %4s %12s %10s %9s %12s %10s  %s\n", "rank", "eigenvalue", "%var",
                "margin", "1/IPR", "kind", "largest loadings");

    std::vector<double> c_rank, c_eig, c_var, c_part, c_ipr, c_margin;
    std::vector<std::vector<std::string>> load_rows;

    for (std::size_t i = 0; i < k_signal; ++i) {
        const std::span<const double> v{r.vectors.col(i), n};
        const double part = participation_count(v);
        const double pct  = 100.0 * r.values[i] / static_cast<double>(n);

        // Label by delocalization only -- how many names the mode actually
        // spreads over -- rather than by guessed financial meaning. The
        // leading delocalized mode is conventionally read as the market and
        // localized ones as sectors or themes, but that reading is the
        // analyst's job, not the solver's.
        const double dn = static_cast<double>(n);
        const char*  kind = (part > 0.60 * dn) ? "delocalized"
                          : (part > 0.20 * dn) ? "mixed"
                                               : "localized";

        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::sort(order.begin(), order.end(), [&](std::size_t x, std::size_t y) {
            return std::fabs(v[x]) > std::fabs(v[y]);
        });

        std::string names;
        for (std::size_t j = 0; j < std::min(top, n); ++j) {
            if (j) names += " ";
            char buf[64];
            std::snprintf(buf, sizeof buf, "%s(%+.2f)", data.names[order[j]].c_str(),
                          v[order[j]]);
            names += buf;
        }

        // How far above the cut, as a multiple. A real factor clears by a
        // wide margin; one the fitted cut swallowed clears by a few percent.
        const double margin = r.values[i] / cut;
        std::printf("  %4zu %12.4f %9.2f%% %8.2fx %12.1f %10s  %s%s\n", i + 1, r.values[i],
                    pct, margin, part, kind, names.c_str(),
                    margin < 1.5 ? "   <- marginal" : "");

        c_rank.push_back(static_cast<double>(i + 1));
        c_eig.push_back(r.values[i]);
        c_var.push_back(pct);
        c_part.push_back(part);
        c_ipr.push_back(inverse_participation_ratio(v));
        c_margin.push_back(r.values[i] / cut);
    }

    write_table(out,
                {"rank", "eigenvalue", "pct_variance", "margin_over_cut",
                 "participation_count", "ipr"},
                {c_rank, c_eig, c_var, c_margin, c_part, c_ipr});

    // Full loadings for the surviving factors, one row per asset.
    {
        std::vector<std::string> headers{"asset"};
        for (std::size_t i = 0; i < k_signal; ++i) {
            headers.push_back("factor" + std::to_string(i + 1));
        }
        for (std::size_t j = 0; j < n; ++j) {
            std::vector<std::string> row{data.names[j]};
            for (std::size_t i = 0; i < k_signal; ++i) {
                char buf[32];
                std::snprintf(buf, sizeof buf, "%.8f", r.vectors(j, i));
                row.emplace_back(buf);
            }
            load_rows.push_back(std::move(row));
        }
        write_rows(loadings, headers, load_rows);
    }
    std::printf("\n  wrote %s and %s\n", out.c_str(), loadings.c_str());

    // ---- what was discarded ------------------------------------------------
    if (k_signal < r.values.size()) {
        const TraceCheck tc = trace_conservation_check(r.values, k_signal, n, q,
                                                       static_cast<double>(n));
        double retained = 0.0;
        for (std::size_t i = 0; i < k_signal; ++i) retained += r.values[i];

        std::printf("\nWhat was discarded (the bulk is never computed)\n");
        std::printf("  tr(C) = %zu exactly; top-%zu hold %.4f (%.2f%% of total variance)\n", n,
                    k_signal, retained, 100.0 * retained / static_cast<double>(n));
        std::printf("  remaining %zu modes carry %.4f, i.e. %.6f each on average\n",
                    n - k_signal, tc.discarded_mass, tc.sigma2_eff);
        std::printf("  implied floor %.6f vs lambda_{k+1} = %.6f  ->  %s\n", tc.lambda_plus_eff,
                    tc.next_eigenvalue,
                    tc.consistent ? "remainder IS consistent with MP noise"
                                  : "remainder still holds signal: raise --k");
    }

    const RsvdDiagnostics d = diagnose(correlation_matrix(data.values), r);
    double sig_resid = 0.0;
    for (std::size_t i = 0; i < k_signal; ++i) {
        sig_resid = std::max(sig_resid, d.residuals[i]);
    }
    std::printf("\nAccuracy of the reported factors\n");
    std::printf("  max residual ||C u - lambda u|| / lambda = %.3e\n", sig_resid);
    std::printf("  orthogonality ||U^T U - I||_F            = %.3e\n", d.orthogonality);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "rnf_analyze: %s\n", e.what());
    return 1;
}
