// Phase 5 deliverable: is the randomized method actually faster, and by how
// much?
//
// The claim under test is that extracting the top-k eigenpairs of an N x N
// correlation matrix costs O(N^2 k) by randomization versus O(N^3) densely,
// so the advantage grows linearly in N. Three things are measured:
//
//   1. Thread scaling of the correlation build, which is a symmetric rank-k
//      update and the single most expensive step in the whole pipeline.
//   2. The main table: forming C, RSVD route A, RSVD route B, and a dense
//      spectrum, across N -- with GFLOP/s from analytic flop counts, and a
//      correctness check comparing the two routes at every size, because a
//      benchmark that does not verify its output measures nothing.
//   3. The packed AVX2 kernel against the naive triple loop it replaced.
//
// The dense baseline is eigenvalues ONLY. That is the cheapest way to get a
// full spectrum, so it makes the speedup figures conservative: RSVD returns
// eigenvectors too, and the dense equivalent costs more again.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "rnf/cli.hpp"
#include "rnf/csv.hpp"
#include "rnf/gemm.hpp"
#include "rnf/matrix.hpp"
#include "rnf/rng.hpp"
#include "rnf/rsvd.hpp"
#include "rnf/stats.hpp"
#include "rnf/syev.hpp"
#include "rnf/threadpool.hpp"
#include "rnf/timer.hpp"

using namespace rnf;

namespace {

double gflops(double flops, double ms) {
    return (ms <= 0.0) ? 0.0 : flops / (ms * 1.0e6);
}

void usage() {
    std::printf(
        "rnf_bench -- randomized vs dense, and kernel throughput\n"
        "\n"
        "  --sizes <list>    N values, comma-separated (default 200,400,800,1600,3200)\n"
        "  --q <float>       aspect ratio N/T          (default 0.25)\n"
        "  --k <int>         target rank               (default 20)\n"
        "  --p <int>         oversampling              (default 10)\n"
        "  --iters <int>     power iterations          (default 2)\n"
        "  --reps <int>      timed repeats             (default 5)\n"
        "  --dense-max <int> skip the dense baseline above this N (default 2000)\n"
        "  --seed <int>      RNG seed                  (default 3)\n"
        "  --out <path>      main table CSV   (default results/bench.csv)\n"
        "  --threads-out <p> scaling CSV      (default results/bench_threads.csv)\n");
}

}  // namespace

int main(int argc, char** argv) try {
    const Args args(argc, argv);
    if (args.has("help")) { usage(); return 0; }

    const auto          sizes = args.counts("sizes", {200, 400, 800, 1600, 3200});
    const double        q     = args.real("q", 0.25);
    const std::size_t   k     = args.count("k", 20);
    const std::size_t   p     = args.count("p", 10);
    const std::size_t   iters = args.count("iters", 2);
    const std::size_t   reps  = std::max<std::size_t>(1, args.count("reps", 5));
    const std::size_t   dense_max = args.count("dense-max", 2000);
    const std::uint64_t seed  = args.seed("seed", 3);
    const std::string   out   = args.str("out", "results/bench.csv");
    const std::string   thr_out = args.str("threads-out", "results/bench_threads.csv");

    if (!(q > 0.0)) throw std::invalid_argument("--q must be positive");

    const unsigned hw = std::thread::hardware_concurrency();
    std::printf("rnf_bench\n");
    std::printf("  hardware threads  : %u\n", hw);
    std::printf("  rank k = %zu, oversampling p = %zu, sketch l = %zu, power iterations = %zu\n",
                k, p, k + p, iters);
    std::printf("  aspect ratio q = %.4f, median of %zu timed repeats\n\n", q, reps);

    // ---- 1. thread scaling of the correlation build -----------------------
    {
        const std::size_t n = 1500;
        const std::size_t t = static_cast<std::size_t>(static_cast<double>(n) / q + 0.5);
        Rng    rng(seed);
        Matrix z = gaussian_matrix(t, n, rng);
        standardize_columns(z.view());
        Matrix c(n, n);

        std::printf("Thread scaling: C = Z^T Z / T at N = %zu, T = %zu (%.2f GFLOP)\n", n, t,
                    syrk_flops(t, n) * 1e-9);
        std::printf("  %8s %12s %12s %10s %12s\n", "threads", "median_ms", "GFLOP/s", "speedup",
                    "efficiency");

        std::vector<double> col_thr, col_ms, col_gf, col_sp;
        double base_ms = 0.0;
        for (std::size_t nthreads = 1; nthreads <= hw; nthreads *= 2) {
            ThreadPool::resize_global(nthreads);
            const TimingStats ts = time_repeats(reps, 1, [&] {
                syrk_trans(1.0 / static_cast<double>(t), z, 0.0, c.view());
            });
            if (base_ms == 0.0) base_ms = ts.median_ms;
            const double sp = base_ms / ts.median_ms;
            std::printf("  %8zu %12.2f %12.2f %10.2fx %11.0f%%\n", nthreads, ts.median_ms,
                        gflops(syrk_flops(t, n), ts.median_ms), sp,
                        100.0 * sp / static_cast<double>(nthreads));
            col_thr.push_back(static_cast<double>(nthreads));
            col_ms.push_back(ts.median_ms);
            col_gf.push_back(gflops(syrk_flops(t, n), ts.median_ms));
            col_sp.push_back(sp);
        }
        write_table(thr_out, {"threads", "median_ms", "gflops", "speedup"},
                    {col_thr, col_ms, col_gf, col_sp});
        std::printf("\n  wrote %s\n", thr_out.c_str());
        ThreadPool::resize_global(0);  // back to all available threads
    }

    // ---- 2. randomized vs dense across N ----------------------------------
    std::printf("\nRandomized vs dense (dense baseline is eigenvalues only, so the\n"
                "speedups below are conservative -- RSVD also returns eigenvectors)\n\n");
    std::printf("  %6s %8s | %10s %10s %10s %10s | %9s %9s | %8s %8s | %10s\n", "N", "T",
                "formC_ms", "rsvdA_ms", "rsvdB_ms", "dense_ms", "A_GF/s", "B_GF/s", "A_vs_D",
                "B_vs_D", "A/B_agree");

    std::vector<double> c_n, c_t, c_form, c_a, c_b, c_d, c_agf, c_bgf, c_spa, c_spb, c_err;

    for (std::size_t n : sizes) {
        const std::size_t t = static_cast<std::size_t>(static_cast<double>(n) / q + 0.5);
        if (n < k + p) {
            std::printf("  %6zu  (skipped: N < k + p)\n", n);
            continue;
        }

        Rng    rng(seed + n);
        Matrix z = gaussian_matrix(t, n, rng);
        standardize_columns(z.view());

        Matrix c(n, n);
        const TimingStats t_form = time_repeats(reps, 1, [&] {
            syrk_trans(1.0 / static_cast<double>(t), z, 0.0, c.view());
        });
        for (std::size_t i = 0; i < n; ++i) c(i, i) = 1.0;

        RsvdOptions opt;
        opt.rank = k;
        opt.oversampling = p;
        opt.power_iterations = iters;
        opt.seed = seed;
        const std::size_t sketch = std::min(k + p, n);

        RsvdResult ra, rb;
        const TimingStats t_a = time_repeats(reps, 1, [&] { ra = rsvd_symmetric(c, opt); });
        const TimingStats t_b = time_repeats(reps, 1, [&] { rb = rsvd_from_returns(z, opt); });

        // Correctness, every size: the two routes compute the same thing by
        // different arithmetic, so agreement is a real check and not a
        // formality.
        double max_rel = 0.0;
        for (std::size_t i = 0; i < k; ++i) {
            const double scale = std::max(std::fabs(ra.values[i]), 1e-300);
            max_rel = std::max(max_rel, std::fabs(ra.values[i] - rb.values[i]) / scale);
        }

        double dense_ms = 0.0;
        if (n <= dense_max) {
            const TimingStats t_d =
                time_repeats(std::max<std::size_t>(1, reps / 2), 0, [&] {
                    (void)syev_values(c);
                });
            dense_ms = t_d.median_ms;
        }

        const double fa = rsvd_flops_route_a(n, sketch, iters);
        const double fb = rsvd_flops_route_b(t, n, sketch, iters);

        std::printf("  %6zu %8zu | %10.2f %10.2f %10.2f %10s | %9.2f %9.2f | %7.1fx %7.1fx | %10.2e\n",
                    n, t, t_form.median_ms, t_a.median_ms, t_b.median_ms,
                    dense_ms > 0.0 ? (std::to_string(static_cast<long long>(dense_ms + 0.5))).c_str()
                                   : "-",
                    gflops(fa, t_a.median_ms), gflops(fb, t_b.median_ms),
                    dense_ms > 0.0 ? dense_ms / t_a.median_ms : 0.0,
                    dense_ms > 0.0 ? dense_ms / t_b.median_ms : 0.0, max_rel);

        c_n.push_back(static_cast<double>(n));
        c_t.push_back(static_cast<double>(t));
        c_form.push_back(t_form.median_ms);
        c_a.push_back(t_a.median_ms);
        c_b.push_back(t_b.median_ms);
        c_d.push_back(dense_ms);
        c_agf.push_back(gflops(fa, t_a.median_ms));
        c_bgf.push_back(gflops(fb, t_b.median_ms));
        c_spa.push_back(dense_ms > 0.0 ? dense_ms / t_a.median_ms : 0.0);
        c_spb.push_back(dense_ms > 0.0 ? dense_ms / t_b.median_ms : 0.0);
        c_err.push_back(max_rel);
    }

    write_table(out,
                {"N", "T", "form_c_ms", "rsvd_a_ms", "rsvd_b_ms", "dense_ms", "rsvd_a_gflops",
                 "rsvd_b_gflops", "speedup_a", "speedup_b", "route_ab_max_rel_diff"},
                {c_n, c_t, c_form, c_a, c_b, c_d, c_agf, c_bgf, c_spa, c_spb, c_err});
    std::printf("\n  wrote %s\n", out.c_str());

    std::printf("\n  Note route B never forms C. Its column does NOT include form_c_ms,\n"
                "  while route A must pay it -- so the fair end-to-end comparison for\n"
                "  starting from returns is (form_c_ms + rsvd_a_ms) against rsvd_b_ms.\n");

    // ---- 3. kernel throughput ---------------------------------------------
    {
        std::printf("\nPacked AVX2 kernel vs the naive triple-loop oracle\n");
        std::printf("  %14s %12s %12s %12s %10s\n", "shape", "naive_ms", "packed_ms",
                    "packed_GF/s", "speedup");

        struct Shape { std::size_t m, n, kk; const char* label; };
        const Shape shapes[] = {
            {512, 512, 512, "512^3 square"},
            {2000, 30, 2000, "2000x30x2000"},   // the shape RSVD actually runs
        };

        for (const Shape& sh : shapes) {
            Rng    rng(11);
            Matrix a = gaussian_matrix(sh.m, sh.kk, rng);
            Matrix b = gaussian_matrix(sh.kk, sh.n, rng);
            Matrix cf(sh.m, sh.n), cn(sh.m, sh.n);

            const TimingStats t_fast = time_repeats(reps, 1, [&] {
                gemm(Op::NoTrans, Op::NoTrans, 1.0, a, b, 0.0, cf.view());
            });
            const TimingStats t_slow = time_repeats(1, 0, [&] {
                gemm_naive(Op::NoTrans, Op::NoTrans, 1.0, a, b, 0.0, cn.view());
            });

            double worst = 0.0;
            for (std::size_t j = 0; j < sh.n; ++j) {
                for (std::size_t i = 0; i < sh.m; ++i) {
                    worst = std::max(worst, std::fabs(cf(i, j) - cn(i, j)));
                }
            }
            const double f = gemm_flops(sh.m, sh.n, sh.kk);
            std::printf("  %14s %12.1f %12.2f %12.2f %9.1fx   (max |diff| %.2e)\n", sh.label,
                        t_slow.median_ms, t_fast.median_ms, gflops(f, t_fast.median_ms),
                        t_slow.median_ms / t_fast.median_ms, worst);
        }
    }

    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "rnf_bench: %s\n", e.what());
    return 1;
}
