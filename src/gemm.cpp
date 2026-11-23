#include "rnf/gemm.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "rnf/threadpool.hpp"

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define RNF_HAVE_AVX2 1
#endif

namespace rnf {
namespace {

// Micro-kernel shape. 8x6 accumulators in __m256d are 2*6 = 12 vector
// registers, leaving room for two A vectors and a broadcast B in the 16 ymm
// registers AVX2 provides. Going wider spills; going narrower wastes the FMA
// units. (This machine has AVX2 + FMA but no AVX-512: 4 doubles per vector.)
constexpr std::size_t MR = 8;
constexpr std::size_t NR = 6;

// Cache blocking. KC*NR*8B of packed B stays in L1; MC*KC*8B of packed A
// targets L2; NC is the sweep width over C's columns.
constexpr std::size_t MC = 192;
constexpr std::size_t KC = 256;
constexpr std::size_t NC = 1024;

// syrk sweeps C's columns in narrower strips than gemm does. The triangular
// skip works at whole-tile granularity, so a wide strip means most of the
// first strip is computed above the diagonal and thrown away by the mirror
// pass. Narrowing it recovers most of the factor-of-two that symmetry is
// supposed to buy, at the cost of re-streaming the packed B panel more often.
constexpr std::size_t NC_SYRK = 256;

// Below this, packing costs more than it saves.
constexpr std::size_t kSmallThreshold = 48;

void check_shapes(Op op_a, Op op_b, ConstMatView a, ConstMatView b, MatView c) {
    const std::size_t m  = op_rows(op_a, a);
    const std::size_t ka = op_cols(op_a, a);
    const std::size_t kb = op_rows(op_b, b);
    const std::size_t n  = op_cols(op_b, b);
    if (ka != kb || c.rows != m || c.cols != n) {
        throw std::invalid_argument(
            "gemm: shape mismatch, op(A) is " + std::to_string(m) + "x" + std::to_string(ka) +
            ", op(B) is " + std::to_string(kb) + "x" + std::to_string(n) +
            ", C is " + std::to_string(c.rows) + "x" + std::to_string(c.cols));
    }
}

void scale_c(double beta, MatView c) {
    if (beta == 1.0) return;
    for (std::size_t j = 0; j < c.cols; ++j) {
        double* cj = c.col(j);
        if (beta == 0.0) {
            std::fill_n(cj, c.rows, 0.0);
        } else {
            for (std::size_t i = 0; i < c.rows; ++i) cj[i] *= beta;
        }
    }
}

// ---------------------------------------------------------------- kernel

// C[0:MR, 0:NR] += alpha * Ap * Bp, where Ap is kc blocks of MR contiguous
// doubles and Bp is kc blocks of NR contiguous doubles.
void micro_kernel(const double* ap, const double* bp, std::size_t kc, double* c,
                  std::size_t ldc, double alpha) {
#if RNF_HAVE_AVX2
    __m256d acc[2][NR];
    for (std::size_t j = 0; j < NR; ++j) {
        acc[0][j] = _mm256_setzero_pd();
        acc[1][j] = _mm256_setzero_pd();
    }

    for (std::size_t p = 0; p < kc; ++p) {
        const __m256d a0 = _mm256_load_pd(ap);
        const __m256d a1 = _mm256_load_pd(ap + 4);
        ap += MR;
        for (std::size_t j = 0; j < NR; ++j) {
            const __m256d bv = _mm256_broadcast_sd(bp + j);
            acc[0][j] = _mm256_fmadd_pd(a0, bv, acc[0][j]);
            acc[1][j] = _mm256_fmadd_pd(a1, bv, acc[1][j]);
        }
        bp += NR;
    }

    const __m256d va = _mm256_set1_pd(alpha);
    for (std::size_t j = 0; j < NR; ++j) {
        double* cj = c + j * ldc;
        _mm256_storeu_pd(cj, _mm256_fmadd_pd(va, acc[0][j], _mm256_loadu_pd(cj)));
        _mm256_storeu_pd(cj + 4, _mm256_fmadd_pd(va, acc[1][j], _mm256_loadu_pd(cj + 4)));
    }
#else
    double acc[MR][NR] = {};
    for (std::size_t p = 0; p < kc; ++p) {
        for (std::size_t i = 0; i < MR; ++i) {
            for (std::size_t j = 0; j < NR; ++j) acc[i][j] += ap[p * MR + i] * bp[p * NR + j];
        }
    }
    for (std::size_t j = 0; j < NR; ++j) {
        for (std::size_t i = 0; i < MR; ++i) c[j * ldc + i] += alpha * acc[i][j];
    }
#endif
}

// ---------------------------------------------------------------- packing

// Pack an mc x kc block of op(A) into MR-row panels: panel-major, and within
// a panel, kc groups of MR contiguous doubles. Partial panels are zero-padded
// so the kernel never needs an edge case.
void pack_a(Op op_a, ConstMatView a, std::size_t i0, std::size_t p0, std::size_t mc,
            std::size_t kc, double* ap) {
    const std::size_t panels = (mc + MR - 1) / MR;
    for (std::size_t ip = 0; ip < panels; ++ip) {
        double*           dst  = ap + ip * MR * kc;
        const std::size_t rows = std::min(MR, mc - ip * MR);
        for (std::size_t p = 0; p < kc; ++p) {
            for (std::size_t i = 0; i < rows; ++i) {
                const std::size_t gi = i0 + ip * MR + i;
                const std::size_t gp = p0 + p;
                dst[p * MR + i] = (op_a == Op::NoTrans) ? a(gi, gp) : a(gp, gi);
            }
            for (std::size_t i = rows; i < MR; ++i) dst[p * MR + i] = 0.0;
        }
    }
}

// Same for op(B), in NR-column panels.
void pack_b(Op op_b, ConstMatView b, std::size_t p0, std::size_t j0, std::size_t kc,
            std::size_t nc, double* bp) {
    const std::size_t panels = (nc + NR - 1) / NR;
    for (std::size_t jp = 0; jp < panels; ++jp) {
        double*           dst  = bp + jp * NR * kc;
        const std::size_t cols = std::min(NR, nc - jp * NR);
        for (std::size_t p = 0; p < kc; ++p) {
            for (std::size_t j = 0; j < cols; ++j) {
                const std::size_t gj = j0 + jp * NR + j;
                const std::size_t gp = p0 + p;
                dst[p * NR + j] = (op_b == Op::NoTrans) ? b(gp, gj) : b(gj, gp);
            }
            for (std::size_t j = cols; j < NR; ++j) dst[p * NR + j] = 0.0;
        }
    }
}

// ---------------------------------------------------------------- macro

// One MC x NC tile of C against the packed panels.
void macro_kernel(const double* ap, const double* bp, std::size_t mc, std::size_t nc,
                  std::size_t kc, MatView c, std::size_t i0, std::size_t j0, double alpha,
                  double* edge) {
    const std::size_t mpanels = (mc + MR - 1) / MR;
    const std::size_t npanels = (nc + NR - 1) / NR;

    for (std::size_t jp = 0; jp < npanels; ++jp) {
        const std::size_t cols = std::min(NR, nc - jp * NR);
        for (std::size_t ip = 0; ip < mpanels; ++ip) {
            const std::size_t rows = std::min(MR, mc - ip * MR);
            const double*     apanel = ap + ip * MR * kc;
            const double*     bpanel = bp + jp * NR * kc;

            if (rows == MR && cols == NR) {
                micro_kernel(apanel, bpanel, kc, &c(i0 + ip * MR, j0 + jp * NR), c.ld, alpha);
            } else {
                // Partial tile: accumulate into a full-size scratch block and
                // copy back only the live part, so the kernel stays uniform.
                std::fill_n(edge, MR * NR, 0.0);
                micro_kernel(apanel, bpanel, kc, edge, MR, 1.0);
                for (std::size_t j = 0; j < cols; ++j) {
                    double* cj = &c(i0 + ip * MR, j0 + jp * NR + j);
                    for (std::size_t i = 0; i < rows; ++i) cj[i] += alpha * edge[j * MR + i];
                }
            }
        }
    }
}

// Straightforward loops, for shapes too small to be worth packing. Loop
// orders chosen so the inner loop is unit-stride in column-major storage.
void gemm_small(Op op_a, Op op_b, double alpha, ConstMatView a, ConstMatView b, MatView c) {
    const std::size_t m = c.rows;
    const std::size_t n = c.cols;
    const std::size_t k = op_cols(op_a, a);

    if (op_a == Op::NoTrans && op_b == Op::NoTrans) {
        for (std::size_t j = 0; j < n; ++j) {
            double* cj = c.col(j);
            for (std::size_t p = 0; p < k; ++p) {
                const double bpj = alpha * b(p, j);
                if (bpj == 0.0) continue;
                const double* apcol = a.col(p);
                for (std::size_t i = 0; i < m; ++i) cj[i] += bpj * apcol[i];
            }
        }
    } else if (op_a == Op::Trans && op_b == Op::NoTrans) {
        for (std::size_t j = 0; j < n; ++j) {
            const double* bj = b.col(j);
            double*       cj = c.col(j);
            for (std::size_t i = 0; i < m; ++i) {
                const double* ai = a.col(i);
                double acc = 0.0;
                for (std::size_t p = 0; p < k; ++p) acc += ai[p] * bj[p];
                cj[i] += alpha * acc;
            }
        }
    } else if (op_a == Op::NoTrans && op_b == Op::Trans) {
        for (std::size_t j = 0; j < n; ++j) {
            double* cj = c.col(j);
            for (std::size_t p = 0; p < k; ++p) {
                const double bjp = alpha * b(j, p);
                if (bjp == 0.0) continue;
                const double* apcol = a.col(p);
                for (std::size_t i = 0; i < m; ++i) cj[i] += bjp * apcol[i];
            }
        }
    } else {
        for (std::size_t j = 0; j < n; ++j) {
            double* cj = c.col(j);
            for (std::size_t i = 0; i < m; ++i) {
                const double* ai = a.col(i);
                double acc = 0.0;
                for (std::size_t p = 0; p < k; ++p) acc += ai[p] * b(j, p);
                cj[i] += alpha * acc;
            }
        }
    }
}

// Per-thread packing scratch, reused across calls.
//
// Allocating these per parallel_for was costing more than the arithmetic at
// high thread counts: a single correlation build issues dozens of blocked
// passes, each of which would otherwise hand every worker a fresh ~400 KB
// buffer. Grow-only, thread_local, so steady-state allocation is zero.
struct PackScratch {
    detail::AlignedPtr a;
    detail::AlignedPtr b;
    detail::AlignedPtr edge;
    std::size_t        cap_a = 0;
    std::size_t        cap_b = 0;

    void ensure(std::size_t need_a, std::size_t need_b) {
        if (cap_a < need_a) { a = detail::aligned_doubles(need_a); cap_a = need_a; }
        if (cap_b < need_b) { b = detail::aligned_doubles(need_b); cap_b = need_b; }
        if (!edge) edge = detail::aligned_doubles(MR * NR);
    }
};

PackScratch& scratch() {
    thread_local PackScratch s;
    return s;
}

// Rows handled per packed A panel. Parallel work is split at MR-row
// granularity rather than MC-block granularity: at N = 1500 there are only
// eight MC blocks, so half of a 16-thread pool would sit idle, and the
// triangular skip in syrk makes what is left badly imbalanced.
std::size_t a_panel_doubles(std::size_t mc) { return ((mc + MR - 1) / MR) * MR * KC; }
std::size_t b_panel_doubles(std::size_t nc) { return ((nc + NR - 1) / NR) * NR * KC; }

// Row-block size for the dynamic work queue. Aiming at roughly two blocks
// per worker keeps every thread fed without re-streaming the packed B panel
// more often than necessary; the MR multiple keeps blocks kernel-aligned and
// the MC ceiling preserves the L2 blocking.
std::size_t dynamic_block_rows(std::size_t m, std::size_t workers) {
    if (workers <= 1) return MC;
    std::size_t want = m / (2 * workers);
    want = (want / MR) * MR;
    return std::clamp(want, std::size_t{64}, MC);
}

}  // namespace

void gemm_naive(Op op_a, Op op_b, double alpha, ConstMatView a, ConstMatView b,
                double beta, MatView c) {
    check_shapes(op_a, op_b, a, b, c);
    const std::size_t m = c.rows;
    const std::size_t n = c.cols;
    const std::size_t k = op_cols(op_a, a);

    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < m; ++i) {
            double acc = 0.0;
            for (std::size_t p = 0; p < k; ++p) {
                const double av = (op_a == Op::NoTrans) ? a(i, p) : a(p, i);
                const double bv = (op_b == Op::NoTrans) ? b(p, j) : b(j, p);
                acc += av * bv;
            }
            const double prev = (beta == 0.0) ? 0.0 : beta * c(i, j);
            c(i, j) = prev + alpha * acc;
        }
    }
}

void gemm(Op op_a, Op op_b, double alpha, ConstMatView a, ConstMatView b, double beta,
          MatView c) {
    check_shapes(op_a, op_b, a, b, c);
    const std::size_t m = c.rows;
    const std::size_t n = c.cols;
    const std::size_t k = op_cols(op_a, a);

    if (m == 0 || n == 0) return;
    scale_c(beta, c);
    if (k == 0 || alpha == 0.0) return;

    if (m < kSmallThreshold && n < kSmallThreshold && k < kSmallThreshold) {
        gemm_small(op_a, op_b, alpha, a, b, c);
        return;
    }

    ThreadPool&       pool      = ThreadPool::global();
    const std::size_t blk_rows  = dynamic_block_rows(m, pool.size());

    for (std::size_t j0 = 0; j0 < n; j0 += NC) {
        const std::size_t nc = std::min(NC, n - j0);
        for (std::size_t p0 = 0; p0 < k; p0 += KC) {
            const std::size_t kc = std::min(KC, k - p0);

            // Parallelize over row blocks of C: for the tall-skinny shapes
            // RSVD produces (m = N, n = k + p ~ 30) m is the only dimension
            // with enough work to split. Each worker packs its own copy of
            // the B panel -- duplicated, but B is tiny here next to the cost
            // of streaming A.
            // Workers pull row blocks from a shared counter instead of being
            // handed equal slices up front. On this machine's hybrid cores
            // that is the difference between being paced by the slowest
            // E-core and keeping every P-core busy.
            std::atomic<std::size_t> next_block{0};
            pool.parallel_invoke([&](std::size_t) {
                PackScratch& buf = scratch();
                buf.ensure(a_panel_doubles(blk_rows), b_panel_doubles(nc));
                pack_b(op_b, b, p0, j0, kc, nc, buf.b.get());

                for (;;) {
                    const std::size_t blk = next_block.fetch_add(1, std::memory_order_relaxed);
                    const std::size_t i0  = blk * blk_rows;
                    if (i0 >= m) break;
                    const std::size_t mc = std::min(blk_rows, m - i0);
                    pack_a(op_a, a, i0, p0, mc, kc, buf.a.get());
                    macro_kernel(buf.a.get(), buf.b.get(), mc, nc, kc, c, i0, j0, alpha,
                                 buf.edge.get());
                }
            });
        }
    }
}

void syrk_trans(double alpha, ConstMatView a, double beta, MatView c) {
    const std::size_t m = a.rows;
    const std::size_t n = a.cols;
    if (c.rows != n || c.cols != n) {
        throw std::invalid_argument("syrk_trans: C must be " + std::to_string(n) + "x" +
                                    std::to_string(n) + ", got " + std::to_string(c.rows) +
                                    "x" + std::to_string(c.cols));
    }
    if (n == 0 || m == 0) return;

    scale_c(beta, c);

    if (n < kSmallThreshold) {
        for (std::size_t j = 0; j < n; ++j) {
            const double* aj = a.col(j);
            for (std::size_t i = j; i < n; ++i) {
                const double* ai = a.col(i);
                double acc = 0.0;
                for (std::size_t p = 0; p < m; ++p) acc += ai[p] * aj[p];
                c(i, j) += alpha * acc;
            }
        }
    } else {
        // C = A^T A is symmetric, so only tiles on or below the diagonal are
        // computed -- roughly half the work of the equivalent gemm. The
        // diagonal tiles are computed in full and trimmed by the mirror pass
        // below, which is simpler than a triangular micro-kernel and costs
        // only O(n * MC).
        ThreadPool&       pool     = ThreadPool::global();
        const std::size_t blk_rows = dynamic_block_rows(n, pool.size());

        for (std::size_t j0 = 0; j0 < n; j0 += NC_SYRK) {
            const std::size_t nc = std::min(NC_SYRK, n - j0);
            // Only rows at or below the diagonal block need computing, so the
            // work queue starts at j0 rather than 0 -- otherwise the first
            // workers to pull would get nothing but skipped tiles.
            const std::size_t i_start = (j0 / blk_rows) * blk_rows;

            for (std::size_t p0 = 0; p0 < m; p0 += KC) {
                const std::size_t kc = std::min(KC, m - p0);

                std::atomic<std::size_t> next_block{0};
                pool.parallel_invoke([&](std::size_t) {
                    PackScratch& buf = scratch();
                    buf.ensure(a_panel_doubles(blk_rows), b_panel_doubles(nc));
                    pack_b(Op::NoTrans, a, p0, j0, kc, nc, buf.b.get());

                    for (;;) {
                        const std::size_t blk = next_block.fetch_add(1, std::memory_order_relaxed);
                        const std::size_t i0  = i_start + blk * blk_rows;
                        if (i0 >= n) break;
                        const std::size_t mc = std::min(blk_rows, n - i0);
                        pack_a(Op::Trans, a, i0, p0, mc, kc, buf.a.get());
                        macro_kernel(buf.a.get(), buf.b.get(), mc, nc, kc, c, i0, j0, alpha,
                                     buf.edge.get());
                    }
                });
            }
        }
    }

    // Mirror the lower triangle up. Also repairs the parts of the diagonal
    // tiles that were computed above the diagonal.
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = j + 1; i < n; ++i) c(j, i) = c(i, j);
    }
}

}  // namespace rnf
