// Randomized SVD of a symmetric positive semi-definite matrix.
//
// The Halko-Martinsson-Tropp range finder, specialized for C = C^T and
// finished with Rayleigh-Ritz rather than the usual B = Q^T A route. For a
// symmetric matrix the SVD would return |lambda| and lose sign information,
// while Rayleigh-Ritz returns real eigenpairs by construction and costs fewer
// flops.
//
// Why this is the right algorithm for a correlation matrix: the number of
// eigenvalues above the Marchenko-Pastur noise floor is small (5 to 30 even
// for N in the thousands), so C is a low-rank signal piece sitting on an MP
// bulk. A full eigendecomposition is O(N^3) and spends nearly all of it
// resolving noise; this is O(N^2 k). And unusually for a low-rank method, the
// rank is not a guess -- mp.hpp supplies it, and the trace-conservation check
// confirms after the fact that what was discarded really was noise.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rnf/matrix.hpp"

namespace rnf {

struct RsvdOptions {
    std::size_t   rank             = 10;  // k: how many eigenpairs are wanted
    std::size_t   oversampling     = 10;  // p: extra sketch columns, discarded at the end
    std::size_t   power_iterations = 2;   // q: subspace iterations
    std::uint64_t seed             = 1;
};

struct RsvdResult {
    std::vector<double> values;   // top-k eigenvalues, descending
    Matrix              vectors;  // N x k, orthonormal columns
    std::size_t         sketch_size      = 0;
    std::size_t         power_iterations = 0;
    bool                used_dense_fallback = false;
    double              elapsed_ms = 0.0;
};

// Route A: sketch the correlation matrix directly. Needs C formed, which
// itself costs N^2 T.
[[nodiscard]] RsvdResult rsvd_symmetric(ConstMatView c, const RsvdOptions& opt);

// Route B: sketch the standardized returns Z (T x N) and never form C.
//
// Preferable whenever N is large: it avoids the N^2 T cost of building C, and
// it is numerically better besides -- forming C = Z^T Z / T squares the
// condition number of exactly the part we care about. Eigenvalues of C are
// recovered as the eigenvalues of (ZQ)^T (ZQ) / T.
//
// `z` must already be standardized (see stats.hpp); the 1/T is applied here.
[[nodiscard]] RsvdResult rsvd_from_returns(ConstMatView z, const RsvdOptions& opt);

// Analytic flop counts, for turning benchmark times into GFLOP/s.
[[nodiscard]] double rsvd_flops_route_a(std::size_t n, std::size_t sketch,
                                        std::size_t power_iterations);
[[nodiscard]] double rsvd_flops_route_b(std::size_t t, std::size_t n, std::size_t sketch,
                                        std::size_t power_iterations);

}  // namespace rnf
