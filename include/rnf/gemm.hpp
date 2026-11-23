// Dense matrix multiply.
//
// Two implementations live here for the life of the project:
//   gemm_naive -- a transparent triple loop, kept permanently as the
//                 correctness oracle for every optimized path.
//   gemm       -- the production kernel (blocked/packed AVX2 + thread pool).
// Every optimization is validated against the oracle, which is why the oracle
// is public API rather than a test-local helper.
#pragma once

#include <cstddef>

#include "matrix.hpp"

namespace rnf {

enum class Op { NoTrans, Trans };

// Logical shape of op(A).
[[nodiscard]] inline std::size_t op_rows(Op o, ConstMatView a) {
    return o == Op::NoTrans ? a.rows : a.cols;
}
[[nodiscard]] inline std::size_t op_cols(Op o, ConstMatView a) {
    return o == Op::NoTrans ? a.cols : a.rows;
}

// C := alpha * op(A) * op(B) + beta * C
void gemm(Op op_a, Op op_b, double alpha, ConstMatView a, ConstMatView b,
          double beta, MatView c);

// Same contract, deliberately unoptimized. The reference answer.
void gemm_naive(Op op_a, Op op_b, double alpha, ConstMatView a, ConstMatView b,
                double beta, MatView c);

// C := alpha * A^T * A + beta * C, with C symmetric (n x n, n = A.cols).
// Computes the lower triangle and mirrors it, so this costs roughly half of
// the equivalent gemm. Used to form the correlation matrix.
void syrk_trans(double alpha, ConstMatView a, double beta, MatView c);

// Multiply-add count for C(m x n) += A(m x k) * B(k x n): 2*m*n*k.
[[nodiscard]] constexpr double gemm_flops(std::size_t m, std::size_t n, std::size_t k) {
    return 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
}

// A^T A for A(m x n) touches only the triangle: m*n*(n+1) flops.
[[nodiscard]] constexpr double syrk_flops(std::size_t m, std::size_t n) {
    return static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(n + 1);
}

}  // namespace rnf
