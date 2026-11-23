// Householder reflector primitives, shared by the QR factorization and the
// symmetric tridiagonal reduction.
//
// Convention throughout (LAPACK's): a reflector is H = I - tau * v * v^T with
// v(0) == 1 implicit, so only the tail v(1:) is ever stored. That lets the
// reflectors live in the subdiagonal part of the matrix being factored.
#pragma once

#include <cmath>
#include <cstddef>

#include "rnf/matrix.hpp"

namespace rnf::householder {

// Build H so that H * x = (beta, 0, ..., 0)^T. On return x[0] holds beta and
// x[1:] holds the reflector tail v(1:). Returns tau; tau == 0 means x was
// already a multiple of e1 and x is left untouched.
//
// beta takes the sign opposite to x[0] so that (alpha - beta) never cancels --
// the single most important detail for numerical stability here.
inline double reflector(double* x, std::size_t len) {
    if (len <= 1) return 0.0;

    const double alpha = x[0];
    double tail_sq = 0.0;
    for (std::size_t i = 1; i < len; ++i) tail_sq += x[i] * x[i];
    if (tail_sq == 0.0) return 0.0;

    const double norm = std::sqrt(alpha * alpha + tail_sq);
    const double beta = (alpha >= 0.0) ? -norm : norm;
    const double tau  = (beta - alpha) / beta;
    const double scal = 1.0 / (alpha - beta);
    for (std::size_t i = 1; i < len; ++i) x[i] *= scal;
    x[0] = beta;
    return tau;
}

// block := H * block, with v(0) == 1 implicit and v(1:) read from vtail.
inline void apply_left(MatView block, const double* vtail, double tau) {
    if (tau == 0.0) return;
    for (std::size_t j = 0; j < block.cols; ++j) {
        double* cj  = block.col(j);
        double  dot = cj[0];  // v(0) == 1
        for (std::size_t i = 1; i < block.rows; ++i) dot += vtail[i - 1] * cj[i];
        const double f = tau * dot;
        cj[0] -= f;
        for (std::size_t i = 1; i < block.rows; ++i) cj[i] -= f * vtail[i - 1];
    }
}

// Symmetric similarity transform S := H * S * H for symmetric S (both
// triangles populated), with v(0) == 1 implicit and v(1:) from vtail.
//
// Expanding H S H gives S - v*u^T - u*v^T + tau*(v.u)*v*v^T with u = tau*S*v;
// folding the last term into w = u - (tau/2)*(v.u)*v turns the whole thing
// into a symmetric rank-2 update, which is both cheaper and exactly
// symmetry-preserving in floating point.
inline void apply_similarity(MatView s, const double* vtail, double tau,
                             double* u_work, double* w_work) {
    if (tau == 0.0) return;
    const std::size_t m = s.rows;

    auto v_at = [&](std::size_t i) { return (i == 0) ? 1.0 : vtail[i - 1]; };

    // u = tau * S * v
    for (std::size_t i = 0; i < m; ++i) u_work[i] = 0.0;
    for (std::size_t j = 0; j < m; ++j) {
        const double vj = v_at(j);
        if (vj == 0.0) continue;
        const double* sj = s.col(j);
        for (std::size_t i = 0; i < m; ++i) u_work[i] += sj[i] * vj;
    }
    for (std::size_t i = 0; i < m; ++i) u_work[i] *= tau;

    double vu = 0.0;
    for (std::size_t i = 0; i < m; ++i) vu += v_at(i) * u_work[i];
    const double half = 0.5 * tau * vu;
    for (std::size_t i = 0; i < m; ++i) w_work[i] = u_work[i] - half * v_at(i);

    // S -= v*w^T + w*v^T, applied over the full block so symmetry is exact.
    for (std::size_t j = 0; j < m; ++j) {
        double*      sj = s.col(j);
        const double wj = w_work[j];
        const double vj = v_at(j);
        for (std::size_t i = 0; i < m; ++i) {
            sj[i] -= v_at(i) * wj + w_work[i] * vj;
        }
    }
}

}  // namespace rnf::householder
