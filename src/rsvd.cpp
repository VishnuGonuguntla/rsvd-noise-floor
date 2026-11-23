#include "rnf/rsvd.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "rnf/gemm.hpp"
#include "rnf/qr.hpp"
#include "rnf/rng.hpp"
#include "rnf/syev.hpp"
#include "rnf/timer.hpp"

namespace rnf {
namespace {

void validate(const RsvdOptions& opt, std::size_t n) {
    if (opt.rank == 0) throw std::invalid_argument("rsvd: rank must be positive");
    if (opt.rank > n) {
        throw std::invalid_argument("rsvd: rank " + std::to_string(opt.rank) +
                                    " exceeds matrix dimension " + std::to_string(n));
    }
}

// Rayleigh-Ritz on the small projected matrix. M must be symmetric in exact
// arithmetic; it is symmetrized explicitly first because Q^T (C Q) is not
// bit-symmetric, and feeding an asymmetric matrix to a symmetric eigensolver
// yields silently wrong eigenvectors.
RsvdResult ritz_finish(Matrix& m, const Matrix& q, std::size_t rank) {
    const std::size_t l = m.rows();
    for (std::size_t j = 0; j < l; ++j) {
        for (std::size_t i = j + 1; i < l; ++i) {
            const double v = 0.5 * (m(i, j) + m(j, i));
            m(i, j) = v;
            m(j, i) = v;
        }
    }

    SyevResult small = syev(m, SyevMode::ValuesAndVectors);

    RsvdResult out;
    out.sketch_size = l;
    out.values.assign(small.values.begin(),
                      small.values.begin() + static_cast<std::ptrdiff_t>(rank));

    // U = Q * V, keeping only the leading `rank` Ritz vectors.
    Matrix v_top(l, rank);
    for (std::size_t j = 0; j < rank; ++j) {
        std::copy_n(small.vectors.col(j), l, v_top.col(j));
    }
    out.vectors = Matrix(q.rows(), rank);
    gemm(Op::NoTrans, Op::NoTrans, 1.0, q, v_top, 0.0, out.vectors.view());
    return out;
}

// When the sketch would be as wide as the matrix, randomization buys nothing
// and costs accuracy: just solve it densely.
RsvdResult dense_fallback(ConstMatView c, std::size_t rank) {
    SyevResult full = syev(c, SyevMode::ValuesAndVectors);
    RsvdResult out;
    out.used_dense_fallback = true;
    out.sketch_size         = c.rows;
    out.values.assign(full.values.begin(),
                      full.values.begin() + static_cast<std::ptrdiff_t>(rank));
    out.vectors = Matrix(c.rows, rank);
    for (std::size_t j = 0; j < rank; ++j) {
        std::copy_n(full.vectors.col(j), c.rows, out.vectors.col(j));
    }
    return out;
}

}  // namespace

RsvdResult rsvd_symmetric(ConstMatView c, const RsvdOptions& opt) {
    if (c.rows != c.cols) {
        throw std::invalid_argument("rsvd_symmetric: matrix must be square, got " +
                                    std::to_string(c.rows) + "x" + std::to_string(c.cols));
    }
    const std::size_t n = c.rows;
    validate(opt, n);

    Stopwatch         sw;
    const std::size_t l = std::min(opt.rank + opt.oversampling, n);

    if (l >= n) {
        RsvdResult out = dense_fallback(c, opt.rank);
        out.power_iterations = 0;
        out.elapsed_ms       = sw.elapsed_ms();
        return out;
    }

    Rng    rng(opt.seed);
    Matrix omega = gaussian_matrix(n, l, rng);

    // Y = C * Omega, then an orthonormal basis for its column space.
    Matrix y(n, l);
    gemm(Op::NoTrans, Op::NoTrans, 1.0, c, omega, 0.0, y.view());
    orthonormalize(y);

    // Subspace iteration. Re-orthonormalizing after EVERY application of C is
    // not optional: with a geometrically decaying spectrum, skipping it costs
    // roughly log10(kappa) digits per step and the sketch collapses onto the
    // single dominant direction.
    Matrix work(n, l);
    for (std::size_t it = 0; it < opt.power_iterations; ++it) {
        gemm(Op::NoTrans, Op::NoTrans, 1.0, c, y, 0.0, work.view());
        std::swap(y, work);
        orthonormalize(y);
    }

    // M = Q^T C Q
    gemm(Op::NoTrans, Op::NoTrans, 1.0, c, y, 0.0, work.view());  // work = C Q
    Matrix m(l, l);
    gemm(Op::Trans, Op::NoTrans, 1.0, y, work, 0.0, m.view());

    RsvdResult out       = ritz_finish(m, y, opt.rank);
    out.power_iterations = opt.power_iterations;
    out.elapsed_ms       = sw.elapsed_ms();
    return out;
}

RsvdResult rsvd_from_returns(ConstMatView z, const RsvdOptions& opt) {
    const std::size_t t = z.rows;
    const std::size_t n = z.cols;
    if (t < 2) throw std::invalid_argument("rsvd_from_returns: need at least 2 observations");
    validate(opt, n);

    Stopwatch         sw;
    const std::size_t l = std::min(opt.rank + opt.oversampling, n);

    if (l >= n) {
        // Forming C is only acceptable here because we are about to spend
        // O(N^3) on it anyway.
        Matrix c(n, n);
        syrk_trans(1.0 / static_cast<double>(t), z, 0.0, c.view());
        RsvdResult out       = dense_fallback(c, opt.rank);
        out.power_iterations = 0;
        out.elapsed_ms       = sw.elapsed_ms();
        return out;
    }

    Rng    rng(opt.seed);
    Matrix omega = gaussian_matrix(n, l, rng);

    // One application of C, without ever forming it: Z^T (Z v). The 1/T is
    // left out because scaling does not change the column space -- it is
    // applied once at the end, on the small projected matrix.
    Matrix zq(t, l);
    Matrix y(n, l);

    gemm(Op::NoTrans, Op::NoTrans, 1.0, z, omega, 0.0, zq.view());
    gemm(Op::Trans, Op::NoTrans, 1.0, z, zq, 0.0, y.view());
    orthonormalize(y);

    for (std::size_t it = 0; it < opt.power_iterations; ++it) {
        gemm(Op::NoTrans, Op::NoTrans, 1.0, z, y, 0.0, zq.view());
        gemm(Op::Trans, Op::NoTrans, 1.0, z, zq, 0.0, y.view());
        orthonormalize(y);
    }

    // M = (ZQ)^T (ZQ) / T, which is exactly Q^T C Q.
    gemm(Op::NoTrans, Op::NoTrans, 1.0, z, y, 0.0, zq.view());
    Matrix m(l, l);
    syrk_trans(1.0 / static_cast<double>(t), zq, 0.0, m.view());

    RsvdResult out       = ritz_finish(m, y, opt.rank);
    out.power_iterations = opt.power_iterations;
    out.elapsed_ms       = sw.elapsed_ms();
    return out;
}

double rsvd_flops_route_a(std::size_t n, std::size_t sketch, std::size_t power_iterations) {
    const double dn = static_cast<double>(n);
    const double dl = static_cast<double>(sketch);
    const double applications = static_cast<double>(power_iterations) + 2.0;  // initial + loop + final
    const double big   = applications * 2.0 * dn * dn * dl;
    const double qr    = (static_cast<double>(power_iterations) + 1.0) * 4.0 * dn * dl * dl;
    const double small = 2.0 * dn * dl * dl   // M = Q^T (CQ)
                       + 2.0 * dn * dl * dl;  // U = Q V
    return big + qr + small;
}

double rsvd_flops_route_b(std::size_t t, std::size_t n, std::size_t sketch,
                          std::size_t power_iterations) {
    const double dt = static_cast<double>(t);
    const double dn = static_cast<double>(n);
    const double dl = static_cast<double>(sketch);
    // Each application of Z^T Z is two gemms of 2*T*N*l.
    const double applications = static_cast<double>(power_iterations) + 1.0;
    const double big   = applications * 4.0 * dt * dn * dl
                       + 2.0 * dt * dn * dl;  // the final ZQ
    const double qr    = applications * 4.0 * dn * dl * dl;
    const double small = dt * dl * (dl + 1.0)   // syrk
                       + 2.0 * dn * dl * dl;    // U = Q V
    return big + qr + small;
}

}  // namespace rnf
