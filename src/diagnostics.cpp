#include "rnf/diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "rnf/gemm.hpp"
#include "rnf/qr.hpp"
#include "rnf/rng.hpp"
#include "rnf/syev.hpp"

namespace rnf {

RsvdDiagnostics diagnose(ConstMatView c, const RsvdResult& r) {
    const std::size_t n = c.rows;
    const std::size_t k = r.values.size();
    if (r.vectors.rows() != n || r.vectors.cols() != k) {
        throw std::invalid_argument("diagnose: eigenvector block does not match C and values");
    }

    RsvdDiagnostics d;
    d.orthogonality = orthogonality_error(r.vectors);

    Matrix cu(n, k);
    gemm(Op::NoTrans, Op::NoTrans, 1.0, c, r.vectors, 0.0, cu.view());

    d.residuals.resize(k);
    for (std::size_t j = 0; j < k; ++j) {
        double acc = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double e = cu(i, j) - r.values[j] * r.vectors(i, j);
            acc += e * e;
        }
        const double scale = std::max(std::fabs(r.values[j]), 1e-300);
        d.residuals[j] = std::sqrt(acc) / scale;
        d.max_residual = std::max(d.max_residual, d.residuals[j]);
    }
    return d;
}

SubspaceComparison compare_to_reference(const std::vector<double>& ref_values_desc,
                                        ConstMatView ref_vectors, const RsvdResult& got) {
    const std::size_t k = got.values.size();
    if (ref_values_desc.size() < k) {
        throw std::invalid_argument("compare_to_reference: reference has only " +
                                    std::to_string(ref_values_desc.size()) +
                                    " eigenvalues, need " + std::to_string(k));
    }
    if (ref_vectors.rows != got.vectors.rows()) {
        throw std::invalid_argument("compare_to_reference: dimension mismatch");
    }

    SubspaceComparison cmp;
    cmp.relative_errors.resize(k);
    for (std::size_t i = 0; i < k; ++i) {
        const double abs_err = std::fabs(got.values[i] - ref_values_desc[i]);
        const double scale   = std::max(std::fabs(ref_values_desc[i]), 1e-300);
        cmp.relative_errors[i] = abs_err / scale;
        cmp.max_absolute_eigenvalue_error = std::max(cmp.max_absolute_eigenvalue_error, abs_err);
        cmp.max_relative_eigenvalue_error =
            std::max(cmp.max_relative_eigenvalue_error, cmp.relative_errors[i]);
    }

    // Largest principal angle, via the Bjorck-Golub SINE formulation:
    //
    //     sin(theta_max) = || (I - U_ref U_ref^T) U_got ||_2
    //
    // The obvious route -- singular values of U_ref^T U_got are cos(theta_i),
    // so take sqrt(1 - cos^2_min) -- cannot resolve small angles at all. For
    // a well-converged subspace cos^2 is within rounding of 1, the
    // subtraction cancels catastrophically, and the computed sine bottoms out
    // at sqrt(eps) ~ 1.5e-8 no matter how accurate the vectors really are.
    // Projecting first and measuring the residual keeps full precision.
    const std::size_t n = got.vectors.rows();
    ConstMatView ref_block{ref_vectors.data, ref_vectors.rows, k, ref_vectors.ld};

    Matrix proj(k, k);
    gemm(Op::Trans, Op::NoTrans, 1.0, ref_block, got.vectors, 0.0, proj.view());

    Matrix resid(n, k);
    for (std::size_t j = 0; j < k; ++j) std::copy_n(got.vectors.col(j), n, resid.col(j));
    gemm(Op::NoTrans, Op::NoTrans, -1.0, ref_block, proj, 1.0, resid.view());

    // ||R||_2 = sqrt(largest eigenvalue of R^T R); R^T R is only k x k.
    Matrix rtr(k, k);
    gemm(Op::Trans, Op::NoTrans, 1.0, resid, resid, 0.0, rtr.view());
    for (std::size_t j = 0; j < k; ++j) {
        for (std::size_t i = j + 1; i < k; ++i) {
            const double v = 0.5 * (rtr(i, j) + rtr(j, i));
            rtr(i, j) = v;
            rtr(j, i) = v;
        }
    }
    const std::vector<double> sq = syev_values(rtr);  // descending
    cmp.sin_theta_max = std::sqrt(std::clamp(sq.front(), 0.0, 1.0));
    return cmp;
}

double estimate_projection_error(ConstMatView c, ConstMatView q, std::size_t probes,
                                 std::uint64_t seed) {
    if (probes == 0) throw std::invalid_argument("estimate_projection_error: need >= 1 probe");
    const std::size_t n = c.rows;
    const std::size_t l = q.cols;

    Rng    rng(seed);
    Matrix w  = gaussian_matrix(n, probes, rng);
    Matrix cw(n, probes);
    gemm(Op::NoTrans, Op::NoTrans, 1.0, c, w, 0.0, cw.view());

    // (I - Q Q^T) C w = C w - Q (Q^T C w)
    Matrix qt_cw(l, probes);
    gemm(Op::Trans, Op::NoTrans, 1.0, q, cw, 0.0, qt_cw.view());
    gemm(Op::NoTrans, Op::NoTrans, -1.0, q, qt_cw, 1.0, cw.view());

    double worst = 0.0;
    for (std::size_t j = 0; j < probes; ++j) {
        double acc = 0.0;
        for (std::size_t i = 0; i < n; ++i) acc += cw(i, j) * cw(i, j);
        worst = std::max(worst, std::sqrt(acc));
    }
    return worst;
}

}  // namespace rnf
