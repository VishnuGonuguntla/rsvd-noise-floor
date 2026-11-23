#include "rnf/qr.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "rnf/gemm.hpp"
#include "rnf/householder.hpp"

namespace rnf {

void householder_qr(MatView a, std::span<double> tau) {
    const std::size_t m = a.rows;
    const std::size_t n = a.cols;
    if (m < n) {
        throw std::invalid_argument("householder_qr: need m >= n, got " + std::to_string(m) +
                                    "x" + std::to_string(n));
    }
    if (tau.size() < n) throw std::invalid_argument("householder_qr: tau too small");

    for (std::size_t j = 0; j < n; ++j) {
        double* xj = a.col(j) + j;          // A(j:m, j)
        const std::size_t len = m - j;
        tau[j] = householder::reflector(xj, len);

        if (j + 1 < n && tau[j] != 0.0) {
            // Apply to the trailing block A(j:m, j+1:n). v(0) is implicit 1,
            // so the saved beta at A(j,j) must not be read as part of v.
            MatView trailing = a.block(j, j + 1, len, n - j - 1);
            householder::apply_left(trailing, xj + 1, tau[j]);
        }
    }
}

void form_thin_q(MatView a, std::span<const double> tau) {
    const std::size_t m = a.rows;
    const std::size_t n = a.cols;
    if (tau.size() < n) throw std::invalid_argument("form_thin_q: tau too small");

    // LAPACK dorg2r: accumulate backwards so each reflector sees the already
    // built trailing block.
    for (std::size_t jj = n; jj-- > 0;) {
        const std::size_t j   = jj;
        const std::size_t len = m - j;
        double*           vj  = a.col(j) + j;

        if (j + 1 < n) {
            const double saved = vj[0];
            vj[0] = 1.0;  // restore the implicit unit entry while applying
            MatView trailing = a.block(j, j + 1, len, n - j - 1);
            householder::apply_left(trailing, vj + 1, tau[j]);
            vj[0] = saved;
        }

        for (std::size_t i = 1; i < len; ++i) vj[i] *= -tau[j];
        vj[0] = 1.0 - tau[j];
        for (std::size_t i = 0; i < j; ++i) a(i, j) = 0.0;
    }
}

Matrix extract_r(ConstMatView a) {
    const std::size_t n = a.cols;
    Matrix r(n, n);
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i <= j; ++i) r(i, j) = a(i, j);
    }
    return r;
}

void orthonormalize(Matrix& a) {
    if (a.empty()) return;
    std::vector<double> tau(a.cols(), 0.0);
    householder_qr(a.view(), tau);
    form_thin_q(a.view(), tau);
}

double orthogonality_error(ConstMatView q) {
    const std::size_t n = q.cols;
    Matrix gram(n, n);
    gemm(Op::Trans, Op::NoTrans, 1.0, q, q, 0.0, gram.view());
    double acc = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) {
            const double target = (i == j) ? 1.0 : 0.0;
            const double d      = gram(i, j) - target;
            acc += d * d;
        }
    }
    return std::sqrt(acc);
}

}  // namespace rnf
