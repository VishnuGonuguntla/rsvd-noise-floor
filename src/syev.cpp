#include "rnf/syev.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "rnf/householder.hpp"

namespace rnf {
namespace {

constexpr int kMaxQlIterations = 50;

// Householder reduction of a symmetric matrix to tridiagonal form
// (LAPACK dsytd2). `a` must hold BOTH triangles of the symmetric input and is
// overwritten: the reflector tail for step k lands in a(k+2:n, k).
//
// Convention: d[k] is the diagonal, e[k] the subdiagonal element coupling
// d[k] and d[k+1], with e[n-1] == 0.
void tridiagonalize(MatView a, std::vector<double>& d, std::vector<double>& e,
                    std::vector<double>& tau) {
    const std::size_t n = a.rows;
    std::fill(d.begin(), d.end(), 0.0);
    std::fill(e.begin(), e.end(), 0.0);
    std::fill(tau.begin(), tau.end(), 0.0);

    if (n == 0) return;
    if (n == 1) { d[0] = a(0, 0); return; }

    std::vector<double> u_work(n, 0.0);
    std::vector<double> w_work(n, 0.0);

    for (std::size_t k = 0; k + 2 < n; ++k) {
        const std::size_t m = n - k - 1;  // length of the subcolumn / trailing block

        double* x = &a(k + 1, k);
        tau[k] = householder::reflector(x, m);
        const double beta = x[0];

        // Row/column k are now final: the tridiagonal entry plus zeros. The
        // reflector tail stays in a(k+2:n, k) for the Q accumulation, so only
        // the row side is zeroed here.
        d[k] = a(k, k);
        e[k] = beta;
        a(k, k + 1) = beta;
        for (std::size_t j = k + 2; j < n; ++j) a(k, j) = 0.0;

        MatView trailing = a.block(k + 1, k + 1, m, m);
        householder::apply_similarity(trailing, x + 1, tau[k], u_work.data(), w_work.data());
    }

    // Trailing 2x2 is already tridiagonal.
    d[n - 2] = a(n - 2, n - 2);
    e[n - 2] = a(n - 1, n - 2);
    d[n - 1] = a(n - 1, n - 1);
    e[n - 1] = 0.0;
}

// Q = H_0 H_1 ... H_{n-3} applied to the identity (LAPACK dorgtr).
// Accumulated backwards so each reflector acts on the block the later ones
// have already filled in. H_k is the identity on indices 0..k, which is why
// row 0 and column 0 of Q come out as e_0.
Matrix accumulate_q(ConstMatView factored, const std::vector<double>& tau) {
    const std::size_t n = factored.rows;
    Matrix q = Matrix::identity(n);
    if (n < 3) return q;

    for (std::size_t kk = n - 2; kk-- > 0;) {
        const std::size_t k = kk;
        const std::size_t m = n - k - 1;
        const double*     vtail = &factored(k + 2, k);
        MatView block = q.block(k + 1, k + 1, m, m);
        householder::apply_left(block, vtail, tau[k]);
    }
    return q;
}

// Implicit-shift QL with Wilkinson shift on a symmetric tridiagonal matrix
// (EISPACK tql2). `z` accumulates the plane rotations when vectors are wanted;
// e is consumed in the convention produced by tridiagonalize above.
void ql_iterate(std::vector<double>& d, std::vector<double>& e, MatView z, bool want_vectors) {
    using idx = std::ptrdiff_t;
    const idx    n   = static_cast<idx>(d.size());
    const double eps = std::numeric_limits<double>::epsilon();

    auto D = [&](idx i) -> double& { return d[static_cast<std::size_t>(i)]; };
    auto E = [&](idx i) -> double& { return e[static_cast<std::size_t>(i)]; };
    auto Z = [&](idx i, idx j) -> double& {
        return z(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
    };

    for (idx l = 0; l < n; ++l) {
        int iter = 0;
        idx m = l;
        do {
            // Split at the first negligible subdiagonal element.
            for (m = l; m + 1 < n; ++m) {
                const double dd = std::fabs(D(m)) + std::fabs(D(m + 1));
                if (std::fabs(E(m)) <= eps * dd) break;
            }

            if (m != l) {
                if (iter++ == kMaxQlIterations) {
                    throw std::runtime_error(
                        "syev: QL iteration failed to converge at index " + std::to_string(l));
                }

                double g = (D(l + 1) - D(l)) / (2.0 * E(l));
                double r = std::hypot(g, 1.0);
                g = D(m) - D(l) + E(l) / (g + std::copysign(r, g));

                double s = 1.0;
                double c = 1.0;
                double p = 0.0;
                bool   deflated = false;

                for (idx i = m - 1; i >= l; --i) {
                    double       f = s * E(i);
                    const double b = c * E(i);
                    r = std::hypot(f, g);
                    E(i + 1) = r;

                    if (r == 0.0) {
                        D(i + 1) -= p;
                        E(m) = 0.0;
                        deflated = true;
                        break;
                    }

                    s = f / r;
                    c = g / r;
                    g = D(i + 1) - p;
                    r = (D(i) - g) * s + 2.0 * c * b;
                    p = s * r;
                    D(i + 1) = g + p;
                    g = c * r - b;

                    if (want_vectors) {
                        for (idx row = 0; row < n; ++row) {
                            f = Z(row, i + 1);
                            Z(row, i + 1) = s * Z(row, i) + c * f;
                            Z(row, i)     = c * Z(row, i) - s * f;
                        }
                    }
                }

                if (deflated) continue;

                D(l) -= p;
                E(l) = g;
                E(m) = 0.0;
            }
        } while (m != l);
    }
}

}  // namespace

SyevResult syev(ConstMatView a, SyevMode mode) {
    if (a.rows != a.cols) {
        throw std::invalid_argument("syev: matrix must be square, got " +
                                    std::to_string(a.rows) + "x" + std::to_string(a.cols));
    }
    const std::size_t n = a.rows;
    SyevResult out;
    if (n == 0) return out;

    const bool want_vectors = (mode == SyevMode::ValuesAndVectors);

    // Work on a private copy: the reduction is destructive.
    Matrix work(n, n);
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = 0; i < n; ++i) work(i, j) = a(i, j);
    }

    std::vector<double> d(n, 0.0);
    std::vector<double> e(n, 0.0);
    std::vector<double> tau(n, 0.0);
    tridiagonalize(work.view(), d, e, tau);

    Matrix z;
    if (want_vectors) z = accumulate_q(work.view(), tau);

    ql_iterate(d, e, want_vectors ? z.view() : MatView{}, want_vectors);

    // Sort descending, permuting eigenvector columns to match.
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(),
              [&](std::size_t x, std::size_t y) { return d[x] > d[y]; });

    out.values.resize(n);
    for (std::size_t i = 0; i < n; ++i) out.values[i] = d[order[i]];

    if (want_vectors) {
        out.vectors = Matrix(n, n);
        for (std::size_t j = 0; j < n; ++j) {
            std::copy_n(z.col(order[j]), n, out.vectors.col(j));
        }
    }
    return out;
}

std::vector<double> syev_values(ConstMatView a) {
    return syev(a, SyevMode::ValuesOnly).values;
}

}  // namespace rnf
