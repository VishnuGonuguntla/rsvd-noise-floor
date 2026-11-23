// Gauss-Legendre quadrature.
//
// Needed because the Marchenko-Pastur CDF has no elementary antiderivative we
// want to rely on, and the density carries inverse-square-root singularities
// at both support edges. After the substitution used in mp.cpp those
// singularities cancel exactly, leaving a smooth integrand that fixed-order
// Gauss-Legendre nails to machine precision.
#pragma once

#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

namespace rnf {

struct GaussLegendre {
    std::vector<double> nodes;    // on [-1, 1]
    std::vector<double> weights;
};

// Nodes are the roots of the degree-n Legendre polynomial, found by Newton
// iteration from the standard Chebyshev-like initial guess. Symmetric by
// construction, so only half are computed and the rest mirrored.
[[nodiscard]] inline GaussLegendre gauss_legendre(std::size_t n) {
    GaussLegendre gl;
    gl.nodes.resize(n);
    gl.weights.resize(n);
    if (n == 0) return gl;

    const std::size_t half = (n + 1) / 2;
    const double      dn   = static_cast<double>(n);

    for (std::size_t i = 0; i < half; ++i) {
        double x = std::cos(std::numbers::pi * (static_cast<double>(i) + 0.75) / (dn + 0.5));
        double dp = 0.0;

        for (int it = 0; it < 100; ++it) {
            // Legendre recurrence: (k+1) P_{k+1} = (2k+1) x P_k - k P_{k-1}
            double p_prev = 1.0;  // P_0
            double p_cur  = x;    // P_1
            // Advance to P_n: k runs to n-1, so the body executes n-1 times.
            // (Stopping at n-2 yields the roots of P_{n-1} and a rule whose
            // nodes and weights disagree -- a 0.3% error in smooth integrals.)
            for (std::size_t k = 1; k <= n - 1; ++k) {
                const double dk = static_cast<double>(k);
                const double p_next = ((2.0 * dk + 1.0) * x * p_cur - dk * p_prev) / (dk + 1.0);
                p_prev = p_cur;
                p_cur  = p_next;
            }
            // p_cur = P_n(x), p_prev = P_{n-1}(x)
            dp = dn * (x * p_cur - p_prev) / (x * x - 1.0);
            const double dx = -p_cur / dp;
            x += dx;
            if (std::fabs(dx) < 1e-15) break;
        }

        gl.nodes[i]           = -x;
        gl.nodes[n - 1 - i]   = x;
        const double w        = 2.0 / ((1.0 - x * x) * dp * dp);
        gl.weights[i]         = w;
        gl.weights[n - 1 - i] = w;
    }
    return gl;
}

// Integrate f over [a, b] with a precomputed rule.
template <typename F>
[[nodiscard]] double integrate(const GaussLegendre& gl, double a, double b, F&& f) {
    const double mid  = 0.5 * (a + b);
    const double half = 0.5 * (b - a);
    double acc = 0.0;
    for (std::size_t i = 0; i < gl.nodes.size(); ++i) {
        acc += gl.weights[i] * f(mid + half * gl.nodes[i]);
    }
    return acc * half;
}

}  // namespace rnf
