#include "rnf/spiked.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>

#include "rnf/mp.hpp"
#include "rnf/qr.hpp"
#include "rnf/rng.hpp"

namespace rnf {

SpikedModel generate_spiked_returns(std::size_t t, std::size_t n,
                                    std::vector<double> strengths, std::uint64_t seed) {
    if (t < 2) throw std::invalid_argument("generate_spiked_returns: need T >= 2");
    if (n == 0) throw std::invalid_argument("generate_spiked_returns: need N >= 1");
    if (strengths.size() > n) {
        throw std::invalid_argument("generate_spiked_returns: more spikes (" +
                                    std::to_string(strengths.size()) + ") than assets (" +
                                    std::to_string(n) + ")");
    }
    for (double s : strengths) {
        if (!(s > 0.0)) {
            throw std::invalid_argument("generate_spiked_returns: spike strengths must be "
                                        "positive, got " + std::to_string(s));
        }
    }
    std::sort(strengths.begin(), strengths.end(), std::greater<>{});

    SpikedModel m;
    m.t = t;
    m.n = n;
    m.q = static_cast<double>(n) / static_cast<double>(t);
    m.strengths = strengths;

    const std::size_t r = strengths.size();
    Rng rng(seed);

    // Random orthonormal planted directions. Random => delocalized, which is
    // what keeps the covariance and correlation models interchangeable here.
    m.directions = gaussian_matrix(n, r, rng);
    if (r > 0) orthonormalize(m.directions);

    // x = Sigma^{1/2} g with Sigma^{1/2} = I + sum_j (sqrt(s_j) - 1) v_j v_j^T,
    // applied as a rank-r update so the cost is O(T*N*r), not O(T*N^2).
    m.returns = gaussian_matrix(t, n, rng);

    std::vector<double> scale(r);
    for (std::size_t j = 0; j < r; ++j) scale[j] = std::sqrt(strengths[j]) - 1.0;

    std::vector<double> proj(r, 0.0);
    for (std::size_t i = 0; i < t; ++i) {
        for (std::size_t j = 0; j < r; ++j) {
            const double* v = m.directions.col(j);
            double dot = 0.0;
            for (std::size_t c = 0; c < n; ++c) dot += v[c] * m.returns(i, c);
            proj[j] = scale[j] * dot;
        }
        for (std::size_t j = 0; j < r; ++j) {
            if (proj[j] == 0.0) continue;
            const double* v = m.directions.col(j);
            for (std::size_t c = 0; c < n; ++c) m.returns(i, c) += proj[j] * v[c];
        }
    }
    return m;
}

double spike_eigenvector_overlap_sq(double s, double q) {
    if (s <= bbp_threshold(q)) return 0.0;
    const double d = s - 1.0;
    return (1.0 - q / (d * d)) / (1.0 + q / d);
}

}  // namespace rnf
