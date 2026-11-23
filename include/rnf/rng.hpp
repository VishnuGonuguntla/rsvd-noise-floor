// Reproducible Gaussian sampling.
//
// std::normal_distribution is deliberately avoided: it is not specified to
// produce the same stream across standard-library implementations, and it
// carries hidden state (Box-Muller caches its second variate), so the numbers
// you get depend on the call pattern as well as the seed. Everything in this
// project must be reproducible from (seed, length) alone, so the transform is
// written out explicitly and buffers are filled in fixed-size pairs.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <random>
#include <span>

#include "rnf/matrix.hpp"

namespace rnf {

class Rng {
  public:
    explicit Rng(std::uint64_t seed) {
        // Spread a single 64-bit seed over the full Mersenne state rather than
        // leaving most of it zero.
        std::seed_seq seq{static_cast<std::uint32_t>(seed & 0xFFFFFFFFu),
                          static_cast<std::uint32_t>(seed >> 32),
                          0x9E3779B9u, 0x85EBCA6Bu};
        gen_.seed(seq);
    }

    std::uint64_t next_u64() { return gen_(); }

    // Uniform on the OPEN interval (0, 1): the log in Box-Muller must never
    // see zero. The +0.5 offset keeps both endpoints excluded.
    double uniform_open() {
        return (static_cast<double>(gen_() >> 11) + 0.5) * 0x1.0p-53;
    }

    // Uniform on [0, 1).
    double uniform() { return static_cast<double>(gen_() >> 11) * 0x1.0p-53; }

    std::mt19937_64& engine() { return gen_; }

  private:
    std::mt19937_64 gen_;
};

// Fill `out` with i.i.d. standard normals via the Box-Muller transform.
// Deterministic in (seed state, out.size()).
inline void fill_normal(std::span<double> out, Rng& rng) {
    constexpr double two_pi = 2.0 * std::numbers::pi;
    const std::size_t n = out.size();
    for (std::size_t i = 0; i < n; i += 2) {
        const double u1 = rng.uniform_open();
        const double u2 = rng.uniform_open();
        const double r  = std::sqrt(-2.0 * std::log(u1));
        const double th = two_pi * u2;
        out[i] = r * std::cos(th);
        if (i + 1 < n) out[i + 1] = r * std::sin(th);
    }
}

// Column-by-column so the result depends only on (seed, rows, cols) and not on
// the padded leading dimension of the destination.
inline void fill_normal(MatView m, Rng& rng) {
    for (std::size_t j = 0; j < m.cols; ++j) {
        fill_normal(std::span<double>{m.col(j), m.rows}, rng);
    }
}

[[nodiscard]] inline Matrix gaussian_matrix(std::size_t rows, std::size_t cols, Rng& rng) {
    Matrix m(rows, cols);
    fill_normal(m.view(), rng);
    return m;
}

}  // namespace rnf
