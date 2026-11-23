#include "rnf/stats.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "rnf/gemm.hpp"

namespace rnf {

ColumnStats standardize_columns(MatView x) {
    const std::size_t t = x.rows;
    const std::size_t n = x.cols;
    if (t == 0 || n == 0) throw std::invalid_argument("standardize_columns: empty matrix");
    if (t < 2) throw std::invalid_argument("standardize_columns: need at least 2 observations");

    ColumnStats stats;
    stats.mean.resize(n);
    stats.stddev.resize(n);

    const double inv_t = 1.0 / static_cast<double>(t);

    for (std::size_t j = 0; j < n; ++j) {
        double* col = x.col(j);

        double mean = 0.0;
        for (std::size_t i = 0; i < t; ++i) mean += col[i];
        mean *= inv_t;

        // Second pass on the centered values: more accurate than the
        // sum-of-squares-minus-square-of-sum shortcut, which cancels badly
        // when the mean dominates the variation (as it does for price levels).
        double var = 0.0;
        for (std::size_t i = 0; i < t; ++i) {
            const double d = col[i] - mean;
            var += d * d;
        }
        var *= inv_t;
        const double sd = std::sqrt(var);

        // Scale-relative threshold: a column of 1e-30 values is fine, a column
        // that is genuinely constant is not.
        const double scale = std::max(std::fabs(mean), 1.0);
        if (!(sd > 1e-13 * scale)) {
            throw std::runtime_error("standardize_columns: column " + std::to_string(j) +
                                     " has zero variance (stddev=" + std::to_string(sd) +
                                     "); a constant series cannot be standardized");
        }

        const double inv_sd = 1.0 / sd;
        for (std::size_t i = 0; i < t; ++i) col[i] = (col[i] - mean) * inv_sd;

        stats.mean[j]   = mean;
        stats.stddev[j] = sd;
    }
    return stats;
}

Matrix correlation_matrix(ConstMatView z) {
    const std::size_t t = z.rows;
    const std::size_t n = z.cols;
    if (t == 0 || n == 0) throw std::invalid_argument("correlation_matrix: empty matrix");

    Matrix c(n, n);
    syrk_trans(1.0 / static_cast<double>(t), z, 0.0, c.view());

    // Pin the diagonal to exactly 1: it is 1 analytically, and rounding noise
    // there would show up as a spurious trace deviation in the
    // trace-conservation test, which is tuned far tighter than 1e-16 * N.
    for (std::size_t i = 0; i < n; ++i) c(i, i) = 1.0;
    return c;
}

Matrix correlation_from_returns(ConstMatView x) {
    Matrix z(x.rows, x.cols);
    for (std::size_t j = 0; j < x.cols; ++j) {
        for (std::size_t i = 0; i < x.rows; ++i) z(i, j) = x(i, j);
    }
    standardize_columns(z.view());
    return correlation_matrix(z);
}

double trace(ConstMatView a) {
    double acc = 0.0;
    const std::size_t n = std::min(a.rows, a.cols);
    for (std::size_t i = 0; i < n; ++i) acc += a(i, i);
    return acc;
}

double inverse_participation_ratio(std::span<const double> v) {
    double norm2 = 0.0;
    double quart = 0.0;
    for (double x : v) {
        const double x2 = x * x;
        norm2 += x2;
        quart += x2 * x2;
    }
    if (norm2 <= 0.0) return 0.0;
    return quart / (norm2 * norm2);  // normalization-independent
}

double participation_count(std::span<const double> v) {
    const double ipr = inverse_participation_ratio(v);
    return ipr > 0.0 ? 1.0 / ipr : 0.0;
}

}  // namespace rnf
