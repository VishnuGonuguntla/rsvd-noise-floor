// Returns -> standardized returns -> correlation matrix.
//
// The normalization convention matters and is fixed here: columns are
// standardized with the POPULATION standard deviation and the correlation
// matrix is Z^T Z / T. The Marchenko-Pastur edges are stated for exactly this
// convention; mixing in a 1/(T-1) somewhere shifts the noise floor and
// quietly invalidates every comparison downstream.
#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "rnf/matrix.hpp"

namespace rnf {

struct ColumnStats {
    std::vector<double> mean;
    std::vector<double> stddev;  // population (1/T)
};

// Standardize each column of x (T x N) in place to zero mean, unit variance.
// Throws if a column has (near) zero variance -- a constant price series is a
// data problem, not something to paper over with an epsilon.
ColumnStats standardize_columns(MatView x);

// C = Z^T Z / T for already-standardized Z (T x N). Symmetric, unit diagonal,
// trace exactly N.
[[nodiscard]] Matrix correlation_matrix(ConstMatView z);

// Convenience: copy, standardize, correlate. Leaves the caller's data alone.
[[nodiscard]] Matrix correlation_from_returns(ConstMatView x);

[[nodiscard]] double trace(ConstMatView a);

// Aspect ratio q = N/T, the single parameter of the MP law.
[[nodiscard]] constexpr double aspect_ratio(std::size_t t, std::size_t n) {
    return static_cast<double>(n) / static_cast<double>(t);
}

// Inverse participation ratio, sum_i v_i^4, for a unit-norm eigenvector.
// Order 1/N for a delocalized mode (the market, loading on everything) and
// order 1/m for a mode concentrated on m names (a sector). This is what
// separates the two by inspection.
[[nodiscard]] double inverse_participation_ratio(std::span<const double> v);

// Number of effective contributors, 1/IPR: reads directly as "this mode is
// spread over about this many names".
[[nodiscard]] double participation_count(std::span<const double> v);

}  // namespace rnf
