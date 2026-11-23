// Accuracy diagnostics for the randomized solver.
//
// A randomized algorithm that returns plausible-looking numbers is worthless
// without a way to tell whether they are right. Three independent checks live
// here: residuals and orthogonality (self-contained, always affordable),
// comparison against a dense reference (exact, but O(N^3), so only at
// moderate N), and the a posteriori range-finder error estimate (cheap, and
// available at sizes where no reference can be computed).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rnf/matrix.hpp"
#include "rnf/rsvd.hpp"

namespace rnf {

struct RsvdDiagnostics {
    std::vector<double> residuals;      // ||C u_i - lambda_i u_i|| / |lambda_i|
    double              max_residual = 0.0;
    double              orthogonality = 0.0;  // ||U^T U - I||_F
};

[[nodiscard]] RsvdDiagnostics diagnose(ConstMatView c, const RsvdResult& r);

struct SubspaceComparison {
    std::vector<double> relative_errors;               // per eigenvalue
    double              max_relative_eigenvalue_error = 0.0;
    double              max_absolute_eigenvalue_error = 0.0;
    // Largest principal angle between the reference and computed invariant
    // subspaces. Eigenvalues can be right while the subspace is wrong, so
    // this is the check that actually matters for a factor model.
    double              sin_theta_max = 0.0;
};

[[nodiscard]] SubspaceComparison compare_to_reference(const std::vector<double>& ref_values_desc,
                                                      ConstMatView ref_vectors,
                                                      const RsvdResult& got);

// A posteriori estimate of ||(I - Q Q^T) C||_2 from r Gaussian probes
// (Halko-Martinsson-Tropp, section 4.3). Returns the raw
// max_i ||(I - Q Q^T) C w_i||; the rigorous bound multiplies this by
// 10*sqrt(2/pi) and holds with probability 1 - 10^(-r), so r = 10 probes is
// already a strong statement. Usable at sizes where a dense reference is out
// of reach, which is the whole point.
[[nodiscard]] double estimate_projection_error(ConstMatView c, ConstMatView q,
                                               std::size_t probes, std::uint64_t seed);

[[nodiscard]] constexpr double projection_error_bound_factor() {
    return 10.0 * 0.7978845608028654;  // 10 * sqrt(2/pi)
}

}  // namespace rnf
