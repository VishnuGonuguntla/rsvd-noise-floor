// The spiked covariance model, and the BBP phase transition.
//
// This is where the Marchenko-Pastur cut stops being a convenient heuristic
// and becomes the provably right rank rule.
//
// Plant r factors into otherwise-i.i.d. data: the population covariance is
// Sigma = I + sum_j (s_j - 1) v_j v_j^T, so its eigenvalues are s_1..s_r and
// then 1 repeated. Baik-Ben Arous-Peche says the sample spectrum undergoes a
// phase transition at s = 1 + sqrt(q):
//
//   above it  -- the sample eigenvalue separates from the bulk, landing at
//                s + q*s/(s-1), and the sample eigenvector retains a
//                non-vanishing overlap with the true direction;
//   below it  -- the eigenvalue is swallowed by the bulk, sitting at the
//                upper edge like any noise eigenvalue, and the eigenvector
//                decorrelates from the truth entirely.
//
// The consequence: below the threshold NO algorithm recovers the factor, from
// the information in the sample. So cutting at lambda_plus discards nothing
// that could have been kept, and randomized SVD gives up nothing by stopping
// there.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rnf/matrix.hpp"

namespace rnf {

struct SpikedModel {
    Matrix              returns;            // T x N
    Matrix              directions;         // N x r, orthonormal planted directions
    std::vector<double> strengths;          // s_j, descending
    std::size_t         t = 0;
    std::size_t         n = 0;
    double              q = 0.0;
};

// Draw T observations with population covariance I + sum_j (s_j - 1) v_j v_j^T.
//
// The planted directions are random and therefore delocalized, which keeps
// diag(Sigma) within O(1/N) of 1 -- so the population CORRELATION matrix is
// the same thing to that order and the BBP predictions, which are stated for
// the covariance model, apply directly.
[[nodiscard]] SpikedModel generate_spiked_returns(std::size_t t, std::size_t n,
                                                  std::vector<double> strengths,
                                                  std::uint64_t seed);

// Limiting squared overlap |<u_hat, v>|^2 between the sample eigenvector and
// the planted direction (Paul 2007):
//
//     (1 - q/(s-1)^2) / (1 + q/(s-1))    for s > 1 + sqrt(q), else 0.
//
// It falls continuously to zero AT the BBP threshold -- the eigenvector stops
// carrying information about the factor before the eigenvalue stops being
// visible, which is why an eigenvalue barely above the floor is not worth
// trusting.
[[nodiscard]] double spike_eigenvector_overlap_sq(double s, double q);

}  // namespace rnf
