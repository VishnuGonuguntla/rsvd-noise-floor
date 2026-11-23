// Householder QR factorization and thin-Q formation.
//
// RSVD needs exactly one thing from QR: an orthonormal basis for the column
// space of a tall-skinny sketch, recomputed after every application of the
// matrix. `orthonormalize` is that operation; the lower-level pieces are
// exposed because the tests check QR = A and Q^T Q = I directly.
#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "rnf/matrix.hpp"

namespace rnf {

// In-place Householder QR of A (m x n, m >= n), LAPACK dgeqr2 conventions:
// on return the upper triangle of A holds R, and the Householder vector for
// step j occupies A(j+1:m, j) with an implicit leading 1. tau must be size n.
void householder_qr(MatView a, std::span<double> tau);

// Overwrite the factored A with the thin Q (m x n, orthonormal columns),
// LAPACK dorg2r. Must be called on the output of householder_qr.
void form_thin_q(MatView a, std::span<const double> tau);

// Copy the n x n upper-triangular R out of a factored A.
[[nodiscard]] Matrix extract_r(ConstMatView a);

// A <- orthonormal basis of the column space of A (m x n, m >= n).
// This is the workhorse called inside every power iteration.
void orthonormalize(Matrix& a);

// ||Q^T Q - I||_F, the orthogonality diagnostic reported by the apps.
[[nodiscard]] double orthogonality_error(ConstMatView q);

}  // namespace rnf
