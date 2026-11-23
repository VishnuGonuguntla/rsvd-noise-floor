// Dense symmetric eigensolver: Householder tridiagonalization followed by the
// implicit-shift QL iteration.
//
// This is the project's ground truth. RSVD's top-k is checked against it, and
// the Marchenko-Pastur null test needs full spectra from it. Values-only mode
// matters: it skips the O(N^3) eigenvector accumulation, which is pure waste
// when all we want is a histogram of eigenvalues.
#pragma once

#include <cstddef>
#include <vector>

#include "rnf/matrix.hpp"

namespace rnf {

enum class SyevMode { ValuesOnly, ValuesAndVectors };

struct SyevResult {
    std::vector<double> values;   // descending
    Matrix              vectors;  // columns are eigenvectors; empty in ValuesOnly mode
};

// `a` must be symmetric with both triangles populated; it is copied, not
// modified. Eigenvalues come back in descending order, with eigenvector
// columns permuted to match.
[[nodiscard]] SyevResult syev(ConstMatView a, SyevMode mode = SyevMode::ValuesAndVectors);

// Convenience for the spectral-density work, which never needs vectors.
[[nodiscard]] std::vector<double> syev_values(ConstMatView a);

}  // namespace rnf
