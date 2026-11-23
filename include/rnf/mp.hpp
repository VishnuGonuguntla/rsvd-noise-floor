// The Marchenko-Pastur law: the noise floor of a sample correlation matrix.
//
// If T observations of N assets are pure i.i.d. noise, the eigenvalues of the
// sample correlation matrix do not cluster at 1 -- they spread over
// [sigma2*(1-sqrt(q))^2, sigma2*(1+sqrt(q))^2] with q = N/T, following the MP
// density. That upper edge is the noise floor: an eigenvalue inside the bulk
// is statistically indistinguishable from estimation error, and only
// eigenvalues above it are credible signal.
//
// This header is what turns "the top few eigenvalues look big" into a
// calibrated, testable statement.
#pragma once

#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace rnf {

struct MpParams {
    double q      = 0.0;  // aspect ratio N/T
    double sigma2 = 1.0;  // noise variance scale
};

struct MpSupport {
    double lo = 0.0;  // lambda_minus
    double hi = 0.0;  // lambda_plus -- THE noise floor
};

[[nodiscard]] MpSupport mp_support(MpParams p);

// Fraction of eigenvalues in the continuous bulk: 1 for q <= 1, else 1/q.
// When q > 1 the matrix is rank deficient and the remaining mass 1 - 1/q sits
// in an atom at exactly zero.
[[nodiscard]] double mp_bulk_mass(MpParams p);
[[nodiscard]] double mp_zero_atom(MpParams p);

// Density at lambda; zero outside the support.
[[nodiscard]] double mp_density(double lambda, MpParams p);

// CDF, including the zero atom when q > 1.
//
// Evaluated by substituting lambda = lo + 4*sqrt(q)*sin^2(theta), which makes
// the inverse-square-root edge singularities cancel exactly and leaves
// 4*sin^2(2 theta) / (pi * lambda(theta)) -- smooth, so fixed-order
// Gauss-Legendre reaches machine precision.
[[nodiscard]] double mp_cdf(double lambda, MpParams p);

// Mean of the MP distribution over its continuous part, per bulk eigenvalue.
// Equals sigma2 for q <= 1; the trace-conservation check leans on this.
[[nodiscard]] double mp_bulk_mean(MpParams p);

// Kolmogorov-Smirnov distance between an empirical spectrum and the MP CDF.
// Input order does not matter (a sorted copy is taken). Note that any genuine
// signal eigenvalues will dominate this statistic -- it is a test of the NULL,
// so feed it a spectrum you believe to be pure noise, or the bulk alone.
[[nodiscard]] double ks_statistic(std::span<const double> eigenvalues, MpParams p);

// Asymptotic Kolmogorov tail probability with the Stephens finite-sample
// correction. Small p means "this spectrum is not MP".
[[nodiscard]] double ks_pvalue(double d, std::size_t n);

// Baik-Ben Arous-Peche threshold. A planted factor of strength below this is
// swallowed by the bulk and is unrecoverable by ANY method -- which is what
// makes the lambda_plus cut the right rank rule rather than merely a
// convenient one.
[[nodiscard]] constexpr double bbp_threshold(double q) {
    return 1.0 + (q > 0.0 ? std::sqrt(q) : 0.0);
}

// Where the eigenvalue of a spike of strength s lands once it is detectable.
// Above the BBP threshold the observed eigenvalue is pushed out to
// s + q*s/(s-1); below it, the spike sticks to the edge.
[[nodiscard]] double spike_to_eigenvalue(double s, double q);

struct NoiseFloor {
    double      sigma2       = 1.0;  // fitted effective noise variance
    double      lambda_plus  = 0.0;  // the cut
    double      lambda_minus = 0.0;
    std::size_t k_signal     = 0;    // eigenvalues strictly above the cut
    int         iterations   = 0;
    bool        converged    = false;
    // True when every supplied eigenvalue was above the cut, so k_signal is a
    // lower bound only -- the caller did not hand over enough of the spectrum.
    bool        saturated    = false;
};

// Self-consistent noise floor.
//
// Fitting the bulk with sigma2 = 1 overestimates the floor, because the market
// mode carries a large share of the total variance and leaves the bulk
// narrower than the naive law predicts. So iterate: given k signal modes, the
// variance left for the bulk is (trace - sum of signal eigenvalues) spread
// over the remaining N - k modes; recompute the cut from that and recount.
// Two or three passes converge. This is the Laloux-Cizeau-Bouchaud-Potters
// procedure.
//
// `eigs_desc` may be the full spectrum or just the leading eigenvalues.
// `total_trace` is N for a correlation matrix, which is why the fit works even
// when the bulk itself was never computed.
[[nodiscard]] NoiseFloor fit_noise_floor(std::span<const double> eigs_desc,
                                         std::size_t n_total, double q,
                                         double total_trace, int max_iter = 32);

// Same, taking total_trace = n_total (the correlation-matrix case).
[[nodiscard]] NoiseFloor fit_noise_floor(std::span<const double> eigs_desc,
                                         std::size_t n_total, double q);

struct TraceCheck {
    double discarded_mass   = 0.0;  // trace - sum of the k retained eigenvalues
    double sigma2_eff       = 0.0;  // discarded_mass / (n_total - k)
    double lambda_plus_eff  = 0.0;  // the floor implied by that variance
    double next_eigenvalue  = 0.0;  // lambda_{k+1}
    double headroom         = 0.0;  // lambda_plus_eff - lambda_{k+1}
    bool   consistent       = false;
};

// Trace conservation: the cheap check that RSVD threw away only noise.
//
// RSVD returns the top k and never computes the bulk -- but tr(C) = N exactly
// for a correlation matrix, so the discarded mass is known without an O(N^3)
// decomposition. If the (k+1)-th eigenvalue sits below the floor implied by
// that discarded mass, the remainder is consistent with MP noise. Requires at
// least k+1 eigenvalues.
[[nodiscard]] TraceCheck trace_conservation_check(std::span<const double> eigs_desc,
                                                  std::size_t k, std::size_t n_total,
                                                  double q, double total_trace);

// Histogram bin counts plus the MP density sampled at bin centres, ready for
// the overlay plot.
struct SpectrumHistogram {
    std::vector<double> bin_centre;
    std::vector<double> empirical_density;  // normalized to integrate to 1
    std::vector<double> mp_density;
    std::size_t         count_outside_support = 0;
};

[[nodiscard]] SpectrumHistogram histogram_vs_mp(std::span<const double> eigenvalues,
                                                MpParams p, std::size_t bins,
                                                double lo, double hi);

}  // namespace rnf
