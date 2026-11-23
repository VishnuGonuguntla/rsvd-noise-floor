#include "rnf/mp.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

#include "rnf/quadrature.hpp"

namespace rnf {
namespace {

// 128 nodes is far more than the smooth post-substitution integrand needs; the
// cost is negligible and it removes quadrature order as a thing to worry about.
const GaussLegendre& rule() {
    static const GaussLegendre gl = gauss_legendre(128);
    return gl;
}

void validate(MpParams p) {
    if (!(p.q > 0.0)) {
        throw std::invalid_argument("mp: aspect ratio q must be positive, got " +
                                    std::to_string(p.q));
    }
    if (!(p.sigma2 > 0.0)) {
        throw std::invalid_argument("mp: sigma2 must be positive, got " +
                                    std::to_string(p.sigma2));
    }
}

// Support of the unit-variance law.
struct UnitSupport {
    double lo;
    double hi;
};
UnitSupport unit_support(double q) {
    const double rq = std::sqrt(q);
    return {(1.0 - rq) * (1.0 - rq), (1.0 + rq) * (1.0 + rq)};
}

// Integral of the unit-variance density from lo up to `x`, via the
// edge-desingularizing substitution x = lo + 4*sqrt(q)*sin^2(theta).
double unit_cdf_integral(double x, double q) {
    const UnitSupport s = unit_support(q);
    if (x <= s.lo) return 0.0;

    const double span = s.hi - s.lo;  // == 4*sqrt(q)
    const double frac = std::clamp((x - s.lo) / span, 0.0, 1.0);
    const double theta1 = std::asin(std::sqrt(frac));

    return integrate(rule(), 0.0, theta1, [&](double th) {
        const double st  = std::sin(th);
        const double lam = s.lo + span * st * st;
        const double s2t = std::sin(2.0 * th);
        return 4.0 * s2t * s2t / (std::numbers::pi * lam);
    });
}

}  // namespace

MpSupport mp_support(MpParams p) {
    validate(p);
    const UnitSupport s = unit_support(p.q);
    return {p.sigma2 * s.lo, p.sigma2 * s.hi};
}

double mp_bulk_mass(MpParams p) {
    validate(p);
    return std::min(1.0, 1.0 / p.q);
}

double mp_zero_atom(MpParams p) {
    validate(p);
    return std::max(0.0, 1.0 - 1.0 / p.q);
}

double mp_density(double lambda, MpParams p) {
    validate(p);
    const double x = lambda / p.sigma2;
    const UnitSupport s = unit_support(p.q);
    if (x <= s.lo || x >= s.hi) return 0.0;
    const double num = std::sqrt((s.hi - x) * (x - s.lo));
    // 1/sigma2 converts the unit-variance density to a density in lambda.
    return num / (2.0 * std::numbers::pi * p.q * x) / p.sigma2;
}

double mp_cdf(double lambda, MpParams p) {
    validate(p);
    const double atom = mp_zero_atom(p);
    if (lambda < 0.0) return 0.0;

    const double x = lambda / p.sigma2;
    const UnitSupport s = unit_support(p.q);
    if (x <= s.lo) return atom;
    if (x >= s.hi) return 1.0;
    return std::clamp(atom + unit_cdf_integral(x, p.q), 0.0, 1.0);
}

double mp_bulk_mean(MpParams p) {
    validate(p);
    // E[lambda] over the whole law is sigma2; the continuous part carries
    // bulk_mass of the probability, so per bulk eigenvalue the mean is
    // sigma2 / bulk_mass (for q <= 1 this is just sigma2).
    return p.sigma2 / mp_bulk_mass(p);
}

double ks_statistic(std::span<const double> eigenvalues, MpParams p) {
    validate(p);
    if (eigenvalues.empty()) return 0.0;

    std::vector<double> x(eigenvalues.begin(), eigenvalues.end());
    std::sort(x.begin(), x.end());

    const double n = static_cast<double>(x.size());
    double d = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double f     = mp_cdf(x[i], p);
        const double upper = static_cast<double>(i + 1) / n - f;  // F_emp just above x_i
        const double lower = f - static_cast<double>(i) / n;      // F_emp just below x_i
        d = std::max(d, std::max(upper, lower));
    }
    return d;
}

double ks_pvalue(double d, std::size_t n) {
    if (n == 0) return 1.0;
    if (d <= 0.0) return 1.0;

    const double rn = std::sqrt(static_cast<double>(n));
    // Stephens' correction: makes the asymptotic form usable at moderate n.
    const double z  = (rn + 0.12 + 0.11 / rn) * d;

    // Q(z) = 2 * sum_{j>=1} (-1)^(j-1) exp(-2 j^2 z^2)
    const double z2 = z * z;
    if (z2 > 100.0) return 0.0;  // underflow territory; the tail is zero
    double sum  = 0.0;
    double term = 0.0;
    for (int j = 1; j <= 200; ++j) {
        const double dj = static_cast<double>(j);
        term = std::exp(-2.0 * dj * dj * z2);
        sum += ((j % 2 == 1) ? 1.0 : -1.0) * term;
        if (term < 1e-18) break;
    }
    return std::clamp(2.0 * sum, 0.0, 1.0);
}

double spike_to_eigenvalue(double s, double q) {
    // Below the BBP threshold the spike does not separate: the leading bulk
    // eigenvalue stays pinned at the upper edge.
    if (s <= bbp_threshold(q)) return (1.0 + std::sqrt(q)) * (1.0 + std::sqrt(q));
    return s + q * s / (s - 1.0);
}

NoiseFloor fit_noise_floor(std::span<const double> eigs_desc, std::size_t n_total, double q,
                           double total_trace, int max_iter) {
    if (!(q > 0.0)) throw std::invalid_argument("fit_noise_floor: q must be positive");
    if (n_total == 0) throw std::invalid_argument("fit_noise_floor: n_total must be positive");
    if (!std::is_sorted(eigs_desc.begin(), eigs_desc.end(), std::greater<>{})) {
        throw std::invalid_argument("fit_noise_floor: eigenvalues must be in descending order");
    }

    const double edge = (1.0 + std::sqrt(q)) * (1.0 + std::sqrt(q));

    NoiseFloor nf;
    nf.sigma2 = 1.0;
    std::size_t k_prev = std::numeric_limits<std::size_t>::max();

    for (int it = 1; it <= max_iter; ++it) {
        nf.iterations = it;
        const double cut = nf.sigma2 * edge;

        std::size_t k = 0;
        double signal_sum = 0.0;
        while (k < eigs_desc.size() && eigs_desc[k] > cut) {
            signal_sum += eigs_desc[k];
            ++k;
        }

        if (k >= n_total) {  // degenerate: nothing left for a bulk
            nf.k_signal    = n_total;
            nf.saturated   = true;
            nf.lambda_plus = cut;
            nf.lambda_minus = nf.sigma2 * (1.0 - std::sqrt(q)) * (1.0 - std::sqrt(q));
            return nf;
        }

        const double residual = total_trace - signal_sum;
        double sigma2_new = residual / static_cast<double>(n_total - k);
        if (!(sigma2_new > 0.0)) sigma2_new = 1e-300;  // pathological input; keep it finite

        const bool stable = (k == k_prev) &&
                            std::fabs(sigma2_new - nf.sigma2) <= 1e-13 * std::max(1.0, nf.sigma2);

        k_prev     = k;
        nf.sigma2  = sigma2_new;
        nf.k_signal = k;

        if (stable) { nf.converged = true; break; }
    }

    nf.lambda_plus  = nf.sigma2 * edge;
    nf.lambda_minus = nf.sigma2 * (1.0 - std::sqrt(q)) * (1.0 - std::sqrt(q));
    nf.saturated    = (nf.k_signal == eigs_desc.size()) && !eigs_desc.empty();
    return nf;
}

NoiseFloor fit_noise_floor(std::span<const double> eigs_desc, std::size_t n_total, double q) {
    return fit_noise_floor(eigs_desc, n_total, q, static_cast<double>(n_total));
}

TraceCheck trace_conservation_check(std::span<const double> eigs_desc, std::size_t k,
                                    std::size_t n_total, double q, double total_trace) {
    if (eigs_desc.size() < k + 1) {
        throw std::invalid_argument(
            "trace_conservation_check: need at least k+1 eigenvalues, have " +
            std::to_string(eigs_desc.size()) + " for k=" + std::to_string(k));
    }
    if (k >= n_total) throw std::invalid_argument("trace_conservation_check: k must be < n_total");

    TraceCheck tc;
    double retained = 0.0;
    for (std::size_t i = 0; i < k; ++i) retained += eigs_desc[i];

    tc.discarded_mass  = total_trace - retained;
    tc.sigma2_eff      = tc.discarded_mass / static_cast<double>(n_total - k);
    tc.lambda_plus_eff = tc.sigma2_eff * (1.0 + std::sqrt(q)) * (1.0 + std::sqrt(q));
    tc.next_eigenvalue = eigs_desc[k];
    tc.headroom        = tc.lambda_plus_eff - tc.next_eigenvalue;
    tc.consistent      = tc.headroom >= 0.0;
    return tc;
}

SpectrumHistogram histogram_vs_mp(std::span<const double> eigenvalues, MpParams p,
                                  std::size_t bins, double lo, double hi) {
    validate(p);
    if (bins == 0) throw std::invalid_argument("histogram_vs_mp: bins must be positive");
    if (!(hi > lo)) throw std::invalid_argument("histogram_vs_mp: need hi > lo");

    SpectrumHistogram h;
    h.bin_centre.resize(bins);
    h.empirical_density.assign(bins, 0.0);
    h.mp_density.resize(bins);

    const double width = (hi - lo) / static_cast<double>(bins);
    for (std::size_t b = 0; b < bins; ++b) {
        h.bin_centre[b] = lo + (static_cast<double>(b) + 0.5) * width;
        h.mp_density[b] = mp_density(h.bin_centre[b], p);
    }

    std::vector<double> counts(bins, 0.0);
    for (double v : eigenvalues) {
        if (v < lo || v >= hi) { ++h.count_outside_support; continue; }
        const auto b = static_cast<std::size_t>((v - lo) / width);
        counts[std::min(b, bins - 1)] += 1.0;
    }

    // Normalize to a density over ALL supplied eigenvalues, so the area under
    // the empirical curve is the fraction that landed inside [lo, hi) and can
    // be compared directly against the MP curve.
    const double n = static_cast<double>(eigenvalues.size());
    if (n > 0.0) {
        for (std::size_t b = 0; b < bins; ++b) h.empirical_density[b] = counts[b] / (n * width);
    }
    return h;
}

}  // namespace rnf
