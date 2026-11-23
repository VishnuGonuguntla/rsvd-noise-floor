// Wall-clock timing for the benchmarks.
//
// Reports the median and median absolute deviation rather than the mean and
// standard deviation: on a laptop with turbo and competing threads the timing
// distribution has a long right tail, and the median is what stays comparable
// between runs.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <vector>

namespace rnf {

class Stopwatch {
  public:
    Stopwatch() : t0_(std::chrono::steady_clock::now()) {}
    void reset() { t0_ = std::chrono::steady_clock::now(); }
    [[nodiscard]] double elapsed_ms() const {
        const auto dt = std::chrono::steady_clock::now() - t0_;
        return std::chrono::duration<double, std::milli>(dt).count();
    }

  private:
    std::chrono::steady_clock::time_point t0_;
};

struct TimingStats {
    double      median_ms = 0.0;
    double      mad_ms    = 0.0;  // median absolute deviation
    double      min_ms    = 0.0;
    std::size_t reps      = 0;
};

[[nodiscard]] inline TimingStats summarize_timings(std::vector<double> samples) {
    TimingStats s;
    if (samples.empty()) return s;
    std::sort(samples.begin(), samples.end());
    s.reps      = samples.size();
    s.min_ms    = samples.front();
    s.median_ms = samples[samples.size() / 2];
    std::vector<double> dev(samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i) {
        dev[i] = std::fabs(samples[i] - s.median_ms);
    }
    std::sort(dev.begin(), dev.end());
    s.mad_ms = dev[dev.size() / 2];
    return s;
}

// Run f() `warmup` times to settle caches and clocks, then `reps` timed times.
template <typename F>
[[nodiscard]] TimingStats time_repeats(std::size_t reps, std::size_t warmup, F&& f) {
    for (std::size_t i = 0; i < warmup; ++i) f();
    std::vector<double> samples;
    samples.reserve(reps);
    for (std::size_t i = 0; i < reps; ++i) {
        Stopwatch sw;
        f();
        samples.push_back(sw.elapsed_ms());
    }
    return summarize_timings(std::move(samples));
}

}  // namespace rnf
