// CSV in and out.
//
// Output goes to CSV (rather than being plotted in-process) so that every
// figure in docs/ is reproducible from a checked-in gnuplot script acting on
// a checked-in data file.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "rnf/matrix.hpp"

namespace rnf {

// Write equal-length numeric columns under the given headers.
void write_table(const std::string& path, const std::vector<std::string>& headers,
                 const std::vector<std::vector<double>>& columns, int precision = 12);

// Write pre-formatted rows, for tables mixing text and numbers.
void write_rows(const std::string& path, const std::vector<std::string>& headers,
                const std::vector<std::vector<std::string>>& rows);

struct ReturnsData {
    Matrix                   values;  // T x N, observations in rows
    std::vector<std::string> names;   // N asset names, from the header row
    std::vector<std::string> dates;   // T labels; empty when there is no date column
};

// Read a returns CSV: an optional leading date/label column, then one column
// per asset, with a header row naming them.
//
// Malformed input is rejected with the offending line and column named --
// silently dropping rows would corrupt T, and T is the denominator of the
// aspect ratio q that the entire noise floor is calibrated on.
[[nodiscard]] ReturnsData read_returns_csv(const std::string& path,
                                           bool has_date_column = true);

}  // namespace rnf
