#include "rnf/csv.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace rnf {
namespace {

// Split on `delim`, preserving empty fields -- including a trailing one.
//
// Not std::getline: it stops at the last delimiter, so "a,b," yields two
// fields rather than three. That turns a missing value at the end of a row
// into a "wrong number of fields" error, which misdiagnoses the problem, and
// makes "a,,b" and "a,b," behave differently for no good reason.
std::vector<std::string> split(const std::string& line, char delim) {
    std::vector<std::string> out;
    std::size_t              start = 0;
    for (;;) {
        const std::size_t pos = line.find(delim, start);
        if (pos == std::string::npos) {
            out.push_back(line.substr(start));
            break;
        }
        out.push_back(line.substr(start, pos - start));
        start = pos + 1;
    }
    return out;
}

std::string trim(std::string s) {
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    if (!s.empty() && s.back() == '\r') s.pop_back();
    return s;
}

}  // namespace

void write_table(const std::string& path, const std::vector<std::string>& headers,
                 const std::vector<std::vector<double>>& columns, int precision) {
    if (headers.size() != columns.size()) {
        throw std::invalid_argument("write_table: " + std::to_string(headers.size()) +
                                    " headers for " + std::to_string(columns.size()) +
                                    " columns");
    }
    std::size_t rows = 0;
    for (const auto& c : columns) rows = std::max(rows, c.size());
    for (const auto& c : columns) {
        if (!c.empty() && c.size() != rows) {
            throw std::invalid_argument("write_table: columns have differing lengths");
        }
    }

    std::ofstream f(path);
    if (!f) throw std::runtime_error("write_table: cannot open '" + path + "' for writing");
    f << std::setprecision(precision);

    for (std::size_t j = 0; j < headers.size(); ++j) {
        if (j) f << ',';
        f << headers[j];
    }
    f << '\n';

    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < columns.size(); ++j) {
            if (j) f << ',';
            if (i < columns[j].size()) f << columns[j][i];
        }
        f << '\n';
    }
    if (!f) throw std::runtime_error("write_table: write failed for '" + path + "'");
}

void write_rows(const std::string& path, const std::vector<std::string>& headers,
                const std::vector<std::vector<std::string>>& rows) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("write_rows: cannot open '" + path + "' for writing");
    for (std::size_t j = 0; j < headers.size(); ++j) {
        if (j) f << ',';
        f << headers[j];
    }
    f << '\n';
    for (const auto& r : rows) {
        for (std::size_t j = 0; j < r.size(); ++j) {
            if (j) f << ',';
            f << r[j];
        }
        f << '\n';
    }
    if (!f) throw std::runtime_error("write_rows: write failed for '" + path + "'");
}

ReturnsData read_returns_csv(const std::string& path, bool has_date_column) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("read_returns_csv: cannot open '" + path + "'");

    std::string header_line;
    if (!std::getline(f, header_line)) {
        throw std::runtime_error("read_returns_csv: '" + path + "' is empty");
    }

    std::vector<std::string> header = split(header_line, ',');
    for (auto& h : header) h = trim(h);
    if (header.size() < (has_date_column ? 2u : 1u)) {
        throw std::runtime_error("read_returns_csv: header has only " +
                                 std::to_string(header.size()) + " field(s)");
    }

    const std::size_t first_data_col = has_date_column ? 1 : 0;
    ReturnsData       data;
    data.names.assign(header.begin() + static_cast<std::ptrdiff_t>(first_data_col), header.end());
    const std::size_t n = data.names.size();

    // Rows arrive one at a time and T is unknown up front, so accumulate
    // row-major and transpose into the column-major Matrix at the end.
    std::vector<std::vector<double>> rows;
    std::string                      line;
    std::size_t                      lineno = 1;

    while (std::getline(f, line)) {
        ++lineno;
        if (trim(line).empty()) continue;

        const std::vector<std::string> fields = split(line, ',');
        if (fields.size() != header.size()) {
            throw std::runtime_error("read_returns_csv: " + path + ":" + std::to_string(lineno) +
                                     " has " + std::to_string(fields.size()) +
                                     " fields, expected " + std::to_string(header.size()));
        }

        if (has_date_column) data.dates.push_back(trim(fields[0]));

        std::vector<double> row(n);
        for (std::size_t j = 0; j < n; ++j) {
            const std::string tok = trim(fields[first_data_col + j]);
            if (tok.empty()) {
                throw std::runtime_error("read_returns_csv: " + path + ":" +
                                         std::to_string(lineno) + " column '" + data.names[j] +
                                         "' is empty; gaps must be filled before use");
            }
            double        v = 0.0;
            std::size_t   consumed = 0;
            try {
                v = std::stod(tok, &consumed);
            } catch (const std::exception&) {
                throw std::runtime_error("read_returns_csv: " + path + ":" +
                                         std::to_string(lineno) + " column '" + data.names[j] +
                                         "' is not a number: '" + tok + "'");
            }
            if (consumed != tok.size()) {
                throw std::runtime_error("read_returns_csv: " + path + ":" +
                                         std::to_string(lineno) + " column '" + data.names[j] +
                                         "' has trailing junk: '" + tok + "'");
            }
            if (!std::isfinite(v)) {
                throw std::runtime_error("read_returns_csv: " + path + ":" +
                                         std::to_string(lineno) + " column '" + data.names[j] +
                                         "' is not finite: '" + tok + "'");
            }
            row[j] = v;
        }
        rows.push_back(std::move(row));
    }

    if (rows.empty()) throw std::runtime_error("read_returns_csv: '" + path + "' has no data rows");

    const std::size_t t = rows.size();
    data.values = Matrix(t, n);
    for (std::size_t i = 0; i < t; ++i) {
        for (std::size_t j = 0; j < n; ++j) data.values(i, j) = rows[i][j];
    }
    return data;
}

}  // namespace rnf
