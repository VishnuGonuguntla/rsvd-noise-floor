// Minimal --key value command-line parsing for the apps.
#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace rnf {

class Args {
  public:
    Args(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string tok = argv[i];
            if (tok.rfind("--", 0) != 0) {
                throw std::invalid_argument("unexpected argument '" + tok +
                                            "' (options look like --name value)");
            }
            std::string key = tok.substr(2);
            // Support --key=value as well as --key value.
            const auto eq = key.find('=');
            if (eq != std::string::npos) {
                values_[key.substr(0, eq)] = key.substr(eq + 1);
                continue;
            }
            if (key == "help") { values_["help"] = "1"; continue; }
            if (i + 1 >= argc) throw std::invalid_argument("option --" + key + " needs a value");
            values_[key] = argv[++i];
        }
    }

    [[nodiscard]] bool has(std::string_view k) const {
        return values_.find(std::string(k)) != values_.end();
    }

    [[nodiscard]] std::string str(std::string_view k, std::string fallback) const {
        auto it = values_.find(std::string(k));
        return it == values_.end() ? std::move(fallback) : it->second;
    }

    [[nodiscard]] double real(std::string_view k, double fallback) const {
        auto it = values_.find(std::string(k));
        if (it == values_.end()) return fallback;
        try {
            return std::stod(it->second);
        } catch (const std::exception&) {
            throw std::invalid_argument("option --" + std::string(k) +
                                        " expects a number, got '" + it->second + "'");
        }
    }

    [[nodiscard]] std::size_t count(std::string_view k, std::size_t fallback) const {
        auto it = values_.find(std::string(k));
        if (it == values_.end()) return fallback;
        try {
            const long long v = std::stoll(it->second);
            if (v < 0) throw std::out_of_range("negative");
            return static_cast<std::size_t>(v);
        } catch (const std::exception&) {
            throw std::invalid_argument("option --" + std::string(k) +
                                        " expects a non-negative integer, got '" +
                                        it->second + "'");
        }
    }

    [[nodiscard]] std::uint64_t seed(std::string_view k, std::uint64_t fallback) const {
        auto it = values_.find(std::string(k));
        if (it == values_.end()) return fallback;
        return static_cast<std::uint64_t>(std::stoull(it->second));
    }

    // Comma-separated list, e.g. --sweep 250,500,1000
    [[nodiscard]] std::vector<std::size_t> counts(std::string_view k,
                                                  std::vector<std::size_t> fallback) const {
        auto it = values_.find(std::string(k));
        if (it == values_.end()) return fallback;
        std::vector<std::size_t> out;
        std::stringstream ss(it->second);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (item.empty()) continue;
            out.push_back(static_cast<std::size_t>(std::stoull(item)));
        }
        if (out.empty()) {
            throw std::invalid_argument("option --" + std::string(k) + " parsed to an empty list");
        }
        return out;
    }

    // "lo:hi:step"
    [[nodiscard]] std::vector<double> range(std::string_view k,
                                            std::vector<double> fallback) const {
        auto it = values_.find(std::string(k));
        if (it == values_.end()) return fallback;
        double lo = 0, hi = 0, step = 0;
        char   c1 = 0, c2 = 0;
        std::stringstream ss(it->second);
        ss >> lo >> c1 >> hi >> c2 >> step;
        if (!ss || c1 != ':' || c2 != ':' || step <= 0.0 || hi < lo) {
            throw std::invalid_argument("option --" + std::string(k) +
                                        " expects lo:hi:step, got '" + it->second + "'");
        }
        std::vector<double> out;
        for (double v = lo; v <= hi + 1e-12; v += step) out.push_back(v);
        return out;
    }

  private:
    std::map<std::string, std::string> values_;
};

}  // namespace rnf
