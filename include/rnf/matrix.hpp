// Column-major dense matrix with 64-byte-aligned storage.
//
// Column-major throughout the project so that the GEMM micro-kernel, the QR
// panel layout and the eigensolver all agree on stride conventions. The
// leading dimension is padded to a multiple of 8 doubles (one cache line) so
// that every column starts aligned -- which is what lets the AVX2 kernel use
// aligned loads on column heads.
#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <utility>

namespace rnf {

inline constexpr std::size_t kAlignBytes   = 64;
inline constexpr std::size_t kAlignDoubles = kAlignBytes / sizeof(double);

namespace detail {

struct AlignedFree {
    void operator()(double* p) const noexcept { std::free(p); }
};

using AlignedPtr = std::unique_ptr<double[], AlignedFree>;

inline AlignedPtr aligned_doubles(std::size_t n) {
    if (n == 0) return AlignedPtr{};
    // std::aligned_alloc requires size to be a multiple of the alignment.
    const std::size_t bytes = ((n * sizeof(double) + kAlignBytes - 1) / kAlignBytes) * kAlignBytes;
    void* raw = std::aligned_alloc(kAlignBytes, bytes);
    if (raw == nullptr) throw std::bad_alloc{};
    return AlignedPtr{static_cast<double*>(raw)};
}

inline std::size_t pad_to(std::size_t n, std::size_t multiple) {
    return ((n + multiple - 1) / multiple) * multiple;
}

}  // namespace detail

// Non-owning view of a column-major block. `ld` is the leading dimension
// (column stride) of the *parent* buffer, so views compose without copying.
struct MatView {
    double*     data = nullptr;
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t ld   = 0;

    double&       operator()(std::size_t i, std::size_t j)       { return data[j * ld + i]; }
    const double& operator()(std::size_t i, std::size_t j) const { return data[j * ld + i]; }
    double*       col(std::size_t j)       { return data + j * ld; }
    const double* col(std::size_t j) const { return data + j * ld; }

    [[nodiscard]] MatView block(std::size_t i0, std::size_t j0,
                                std::size_t nr, std::size_t nc) const {
        assert(i0 + nr <= rows && j0 + nc <= cols);
        return MatView{data + j0 * ld + i0, nr, nc, ld};
    }
};

struct ConstMatView {
    const double* data = nullptr;
    std::size_t   rows = 0;
    std::size_t   cols = 0;
    std::size_t   ld   = 0;

    ConstMatView() = default;
    ConstMatView(const double* d, std::size_t r, std::size_t c, std::size_t l)
        : data(d), rows(r), cols(c), ld(l) {}
    // NOLINTNEXTLINE(google-explicit-constructor) -- implicit mutable->const is intended
    ConstMatView(const MatView& v) : data(v.data), rows(v.rows), cols(v.cols), ld(v.ld) {}

    const double& operator()(std::size_t i, std::size_t j) const { return data[j * ld + i]; }
    const double* col(std::size_t j) const { return data + j * ld; }

    [[nodiscard]] ConstMatView block(std::size_t i0, std::size_t j0,
                                     std::size_t nr, std::size_t nc) const {
        assert(i0 + nr <= rows && j0 + nc <= cols);
        return ConstMatView{data + j0 * ld + i0, nr, nc, ld};
    }
};

class Matrix {
  public:
    Matrix() = default;

    Matrix(std::size_t rows, std::size_t cols)
        : rows_(rows), cols_(cols), ld_(detail::pad_to(rows, kAlignDoubles)) {
        if (rows_ == 0 || cols_ == 0) { rows_ = cols_ = ld_ = 0; return; }
        storage_ = detail::aligned_doubles(ld_ * cols_);
        std::fill_n(storage_.get(), ld_ * cols_, 0.0);
    }

    Matrix(const Matrix& other)
        : rows_(other.rows_), cols_(other.cols_), ld_(other.ld_) {
        if (other.storage_) {
            storage_ = detail::aligned_doubles(ld_ * cols_);
            std::copy_n(other.storage_.get(), ld_ * cols_, storage_.get());
        }
    }
    Matrix& operator=(const Matrix& other) {
        if (this != &other) { Matrix tmp(other); swap(tmp); }
        return *this;
    }
    Matrix(Matrix&&) noexcept            = default;
    Matrix& operator=(Matrix&&) noexcept = default;
    ~Matrix()                            = default;

    void swap(Matrix& other) noexcept {
        std::swap(rows_, other.rows_);
        std::swap(cols_, other.cols_);
        std::swap(ld_, other.ld_);
        std::swap(storage_, other.storage_);
    }

    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] std::size_t ld()   const noexcept { return ld_; }
    [[nodiscard]] bool        empty() const noexcept { return rows_ == 0 || cols_ == 0; }

    double&       operator()(std::size_t i, std::size_t j)       { return storage_[j * ld_ + i]; }
    const double& operator()(std::size_t i, std::size_t j) const { return storage_[j * ld_ + i]; }

    double*       data()       noexcept { return storage_.get(); }
    const double* data() const noexcept { return storage_.get(); }
    double*       col(std::size_t j)       noexcept { return storage_.get() + j * ld_; }
    const double* col(std::size_t j) const noexcept { return storage_.get() + j * ld_; }

    [[nodiscard]] std::span<double> column_span(std::size_t j) {
        return {col(j), rows_};
    }
    [[nodiscard]] std::span<const double> column_span(std::size_t j) const {
        return {col(j), rows_};
    }

    // NOLINTNEXTLINE(google-explicit-constructor)
    operator MatView()      { return MatView{storage_.get(), rows_, cols_, ld_}; }
    // NOLINTNEXTLINE(google-explicit-constructor)
    operator ConstMatView() const { return ConstMatView{storage_.get(), rows_, cols_, ld_}; }

    [[nodiscard]] MatView      view()       { return MatView{storage_.get(), rows_, cols_, ld_}; }
    [[nodiscard]] ConstMatView view() const { return ConstMatView{storage_.get(), rows_, cols_, ld_}; }

    [[nodiscard]] MatView block(std::size_t i0, std::size_t j0,
                                std::size_t nr, std::size_t nc) {
        return view().block(i0, j0, nr, nc);
    }
    [[nodiscard]] ConstMatView block(std::size_t i0, std::size_t j0,
                                     std::size_t nr, std::size_t nc) const {
        return view().block(i0, j0, nr, nc);
    }

    void fill(double v) {
        for (std::size_t j = 0; j < cols_; ++j) std::fill_n(col(j), rows_, v);
    }
    void set_zero() { fill(0.0); }

    void set_identity() {
        set_zero();
        for (std::size_t i = 0; i < std::min(rows_, cols_); ++i) (*this)(i, i) = 1.0;
    }

    static Matrix identity(std::size_t n) {
        Matrix m(n, n);
        m.set_identity();
        return m;
    }

    // Deep copy of the logical rows x cols content into fresh storage.
    [[nodiscard]] Matrix clone() const { return Matrix(*this); }

  private:
    std::size_t         rows_ = 0;
    std::size_t         cols_ = 0;
    std::size_t         ld_   = 0;
    detail::AlignedPtr  storage_{};
};

inline void swap(Matrix& a, Matrix& b) noexcept { a.swap(b); }

}  // namespace rnf
