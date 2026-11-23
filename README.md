# rsvd-noise-floor

Randomized SVD of a financial correlation matrix, validated against
Marchenko–Pastur. C++20, no required dependencies.

A sample correlation matrix is mostly noise: with `q = N/T`, random-matrix
theory says the bulk of its eigenvalues follow the Marchenko–Pastur law even
when the true correlation is the identity. Only a handful of eigenvalues carry
signal — yet a full eigendecomposition costs O(N³) and spends nearly all of it
resolving the noise.

This project pairs randomized SVD (O(N²k) for the top-k) with the MP law, which
supplies the rank and certifies that what was discarded really was noise. At
N = 1600 that is **77× faster** than a dense spectrum, and the BBP phase
transition shows the truncation gives up nothing recoverable.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
ctest --test-dir build --output-on-failure
./build/rnf_mp_null --N 1000 --T 4000
```

**[docs/notes.md](docs/notes.md)** is the write-up: the mathematics, the
results with figures, the performance work, and the six things that were not
obvious.
