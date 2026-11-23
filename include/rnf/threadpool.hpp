// A small persistent thread pool.
//
// Persistent because the alternative -- spawning threads per GEMM call -- has
// a startup cost comparable to the work itself for the tall-skinny shapes
// RSVD spends its time in. Workers park on a condition variable between jobs
// and are woken by a generation counter.
//
// Deliberately hand-written rather than OpenMP: this project uses the C++
// standard library only, and the pool is small enough to be read in one
// sitting.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace rnf {

class ThreadPool {
  public:
    explicit ThreadPool(std::size_t workers = 0) {
        if (workers == 0) {
            const unsigned hw = std::thread::hardware_concurrency();
            workers = (hw == 0) ? 1 : hw;
        }
        n_ = workers;
        if (n_ <= 1) return;  // run everything on the calling thread

        threads_.reserve(n_ - 1);
        for (std::size_t w = 1; w < n_; ++w) {
            threads_.emplace_back([this, w] { worker_loop(w); });
        }
    }

    ThreadPool(const ThreadPool&)            = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
            ++generation_;
        }
        cv_.notify_all();
        for (auto& t : threads_) {
            if (t.joinable()) t.join();
        }
    }

    [[nodiscard]] std::size_t size() const noexcept { return n_; }

    // Invoke body(worker_index) exactly once on every worker, blocking until
    // all have returned. The calling thread is worker 0, so a single-threaded
    // pool costs nothing but a function call.
    //
    // This is the primitive the GEMM uses, rather than a chunked
    // parallel_for, for two reasons. It lets each worker do its own one-time
    // setup (packing the B panel) before touching shared work, and it leaves
    // the work distribution to the caller -- which matters on a hybrid CPU,
    // where equal static chunks are paced by the slowest E-core and
    // pulling work dynamically is substantially better.
    void parallel_invoke(const std::function<void(std::size_t)>& body) {
        if (n_ <= 1) { body(0); return; }

        {
            std::lock_guard<std::mutex> lk(mu_);
            body_      = &body;
            remaining_ = n_ - 1;
            ++generation_;
        }
        cv_.notify_all();

        body(0);

        std::unique_lock<std::mutex> lk(mu_);
        done_cv_.wait(lk, [this] { return remaining_ == 0; });
        body_ = nullptr;
    }

    // Static equal-chunk split, kept for callers whose work really is
    // uniform. Implemented on top of parallel_invoke.
    void parallel_for(std::size_t count,
                      const std::function<void(std::size_t, std::size_t)>& body) {
        if (count == 0) return;
        if (n_ <= 1 || count == 1) { body(0, count); return; }
        parallel_invoke([&](std::size_t w) {
            const std::size_t per = count / n_;
            const std::size_t rem = count % n_;
            const std::size_t lo  = w * per + (w < rem ? w : rem);
            const std::size_t hi  = lo + per + (w < rem ? 1 : 0);
            if (lo < hi) body(lo, hi);
        });
    }

    // One pool for the whole process. Sizing it per call would defeat the
    // point of making it persistent.
    static ThreadPool& global() { return *global_slot(); }

    // Replace the process-wide pool, for thread-scaling studies. NOT safe to
    // call while any parallel_for is in flight, or from a worker; the
    // benchmark driver is single-threaded at the top level, which is the only
    // place this is used.
    static void resize_global(std::size_t workers) {
        global_slot() = std::make_unique<ThreadPool>(workers);
    }

  private:
    static std::unique_ptr<ThreadPool>& global_slot() {
        static std::unique_ptr<ThreadPool> pool = std::make_unique<ThreadPool>();
        return pool;
    }

    void worker_loop(std::size_t w) {
        std::size_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this, seen] { return generation_ != seen; });
                seen = generation_;
                if (stop_) return;
            }

            (*body_)(w);

            {
                std::lock_guard<std::mutex> lk(mu_);
                --remaining_;
            }
            done_cv_.notify_one();
        }
    }

    std::size_t              n_ = 1;
    std::vector<std::thread> threads_;
    std::mutex               mu_;
    std::condition_variable  cv_;
    std::condition_variable  done_cv_;

    const std::function<void(std::size_t)>* body_ = nullptr;
    std::size_t remaining_  = 0;
    std::size_t generation_ = 0;
    bool        stop_       = false;
};

}  // namespace rnf
