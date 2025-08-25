// Benchmark LPM & UPM using your unmodified header implementation.
// Prints timings in milliseconds for a sample of size 12,000,000 by default.
//
// Build (GCC/Clang):
//   g++ -O3 -DNDEBUG -march=native -std=c++20 bench_lpm_upm.cpp -o bench
//
// Run (defaults: N=12000000, degree=2.0, target=0.0):
//   ./bench
//   ./bench 12000000 2.0 0.0
//
// Notes:
// - Uses normal(0,1) data with a fixed seed for reproducibility.
// - Reports best-of-3 timings to reduce noise.
// - Includes your header exactly; no edits to your code.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "include/NNS/core/partial_moments.hpp"  // <-- put YOUR header (exact code you sent) here

using clock_type = std::chrono::steady_clock;

template <typename F>
static long long time_ms(F&& fn) {
    auto t0 = clock_type::now();
    fn();
    auto t1 = clock_type::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
}

template <typename F>
static long long bench_best_ms(F&& fn, int reps = 3) {
    long long best = std::numeric_limits<long long>::max();
    for (int i = 0; i < reps; ++i) best = std::min(best, time_ms(fn));
    return best;
}

int main(int argc, char** argv) {
    // Defaults to match the Python note ("The sample size when 12,000,000")
    std::size_t N = 12'000'000ULL;
    double degree = 2.0;
    double target = 0.0;

    if (argc > 1) N = static_cast<std::size_t>(std::stoll(argv[1]));
    if (argc > 2) degree = std::stod(argv[2]);
    if (argc > 3) target = std::stod(argv[3]);

    // Generate input data: N samples ~ N(0,1) with fixed seed
    std::vector<double> x(N);
    std::mt19937_64 rng(123456789ULL);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (std::size_t i = 0; i < N; ++i) x[i] = dist(rng);

    // Warm-up one pass (avoid cold-start skew)
    volatile double guard = 0.0;
    guard += NNS::LPM(degree, x, target);
    guard += NNS::UPM(degree, x, target);

    double lpm_val = 0.0, upm_val = 0.0;

    // Time LPM
    const auto lpm_ms = bench_best_ms([&] {
        lpm_val = NNS::LPM(degree, x, target);
    });

    // Time UPM
    const auto upm_ms = bench_best_ms([&] {
        upm_val = NNS::UPM(degree, x, target);
    });

    // Output in the same style
    std::cout << "===== LPM & UPM Performance (ms) =====\n";
    std::cout << "C++ (NNS) LPM: " << lpm_ms << " ms, UPM: " << upm_ms << " ms\n";
    std::cout << "The sample size when " << N << "\n\n";

    // Show the numerical results once (prevents over-optimization; sanity check)
    std::cout << std::setprecision(10);
    std::cout << "Results (degree=" << degree << ", target=" << target << ")\n";
    std::cout << "LPM = " << lpm_val << "\n";
    std::cout << "UPM = " << upm_val << "\n";

    (void)guard; // keep optimizer from dropping warm-up
    return 0;
}