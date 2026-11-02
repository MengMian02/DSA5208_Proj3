#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int D = 54;
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

std::vector<double> read_data(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open: " + path);
    std::vector<double> x;
    std::string line;
    size_t rows = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const char* p = line.c_str();
        for (int j = 0; j < D + 1; ++j) {
            char* end = nullptr;
            double value = std::strtod(p, &end);
            if (end == p || !std::isfinite(value))
                throw std::runtime_error("Invalid number on row " + std::to_string(rows + 1));
            if (j < D) x.push_back(value); // Column 55 is the known class label.
            if (j < D && *end != ',')
                throw std::runtime_error("Missing column on row " + std::to_string(rows + 1));
            if (j == D && *end != '\0' && *end != '\r')
                throw std::runtime_error("Extra column on row " + std::to_string(rows + 1));
            p = end + (j < D ? 1 : 0);
        }
        ++rows;
    }
    if (rows == 0) throw std::runtime_error("Empty dataset");
    return x;
}

void scale_continuous(std::vector<double>& x) {
    const size_t n = x.size() / D;
    for (int j = 0; j < 10; ++j) {
        double mean = 0.0, m2 = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double v = x[i * D + j];
            const double delta = v - mean;
            mean += delta / double(i + 1);
            m2 += delta * (v - mean);
        }
        const double sd = std::sqrt(m2 / double(n));
        for (size_t i = 0; i < n; ++i)
            x[i * D + j] = sd > 0 ? (x[i * D + j] - mean) / sd : 0.0;
    }
}

double distance2(const double* a, const double* b) {
    double sum = 0.0;
    for (int j = 0; j < D; ++j) {
        const double diff = a[j] - b[j];
        sum += diff * diff;
    }
    return sum;
}

std::vector<double> initialize(const std::vector<double>& x, int k, uint64_t seed) {
    const size_t n = x.size() / D;
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> first(0, n - 1);
    std::vector<double> c(size_t(k) * D), nearest(n, std::numeric_limits<double>::infinity());
    const size_t first_index = first(rng);
    std::copy_n(&x[first_index * D], D, c.data());
    for (int cluster = 1; cluster < k; ++cluster) {
        double total = 0.0;
        for (size_t i = 0; i < n; ++i) {
            nearest[i] = std::min(nearest[i], distance2(&x[i * D], &c[size_t(cluster - 1) * D]));
            total += nearest[i];
        }
        if (total <= 0.0) throw std::runtime_error("Fewer distinct points than K");
        std::uniform_real_distribution<double> draw(0.0, total);
        const double target = draw(rng);
        double cumulative = 0.0;
        size_t selected = n - 1;
        for (size_t i = 0; i < n; ++i) {
            cumulative += nearest[i];
            if (cumulative >= target) { selected = i; break; }
        }
        std::copy_n(&x[selected * D], D, &c[size_t(cluster) * D]);
    }
    return c;
}

double final_inertia(const std::vector<double>& x, const std::vector<double>& c, int k) {
    double sse = 0.0;
    for (size_t i = 0, n = x.size() / D; i < n; ++i) {
        double best = std::numeric_limits<double>::infinity();
        for (int cluster = 0; cluster < k; ++cluster)
            best = std::min(best, distance2(&x[i * D], &c[size_t(cluster) * D]));
        sse += best;
    }
    return sse;
}

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 5) {
            std::cerr << "Usage: kmeans_serial DATA.csv [K=7] [MAX_ITER=100] [SEED=42]\n";
            return 2;
        }
        const int k = argc > 2 ? std::stoi(argv[2]) : 7;
        const int max_iter = argc > 3 ? std::stoi(argv[3]) : 100;
        const uint64_t seed = argc > 4 ? std::stoull(argv[4]) : 42;
        if (k < 1 || max_iter < 1) throw std::runtime_error("K and MAX_ITER must be positive");
        auto t0 = Clock::now();
        std::vector<double> x = read_data(argv[1]);
        auto t1 = Clock::now();
        const size_t n = x.size() / D;
        if (size_t(k) > n) throw std::runtime_error("K exceeds number of rows");
        scale_continuous(x);
        auto t2 = Clock::now();
        std::vector<double> c = initialize(x, k, seed);
        auto t3 = Clock::now();
        std::vector<int> labels(n, -1);
        std::vector<double> sums(size_t(k) * D);
        std::vector<size_t> counts(k);
        int iterations = 0;
        bool converged = false;
        for (int iter = 0; iter < max_iter; ++iter) {
            std::fill(sums.begin(), sums.end(), 0.0);
            std::fill(counts.begin(), counts.end(), 0);
            size_t changed = 0;
            for (size_t i = 0; i < n; ++i) {
                int best_id = 0;
                double best = distance2(&x[i * D], c.data());
                for (int cluster = 1; cluster < k; ++cluster) {
                    const double d = distance2(&x[i * D], &c[size_t(cluster) * D]);
                    if (d < best) { best = d; best_id = cluster; }
                }
                changed += labels[i] != best_id;
                labels[i] = best_id;
                ++counts[best_id];
                for (int j = 0; j < D; ++j) sums[size_t(best_id) * D + j] += x[i * D + j];
            }
            for (int cluster = 0; cluster < k; ++cluster)
                if (counts[cluster])
                    for (int j = 0; j < D; ++j)
                        c[size_t(cluster) * D + j] = sums[size_t(cluster) * D + j] / double(counts[cluster]);
            iterations = iter + 1;
            if (changed == 0) { converged = true; break; }
        }
        auto t4 = Clock::now();
        const double inertia = final_inertia(x, c, k);
        auto t5 = Clock::now();
        std::cout << std::fixed << std::setprecision(6)
                  << "rows=" << n << " features=" << D << " k=" << k
                  << " max_iter=" << max_iter << " seed=" << seed << '\n'
                  << "iterations=" << iterations << " inertia=" << inertia << '\n'
                  << "load_s=" << seconds(t0, t1) << " preprocess_s=" << seconds(t1, t2)
                  << " init_s=" << seconds(t2, t3) << " fit_s=" << seconds(t3, t4)
                  << " validation_s=" << seconds(t4, t5)
                  << " total_s=" << seconds(t0, t5) << '\n';
        std::cout << "cluster_sizes=";
        for (int cluster = 0; cluster < k; ++cluster)
            std::cout << (cluster ? "," : "") << counts[cluster];
        std::cout << '\n';
        uint64_t label_hash = 1469598103934665603ULL;
        for (int label : labels) {
            label_hash ^= uint64_t(label + 1);
            label_hash *= 1099511628211ULL;
        }
        std::cout << "label_hash=" << label_hash << '\n';
        std::cout << "csv_header=implementation,threads,rows,features,k,iterations,fit_s,sse,converged\n"
                  << "csv=serial,1," << n << ',' << D << ',' << k << ',' << iterations << ','
                  << seconds(t3, t4) << ',' << inertia << ','
                  << (converged ? "true" : "false") << '\n';
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
