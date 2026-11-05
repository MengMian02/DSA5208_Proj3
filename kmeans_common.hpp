#pragma once
// Shared Covertype K-means core used by every CPU implementation.
// Conventions preserved from the original programs: C++17, D=54, contiguous
// row-major std::vector<double>, z-score of the first 10 columns, k-means++
// with std::mt19937_64, squared Euclidean distance, label-change convergence,
// empty clusters retain their previous centroid.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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
constexpr int CONTINUOUS_FEATURES = 10;

using Clock = std::chrono::steady_clock;

inline double seconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

inline const char* bool_str(bool value) { return value ? "true" : "false"; }

struct Options {
    std::string program = "kmeans";
    std::string data_path;
    int k = 7;
    int max_iter = 100;
    uint64_t seed = 42;
    uint64_t rows = 0;            // 0 = all rows; otherwise systematic subset size
    bool fixed_iterations = false;
    std::string centroid_path;    // where to write final centroids
    std::string init_centroid_path; // read fixed initial centroids instead of k-means++
    uint64_t weak_per_rank = 0;   // >0 enables deterministic weak-scaling generation
    std::string source_bin;       // optional cached binary copy of the preprocessed source
    std::string dump_generated;   // weak mode: write the generated local rows for verification
};

inline void print_usage(const std::string& program) {
    std::cerr
        << "Usage: " << program << " DATA.csv [K=7] [MAX_ITER=100] [SEED=42]\n"
        << "         [--fixed-iterations] [--centroids FILE] [--rows N]\n"
        << "         [--init-centroids FILE] [--weak-per-rank R]\n"
        << "         [--source-bin FILE] [--dump-generated FILE]\n";
}

inline Options parse_options(int argc, char** argv, const std::string& program) {
    if (argc < 2) {
        print_usage(program);
        throw std::runtime_error("missing dataset path");
    }
    Options options;
    options.program = program;
    options.data_path = argv[1];
    if (argc > 2) options.k = std::stoi(argv[2]);
    if (argc > 3) options.max_iter = std::stoi(argv[3]);
    if (argc > 4) options.seed = std::stoull(argv[4]);
    for (int i = 5; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--fixed-iterations") {
            options.fixed_iterations = true;
        } else if (option == "--centroids" && i + 1 < argc) {
            options.centroid_path = argv[++i];
        } else if (option == "--init-centroids" && i + 1 < argc) {
            options.init_centroid_path = argv[++i];
        } else if (option == "--rows" && i + 1 < argc) {
            options.rows = std::stoull(argv[++i]);
        } else if (option == "--weak-per-rank" && i + 1 < argc) {
            options.weak_per_rank = std::stoull(argv[++i]);
        } else if (option == "--source-bin" && i + 1 < argc) {
            options.source_bin = argv[++i];
        } else if (option == "--dump-generated" && i + 1 < argc) {
            options.dump_generated = argv[++i];
        } else {
            print_usage(program);
            throw std::runtime_error("Unknown or incomplete option: " + option);
        }
    }
    if (options.k < 1 || options.max_iter < 1)
        throw std::runtime_error("K and MAX_ITER must be positive");
    if (options.rows == 1)
        throw std::runtime_error("--rows must be 0 (all) or at least 2");
    if (options.weak_per_rank == 1)
        throw std::runtime_error("--weak-per-rank must be 0 (off) or at least 2");
    return options;
}

// Systematic (evenly spaced) subsampling keeps class proportions of the
// class-ordered Covertype file. Selection is a pure function of
// (source_rows, target_rows), so every implementation and every MPI rank
// configuration observes the identical subset in the identical order.
inline std::vector<size_t> subset_source_indices(size_t source_rows, size_t target_rows) {
    std::vector<size_t> indices;
    if (target_rows == 0 || target_rows >= source_rows) {
        indices.resize(source_rows);
        for (size_t i = 0; i < source_rows; ++i) indices[i] = i;
        return indices;
    }
    indices.reserve(target_rows);
    for (size_t j = 0; j < target_rows; ++j) {
        size_t src = size_t((double(j) + 0.5) * double(source_rows) / double(target_rows));
        if (src >= source_rows) src = source_rows - 1;
        indices.push_back(src);
    }
    return indices;
}

inline std::vector<double> read_data(const std::string& path, uint64_t row_limit = 0) {
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
    const size_t source_rows = rows;
    if (row_limit == 0 || row_limit >= source_rows) return x;

    const std::vector<size_t> keep = subset_source_indices(source_rows, size_t(row_limit));
    for (size_t j = 0; j < keep.size(); ++j) {
        const size_t src = keep[j];
        if (src == j) continue;
        std::copy_n(&x[src * D], D, &x[j * D]); // indices are increasing: forward copy is safe
    }
    x.resize(keep.size() * D);
    return x;
}

inline void scale_continuous(std::vector<double>& x) {
    const size_t n = x.size() / D;
    for (int j = 0; j < CONTINUOUS_FEATURES; ++j) {
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

inline double distance2(const double* a, const double* b) {
    double sum = 0.0;
    for (int j = 0; j < D; ++j) {
        const double diff = a[j] - b[j];
        sum += diff * diff;
    }
    return sum;
}

inline std::vector<double> initialize(const std::vector<double>& x, int k, uint64_t seed) {
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

inline void update_centroids(std::vector<double>& centroids,
                             const std::vector<double>& sums,
                             const std::vector<uint64_t>& counts,
                             int k) {
    for (int cluster = 0; cluster < k; ++cluster) {
        if (counts[cluster])
            for (int j = 0; j < D; ++j)
                centroids[size_t(cluster) * D + j] =
                    sums[size_t(cluster) * D + j] / double(counts[cluster]);
        // Empty clusters retain their previous centroid.
    }
}

inline double local_inertia(const std::vector<double>& x, const std::vector<double>& c, int k) {
    double sse = 0.0;
    for (size_t i = 0, n = x.size() / D; i < n; ++i) {
        double best = std::numeric_limits<double>::infinity();
        for (int cluster = 0; cluster < k; ++cluster)
            best = std::min(best, distance2(&x[i * D], &c[size_t(cluster) * D]));
        sse += best;
    }
    return sse;
}

inline double final_inertia(const std::vector<double>& x, const std::vector<double>& c, int k) {
    return local_inertia(x, c, k);
}

inline void write_centroids(const std::string& path, const std::vector<double>& c, int k) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open centroid output: " + path);
    out << std::setprecision(17);
    for (int cluster = 0; cluster < k; ++cluster) {
        for (int j = 0; j < D; ++j)
            out << (j ? "," : "") << c[size_t(cluster) * D + j];
        out << '\n';
    }
}

inline std::vector<double> read_centroids(const std::string& path, int k) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open initial centroids: " + path);
    std::vector<double> c;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const char* p = line.c_str();
        for (int j = 0; j < D; ++j) {
            char* end = nullptr;
            const double value = std::strtod(p, &end);
            if (end == p) throw std::runtime_error("Malformed centroid row in " + path);
            c.push_back(value);
            p = end + (*end == ',' ? 1 : 0);
        }
    }
    if (c.size() != size_t(k) * D)
        throw std::runtime_error("Initial centroid file must hold K rows of D values");
    return c;
}

// FNV-1a over the label vector, identical to the original programs.
inline uint64_t label_hash(const std::vector<int>& labels) {
    uint64_t hash = 1469598103934665603ULL;
    for (int label : labels) {
        hash ^= uint64_t(label + 1);
        hash *= 1099511628211ULL;
    }
    return hash;
}

// ---- deterministic weak-scaling workload generation ----------------------
// splitmix64 is a fixed, self-contained integer mixer (no std::hash, no
// library-dependent behaviour), so the mapping from a global observation
// index to a source Covertype row is reproducible on every platform.
inline uint64_t splitmix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

inline uint64_t weak_source_index(uint64_t seed, uint64_t global_index, uint64_t source_rows) {
    const uint64_t stream = splitmix64(seed ^ 0x5DEECE66DULL);
    const uint64_t mixed = splitmix64(global_index * 0x9E3779B97F4A7C15ULL ^ stream);
    return mixed % source_rows;
}

// Deterministically materialise global indices [begin, begin + count) from the
// preprocessed source matrix. The logical expanded dataset depends only on
// (seed, count_total, source_rows), never on the MPI rank count.
inline void materialise_indices(const std::vector<double>& source,
                                uint64_t source_rows,
                                uint64_t seed,
                                uint64_t begin,
                                uint64_t count,
                                std::vector<double>& out) {
    out.resize(size_t(count) * D);
    for (uint64_t i = 0; i < count; ++i) {
        const uint64_t src = weak_source_index(seed, begin + i, source_rows);
        std::copy_n(&source[size_t(src) * D], D, &out[size_t(i) * D]);
    }
}

// ---- cached binary source ------------------------------------------------
inline bool load_source_binary(const std::string& path, std::vector<double>& x, uint64_t& rows) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    uint64_t header[2] = {0, 0};
    in.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!in || header[1] != uint64_t(D)) return false;
    x.resize(size_t(header[0]) * D);
    in.read(reinterpret_cast<char*>(x.data()), std::streamsize(x.size() * sizeof(double)));
    if (!in) { x.clear(); return false; }
    rows = header[0];
    return true;
}

inline void save_source_binary(const std::string& path, const std::vector<double>& x) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Cannot write source binary: " + path);
    const uint64_t header[2] = {uint64_t(x.size() / D), uint64_t(D)};
    out.write(reinterpret_cast<const char*>(header), sizeof(header));
    out.write(reinterpret_cast<const char*>(x.data()), std::streamsize(x.size() * sizeof(double)));
    if (!out) throw std::runtime_error("Failed writing source binary: " + path);
}

// ---- shared reporting ----------------------------------------------------
inline void print_dataset_line(uint64_t n, int k, int max_iter, uint64_t seed) {
    std::cout << std::fixed << std::setprecision(6)
              << "rows=" << n << " features=" << D << " k=" << k
              << " max_iter=" << max_iter << " seed=" << seed << '\n';
}

inline void print_iteration_line(int iterations, double inertia) {
    std::cout << std::fixed << std::setprecision(6)
              << "iterations=" << iterations << " inertia=" << inertia << '\n';
}

inline void print_timing_line(double load_s, double preprocess_s, double init_s,
                              double fit_s, double validation_s, double total_s) {
    std::cout << std::fixed << std::setprecision(6)
              << "load_s=" << load_s << " preprocess_s=" << preprocess_s
              << " init_s=" << init_s << " fit_s=" << fit_s
              << " validation_s=" << validation_s << " total_s=" << total_s << '\n';
}

template <typename Count>
inline void print_cluster_sizes(const Count& counts, int k) {
    std::cout << "cluster_sizes=";
    for (int cluster = 0; cluster < k; ++cluster)
        std::cout << (cluster ? "," : "") << counts[cluster];
    std::cout << '\n';
}
