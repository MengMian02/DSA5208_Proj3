#include <algorithm>
#include <climits>
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

#include <mpi.h>

constexpr int D = 54;

struct Options {
    std::string data_path;
    int k = 7;
    int max_iter = 100;
    uint64_t seed = 42;
    bool fixed_iterations = false;
    std::string centroid_path;
};

Options parse_options(int argc, char** argv) {
    if (argc < 2) {
        throw std::runtime_error(
            "Usage: kmeans_mpi DATA.csv [K=7] [MAX_ITER=100] [SEED=42]"
            " [--fixed-iterations] [--centroids FILE]");
    }
    Options options;
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
        } else {
            throw std::runtime_error("Unknown or incomplete option: " + option);
        }
    }
    if (options.k < 1 || options.max_iter < 1)
        throw std::runtime_error("K and MAX_ITER must be positive");
    return options;
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
            nearest[i] = std::min(
                nearest[i], distance2(&x[i * D], &c[size_t(cluster - 1) * D]));
            total += nearest[i];
        }
        if (total <= 0.0) throw std::runtime_error("Fewer distinct points than K");
        std::uniform_real_distribution<double> draw(0.0, total);
        const double target = draw(rng);
        double cumulative = 0.0;
        size_t selected = n - 1;
        for (size_t i = 0; i < n; ++i) {
            cumulative += nearest[i];
            if (cumulative >= target) {
                selected = i;
                break;
            }
        }
        std::copy_n(&x[selected * D], D, &c[size_t(cluster) * D]);
    }
    return c;
}

double local_inertia(const std::vector<double>& x, const std::vector<double>& c, int k) {
    double sse = 0.0;
    for (size_t i = 0, n = x.size() / D; i < n; ++i) {
        double best = std::numeric_limits<double>::infinity();
        for (int cluster = 0; cluster < k; ++cluster)
            best = std::min(best, distance2(&x[i * D], &c[size_t(cluster) * D]));
        sse += best;
    }
    return sse;
}

void update_centroids(std::vector<double>& centroids,
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

void write_centroids(const std::string& path, const std::vector<double>& c, int k) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open centroid output: " + path);
    out << std::setprecision(17);
    for (int cluster = 0; cluster < k; ++cluster) {
        for (int j = 0; j < D; ++j)
            out << (j ? "," : "") << c[size_t(cluster) * D + j];
        out << '\n';
    }
}

void check_mpi(int code, const char* operation) {
    if (code != MPI_SUCCESS) throw std::runtime_error(std::string("MPI failure in ") + operation);
}

void run(const Options& options, int rank, int ranks) {
    const double program_start = MPI_Wtime();
    uint64_t n = 0;
    std::vector<double> x;
    std::vector<double> centroids;
    double load_s = 0.0;
    double preprocess_s = 0.0;
    double init_s = 0.0;

    // Only rank 0 reads and preprocesses the complete matrix. Other ranks
    // receive their rows, so no rank except rank 0 ever holds all observations.
    if (rank == 0) {
        const double load_start = MPI_Wtime();
        x = read_data(options.data_path);
        const double load_end = MPI_Wtime();
        n = x.size() / D;
        if (uint64_t(options.k) > n) throw std::runtime_error("K exceeds number of rows");
        scale_continuous(x);
        const double preprocess_end = MPI_Wtime();
        centroids = initialize(x, options.k, options.seed);
        const double init_end = MPI_Wtime();
        load_s = load_end - load_start;
        preprocess_s = preprocess_end - load_end;
        init_s = init_end - preprocess_end;
    }

    const double distribution_start = MPI_Wtime();
    check_mpi(MPI_Bcast(&n, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD), "dataset size broadcast");
    if (n > uint64_t(INT_MAX) / uint64_t(D))
        throw std::runtime_error("Dataset is too large for MPI int-count collectives");

    const uint64_t base_rows = n / uint64_t(ranks);
    const uint64_t remainder = n % uint64_t(ranks);
    const uint64_t local_rows64 = base_rows + (uint64_t(rank) < remainder ? 1 : 0);
    const int local_rows = int(local_rows64);
    std::vector<int> row_counts(ranks);
    std::vector<int> element_counts(ranks);
    std::vector<int> element_displacements(ranks);
    std::vector<int> row_displacements(ranks);
    uint64_t displacement_rows = 0;
    for (int r = 0; r < ranks; ++r) {
        const int rows = int(base_rows + (uint64_t(r) < remainder ? 1 : 0));
        row_counts[r] = rows;
        element_counts[r] = rows * D;
        element_displacements[r] = int(displacement_rows * uint64_t(D));
        row_displacements[r] = int(displacement_rows);
        displacement_rows += uint64_t(rows);
    }

    std::vector<double> local_x(size_t(local_rows) * D);
    check_mpi(
        MPI_Scatterv(rank == 0 ? x.data() : nullptr,
                     rank == 0 ? element_counts.data() : nullptr,
                     rank == 0 ? element_displacements.data() : nullptr,
                     MPI_DOUBLE,
                     local_x.data(),
                     local_rows * D,
                     MPI_DOUBLE,
                     0,
                     MPI_COMM_WORLD),
        "row distribution");
    if (rank != 0) centroids.resize(size_t(options.k) * D);
    check_mpi(
        MPI_Bcast(centroids.data(), options.k * D, MPI_DOUBLE, 0, MPI_COMM_WORLD),
        "initial centroid broadcast");
    const double distribution_elapsed = MPI_Wtime() - distribution_start;
    double distribution_s = 0.0;
    check_mpi(
        MPI_Reduce(&distribution_elapsed, &distribution_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD),
        "distribution timing reduction");

    // Rank 0 no longer needs the replicated matrix after Scatterv.
    if (rank == 0) std::vector<double>().swap(x);

    std::vector<int> labels(local_rows, -1);
    std::vector<double> local_sums(size_t(options.k) * D);
    std::vector<double> global_sums(size_t(options.k) * D);
    std::vector<uint64_t> local_counts(options.k);
    std::vector<uint64_t> count_stats(options.k + 1);
    std::vector<uint64_t> global_count_stats(options.k + 1);

    int iterations = 0;
    bool converged = false;
    const double fit_start = MPI_Wtime();
    double local_compute_s = 0.0;
    double mpi_reduction_s = 0.0;

    for (int iter = 0; iter < options.max_iter; ++iter) {
        std::fill(local_sums.begin(), local_sums.end(), 0.0);
        std::fill(local_counts.begin(), local_counts.end(), 0);
        uint64_t local_changed = 0;

        const double compute_start = MPI_Wtime();
        for (int i = 0; i < local_rows; ++i) {
            const double* point = &local_x[size_t(i) * D];
            int best_id = 0;
            double best = distance2(point, centroids.data());
            for (int cluster = 1; cluster < options.k; ++cluster) {
                const double d = distance2(point, &centroids[size_t(cluster) * D]);
                if (d < best) {
                    best = d;
                    best_id = cluster;
                }
            }
            local_changed += labels[i] != best_id;
            labels[i] = best_id;
            ++local_counts[best_id];
            for (int j = 0; j < D; ++j)
                local_sums[size_t(best_id) * D + j] += point[j];
        }
        local_compute_s += MPI_Wtime() - compute_start;

        std::copy(local_counts.begin(), local_counts.end(), count_stats.begin());
        count_stats[options.k] = local_changed;
        const double reduction_start = MPI_Wtime();
        check_mpi(
            MPI_Allreduce(local_sums.data(),
                          global_sums.data(),
                          options.k * D,
                          MPI_DOUBLE,
                          MPI_SUM,
                          MPI_COMM_WORLD),
            "cluster sum all-reduce");
        check_mpi(
            MPI_Allreduce(count_stats.data(),
                          global_count_stats.data(),
                          options.k + 1,
                          MPI_UINT64_T,
                          MPI_SUM,
                          MPI_COMM_WORLD),
            "count and changed-label all-reduce");
        mpi_reduction_s += MPI_Wtime() - reduction_start;

        const uint64_t global_changed = global_count_stats[options.k];
        update_centroids(centroids, global_sums, global_count_stats, options.k);

        iterations = iter + 1;
        converged = global_changed == 0;
        if (converged && !options.fixed_iterations) break;
    }

    const double fit_elapsed = MPI_Wtime() - fit_start;
    double fit_stats[3] = {fit_elapsed, local_compute_s, mpi_reduction_s};
    double fit_stats_max[3] = {0.0, 0.0, 0.0};
    check_mpi(
        MPI_Reduce(fit_stats, fit_stats_max, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD),
        "fit timing reduction");

    const double validation_start = MPI_Wtime();
    const double local_sse = local_inertia(local_x, centroids, options.k);
    double inertia = 0.0;
    check_mpi(
        MPI_Reduce(&local_sse, &inertia, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD),
        "inertia reduction");

    // Gathering labels once after fitting preserves the original observation
    // order for an exact serial-compatible hash without communicating labels
    // on every iteration.
    std::vector<int> all_labels;
    if (rank == 0) all_labels.resize(size_t(n), -1);
    check_mpi(
        MPI_Gatherv(labels.data(),
                    local_rows,
                    MPI_INT,
                    rank == 0 ? all_labels.data() : nullptr,
                    rank == 0 ? row_counts.data() : nullptr,
                    rank == 0 ? row_displacements.data() : nullptr,
                    MPI_INT,
                    0,
                    MPI_COMM_WORLD),
        "final label gather");
    const double validation_elapsed = MPI_Wtime() - validation_start;
    double validation_s = 0.0;
    check_mpi(
        MPI_Reduce(&validation_elapsed, &validation_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD),
        "validation timing reduction");
    const double total_elapsed = MPI_Wtime() - program_start;
    double total_s = 0.0;
    check_mpi(
        MPI_Reduce(&total_elapsed, &total_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD),
        "total timing reduction");

    if (rank == 0) {
        if (!options.centroid_path.empty())
            write_centroids(options.centroid_path, centroids, options.k);

        uint64_t label_hash = 1469598103934665603ULL;
        for (int label : all_labels) {
            label_hash ^= uint64_t(label + 1);
            label_hash *= 1099511628211ULL;
        }

        const std::string stopping_mode =
            options.fixed_iterations ? "fixed_iterations" : "convergence";
        std::cout << std::fixed << std::setprecision(6)
                  << "rows=" << n << " features=" << D << " k=" << options.k
                  << " max_iter=" << options.max_iter << " seed=" << options.seed << '\n'
                  << "iterations=" << iterations << " inertia=" << inertia << '\n'
                  << "load_s=" << load_s << " preprocess_s=" << preprocess_s
                  << " init_s=" << init_s << " fit_s=" << fit_stats_max[0]
                  << " validation_s=" << validation_s
                  << " total_s=" << total_s << '\n'
                  << "implementation=mpi ranks=" << ranks << " threads=1"
                  << " total_workers=" << ranks
                  << " stopping_mode=" << stopping_mode
                  << " converged=" << (converged ? "true" : "false") << '\n'
                  << "distribution_s=" << distribution_s
                  << " mpi_reduction_s=" << fit_stats_max[2]
                  << " local_compute_s=" << fit_stats_max[1] << '\n'
                  << "cluster_sizes=";
        for (int cluster = 0; cluster < options.k; ++cluster)
            std::cout << (cluster ? "," : "") << global_count_stats[cluster];
        std::cout << '\n' << "label_hash=" << label_hash << '\n';
        std::cout << "csv_header=implementation,ranks,threads,total_workers,rows,features,k,"
                     "max_iter,iterations,fit_s,sse,converged,stopping_mode,"
                     "distribution_s,mpi_reduction_s,local_compute_s\n"
                  << "csv=mpi," << ranks << ",1," << ranks << ',' << n << ',' << D << ','
                  << options.k << ',' << options.max_iter << ',' << iterations << ','
                  << fit_stats_max[0] << ',' << inertia << ','
                  << (converged ? "true" : "false") << ',' << stopping_mode << ','
                  << distribution_s << ',' << fit_stats_max[2] << ','
                  << fit_stats_max[1] << '\n';
    }
}

int main(int argc, char** argv) {
    int init_code = MPI_Init(&argc, &argv);
    if (init_code != MPI_SUCCESS) {
        std::cerr << "Error: MPI_Init failed\n";
        return 1;
    }

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    try {
        const Options options = parse_options(argc, argv);
        run(options, rank, ranks);
    } catch (const std::exception& e) {
        std::cerr << "Error (rank " << rank << "): " << e.what() << '\n';
        // A rank that fails before a collective can otherwise leave peers
        // blocked forever. MPI_Abort terminates the complete MPI job.
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    int finalize_code = MPI_Finalize();
    if (finalize_code != MPI_SUCCESS) {
        std::cerr << "Error: MPI_Finalize failed\n";
        return 1;
    }
    return 0;
}
