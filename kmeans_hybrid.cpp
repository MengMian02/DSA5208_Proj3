#include "kmeans_common.hpp"

#include <climits>
#include <cstdint>

#include <mpi.h>
#include <omp.h>

void check_mpi(int code, const char* operation) {
    if (code != MPI_SUCCESS) throw std::runtime_error(std::string("MPI failure in ") + operation);
}

// Broadcasts a double buffer of arbitrary length in bounded chunks so that no
// single collective message exceeds the MPI int count limit or stresses the
// transport with one enormous eager message.
void broadcast_vector(std::vector<double>& buffer, uint64_t elements, int root, const char* what) {
    const uint64_t chunk = uint64_t(1) << 22; // 4M doubles = 32 MB per message
    for (uint64_t offset = 0; offset < elements; offset += chunk) {
        const int count = int(std::min(chunk, elements - offset));
        check_mpi(MPI_Bcast(buffer.data() + offset, count, MPI_DOUBLE, root, MPI_COMM_WORLD), what);
    }
}

void run(const Options& options, int rank, int ranks) {
    const double program_start = MPI_Wtime();
    const int k = options.k;
    const bool weak = options.weak_per_rank > 0;

    uint64_t n = 0;          // logical global observation count
    uint64_t source_rows = 0;
    std::vector<double> source;   // preprocessed source matrix (rank 0 / weak mode)
    std::vector<double> centroids;
    std::vector<double> local_x;
    uint64_t local_rows = 0;
    double load_s = 0.0, preprocess_s = 0.0, init_s = 0.0;
    double source_bcast_s = 0.0, generation_s = 0.0;

    const double distribution_start = MPI_Wtime();

    if (!weak) {
        // Only rank 0 reads and preprocesses the complete matrix. Other ranks
        // receive their rows, so no rank except rank 0 ever holds all observations.
        if (rank == 0) {
            const double load_start = MPI_Wtime();
            source = read_data(options.data_path, options.rows);
            const double load_end = MPI_Wtime();
            source_rows = source.size() / D;
            if (uint64_t(k) > source_rows) throw std::runtime_error("K exceeds number of rows");
            scale_continuous(source);
            const double preprocess_end = MPI_Wtime();
            centroids = options.init_centroid_path.empty()
                            ? initialize(source, k, options.seed)
                            : read_centroids(options.init_centroid_path, k);
            const double init_end = MPI_Wtime();
            load_s = load_end - load_start;
            preprocess_s = preprocess_end - load_end;
            init_s = init_end - preprocess_end;
        }
        n = source_rows; // corrected by the broadcast below for all other ranks
        check_mpi(MPI_Bcast(&n, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD), "dataset size broadcast");
        if (n > uint64_t(INT_MAX) / uint64_t(D))
            throw std::runtime_error("Dataset is too large for MPI int-count collectives");
        source_rows = n;

        const uint64_t base_rows = n / uint64_t(ranks);
        const uint64_t remainder = n % uint64_t(ranks);
        local_rows = base_rows + (uint64_t(rank) < remainder ? 1 : 0);

        std::vector<int> row_counts(ranks), element_counts(ranks);
        std::vector<int> element_displacements(ranks), row_displacements(ranks);
        uint64_t displacement_rows = 0;
        for (int r = 0; r < ranks; ++r) {
            const int rows = int(base_rows + (uint64_t(r) < remainder ? 1 : 0));
            row_counts[r] = rows;
            element_counts[r] = rows * D;
            element_displacements[r] = int(displacement_rows * uint64_t(D));
            row_displacements[r] = int(displacement_rows);
            displacement_rows += uint64_t(rows);
        }
        local_x.assign(size_t(local_rows) * D, 0.0);
        check_mpi(MPI_Scatterv(rank == 0 ? source.data() : nullptr,
                              rank == 0 ? element_counts.data() : nullptr,
                              rank == 0 ? element_displacements.data() : nullptr,
                              MPI_DOUBLE, local_x.data(), int(local_rows * uint64_t(D)), MPI_DOUBLE,
                              0, MPI_COMM_WORLD),
                  "row distribution");
        if (rank != 0) centroids.assign(size_t(k) * D, 0.0);
        check_mpi(MPI_Bcast(centroids.data(), k * D, MPI_DOUBLE, 0, MPI_COMM_WORLD),
                  "initial centroid broadcast");
    } else {
        // ---- E3 weak scaling: deterministic local generation --------------
        // The logical expanded dataset is a pure function of
        // (source Covertype, seed, global observation count); each rank
        // materialises only its own contiguous global-index block, so nothing
        // is written to disk and no rank holds the expanded dataset.
        n = options.weak_per_rank * uint64_t(ranks);

        if (rank == 0) {
            const double load_start = MPI_Wtime();
            bool loaded = false;
            if (!options.source_bin.empty())
                loaded = load_source_binary(options.source_bin, source, source_rows);
            if (loaded) {
                load_s = MPI_Wtime() - load_start;
                preprocess_s = 0.0;
            } else {
                source = read_data(options.data_path, options.rows);
                const double load_end = MPI_Wtime();
                source_rows = source.size() / D;
                if (uint64_t(k) > source_rows) throw std::runtime_error("K exceeds number of rows");
                scale_continuous(source);
                const double preprocess_end = MPI_Wtime();
                if (!options.source_bin.empty()) save_source_binary(options.source_bin, source);
                load_s = load_end - load_start;
                preprocess_s = preprocess_end - load_end;
            }
            if (source_rows == 0) throw std::runtime_error("Empty source dataset");
            const double init_start = MPI_Wtime();
            centroids = options.init_centroid_path.empty()
                            ? initialize(source, k, options.seed)
                            : read_centroids(options.init_centroid_path, k);
            init_s = MPI_Wtime() - init_start;
        }
        check_mpi(MPI_Bcast(&source_rows, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD),
                  "source row count broadcast");

        const double bcast_start = MPI_Wtime();
        if (!options.source_bin.empty()) {
            // Every rank reads the shared cached binary: no 250 MB broadcast.
            if (rank != 0) {
                uint64_t rows_from_file = 0;
                if (!load_source_binary(options.source_bin, source, rows_from_file))
                    throw std::runtime_error("Cannot read source binary: " + options.source_bin);
                if (rows_from_file != source_rows)
                    throw std::runtime_error("Source binary row count mismatch");
            }
        } else {
            if (rank != 0) source.assign(size_t(source_rows) * D, 0.0);
            broadcast_vector(source, source_rows * uint64_t(D), 0, "source matrix broadcast");
        }
        source_bcast_s = MPI_Wtime() - bcast_start;

        if (rank != 0) centroids.assign(size_t(k) * D, 0.0);
        check_mpi(MPI_Bcast(centroids.data(), k * D, MPI_DOUBLE, 0, MPI_COMM_WORLD),
                  "initial centroid broadcast");

        const uint64_t base_rows = n / uint64_t(ranks);
        const uint64_t remainder = n % uint64_t(ranks);
        local_rows = base_rows + (uint64_t(rank) < remainder ? 1 : 0);
        const uint64_t begin =
            uint64_t(rank) * base_rows + std::min<uint64_t>(uint64_t(rank), remainder);
        const double generate_start = MPI_Wtime();
        materialise_indices(source, source_rows, options.seed, begin, local_rows, local_x);
        generation_s = MPI_Wtime() - generate_start;
        if (rank == 0) std::vector<double>().swap(source);
    }

    const double distribution_elapsed = MPI_Wtime() - distribution_start;
    double distribution_s = 0.0;
    check_mpi(MPI_Reduce(&distribution_elapsed, &distribution_s, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD),
              "distribution timing reduction");
    double distributed[2] = {source_bcast_s, generation_s};
    double distributed_max[2] = {0.0, 0.0};
    check_mpi(MPI_Reduce(distributed, distributed_max, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD),
              "generation timing reduction");

    // Rank 0 no longer needs the replicated matrix after distribution.
    if (!weak && rank == 0) std::vector<double>().swap(source);

    omp_set_dynamic(0);
    int threads = omp_get_max_threads(); // OMP_NUM_THREADS requests the team size.
    // Record the actual team size rather than only the requested maximum; this
    // also keeps the reported worker count correct under OMP_THREAD_LIMIT.
#pragma omp parallel
    {
#pragma omp single
        threads = omp_get_num_threads();
    }
    omp_set_num_threads(threads);
    int min_threads = 0, max_threads = 0, total_workers = 0;
    check_mpi(MPI_Allreduce(&threads, &min_threads, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD),
              "minimum thread count all-reduce");
    check_mpi(MPI_Allreduce(&threads, &max_threads, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD),
              "maximum thread count all-reduce");
    check_mpi(MPI_Allreduce(&threads, &total_workers, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD),
              "total worker all-reduce");
    if (min_threads != max_threads)
        throw std::runtime_error("OpenMP thread counts differ between MPI ranks");

    std::vector<int> labels(size_t(local_rows), -1);
    std::vector<double> local_sums(size_t(threads) * size_t(k) * D);
    std::vector<uint64_t> local_counts(size_t(threads) * size_t(k));
    std::vector<uint64_t> local_changed(static_cast<size_t>(threads));
    std::vector<double> rank_sums(size_t(k) * D);
    std::vector<uint64_t> rank_counts(static_cast<size_t>(k));
    std::vector<double> global_sums(size_t(k) * D);
    std::vector<uint64_t> count_stats(size_t(k) + 1);
    std::vector<uint64_t> global_count_stats(size_t(k) + 1);

    int iterations = 0;
    bool converged = false;
    const double fit_start = MPI_Wtime();
    double local_compute_s = 0.0;
    double thread_reduction_s = 0.0;
    double collective_s = 0.0;
    double centroid_update_s = 0.0;
    uint64_t reduction_calls = 0;

    for (int iter = 0; iter < options.max_iter; ++iter) {
        std::fill(rank_sums.begin(), rank_sums.end(), 0.0);
        std::fill(rank_counts.begin(), rank_counts.end(), 0);
        std::fill(local_sums.begin(), local_sums.end(), 0.0);
        std::fill(local_counts.begin(), local_counts.end(), 0);
        std::fill(local_changed.begin(), local_changed.end(), 0);

        const double compute_start = MPI_Wtime();
        // Centroids are read-only here, and each local observation has one
        // writer. Thread-local totals avoid atomics and false sharing of the
        // rank-level sum/count arrays.
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            double* thread_sums = &local_sums[size_t(tid) * size_t(k) * D];
            uint64_t* thread_counts = &local_counts[size_t(tid) * size_t(k)];
            uint64_t thread_changed = 0;
#pragma omp for schedule(static)
            for (int64_t i = 0; i < int64_t(local_rows); ++i) {
                const double* point = &local_x[size_t(i) * D];
                int best_id = 0;
                double best = distance2(point, centroids.data());
                for (int cluster = 1; cluster < k; ++cluster) {
                    const double d = distance2(point, &centroids[size_t(cluster) * D]);
                    if (d < best) { best = d; best_id = cluster; }
                }
                thread_changed += labels[size_t(i)] != best_id;
                labels[size_t(i)] = best_id;
                ++thread_counts[best_id];
                for (int j = 0; j < D; ++j)
                    thread_sums[size_t(best_id) * D + j] += point[j];
            }
            local_changed[size_t(tid)] = thread_changed;
        }
        local_compute_s += MPI_Wtime() - compute_start;

        // The implicit end-of-parallel barrier makes all thread-local totals
        // visible. The main thread combines them in fixed thread-ID order.
        const double thread_reduce_start = MPI_Wtime();
        uint64_t local_changed_total = 0;
        for (int tid = 0; tid < threads; ++tid) {
            local_changed_total += local_changed[size_t(tid)];
            for (int cluster = 0; cluster < k; ++cluster) {
                rank_counts[size_t(cluster)] +=
                    local_counts[size_t(tid) * size_t(k) + size_t(cluster)];
                for (int j = 0; j < D; ++j)
                    rank_sums[size_t(cluster) * D + j] +=
                        local_sums[(size_t(tid) * size_t(k) + size_t(cluster)) * D + size_t(j)];
            }
        }
        thread_reduction_s += MPI_Wtime() - thread_reduce_start;

        std::copy(rank_counts.begin(), rank_counts.end(), count_stats.begin());
        count_stats[size_t(k)] = local_changed_total;
        const double reduction_start = MPI_Wtime();
        check_mpi(MPI_Allreduce(rank_sums.data(), global_sums.data(), k * D, MPI_DOUBLE, MPI_SUM,
                                MPI_COMM_WORLD),
                  "cluster sum all-reduce");
        check_mpi(MPI_Allreduce(count_stats.data(), global_count_stats.data(), k + 1, MPI_UINT64_T,
                                MPI_SUM, MPI_COMM_WORLD),
                  "count and changed-label all-reduce");
        collective_s += MPI_Wtime() - reduction_start;
        reduction_calls += 2;

        const uint64_t global_changed = global_count_stats[size_t(k)];
        const double update_start = MPI_Wtime();
        update_centroids(centroids, global_sums, global_count_stats, k);
        centroid_update_s += MPI_Wtime() - update_start;

        iterations = iter + 1;
        converged = global_changed == 0;
        if (converged && !options.fixed_iterations) break;
    }

    const double fit_elapsed = MPI_Wtime() - fit_start;
    double fit_stats[3] = {fit_elapsed, local_compute_s, collective_s};
    double fit_stats_max[3] = {0.0, 0.0, 0.0};
    check_mpi(MPI_Reduce(fit_stats, fit_stats_max, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD),
              "fit timing reduction");
    double fit_min_s = 0.0;
    check_mpi(MPI_Reduce(&fit_elapsed, &fit_min_s, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD),
              "fit minimum timing reduction");
    double update_max_s = 0.0;
    check_mpi(MPI_Reduce(&centroid_update_s, &update_max_s, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD),
              "centroid update timing reduction");
    double thread_reduction_max_s = 0.0;
    check_mpi(MPI_Reduce(&thread_reduction_s, &thread_reduction_max_s, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD),
              "thread reduction timing reduction");

    const double validation_start = MPI_Wtime();
    const double local_sse = local_inertia(local_x, centroids, k);
    double inertia = 0.0;
    check_mpi(MPI_Reduce(&local_sse, &inertia, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD),
              "inertia reduction");

    // Gathering labels once after fitting preserves the original observation
    // order for an exact serial-compatible hash without communicating labels
    // on every iteration.
    std::vector<int> all_labels;
    if (rank == 0) all_labels.assign(size_t(n), -1);
    std::vector<int> gather_counts(ranks), gather_displacements(ranks);
    {
        const uint64_t base_rows = n / uint64_t(ranks);
        const uint64_t remainder = n % uint64_t(ranks);
        int displacement = 0;
        for (int r = 0; r < ranks; ++r) {
            const int rows = int(base_rows + (uint64_t(r) < remainder ? 1 : 0));
            gather_counts[r] = rows;
            gather_displacements[r] = displacement;
            displacement += rows;
        }
    }
    check_mpi(MPI_Gatherv(labels.data(), int(local_rows), MPI_INT,
                          rank == 0 ? all_labels.data() : nullptr,
                          rank == 0 ? gather_counts.data() : nullptr,
                          rank == 0 ? gather_displacements.data() : nullptr, MPI_INT, 0,
                          MPI_COMM_WORLD),
              "final label gather");
    const double validation_elapsed = MPI_Wtime() - validation_start;
    double validation_s = 0.0;
    check_mpi(MPI_Reduce(&validation_elapsed, &validation_s, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD),
              "validation timing reduction");
    const double total_elapsed = MPI_Wtime() - program_start;
    double total_s = 0.0;
    check_mpi(MPI_Reduce(&total_elapsed, &total_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD),
              "total timing reduction");

    if (rank == 0) {
        if (!options.centroid_path.empty()) write_centroids(options.centroid_path, centroids, k);

        const double fit_s = fit_stats_max[0];
        const double collective_max_s = fit_stats_max[2];
        const double compute_max_s = fit_stats_max[1];
        const std::string stopping_mode =
            options.fixed_iterations ? "fixed_iterations" : "convergence";
        print_dataset_line(n, k, options.max_iter, options.seed);
        print_iteration_line(iterations, inertia);
        print_timing_line(load_s, preprocess_s, init_s, fit_s, validation_s, total_s);
        std::cout << "implementation=hybrid ranks=" << ranks << " threads=" << threads
                  << " total_workers=" << total_workers
                  << " stopping_mode=" << stopping_mode
                  << " converged=" << bool_str(converged) << '\n';
        std::cout << std::fixed << std::setprecision(6)
                  << "distribution_s=" << distribution_s
                  << " mpi_reduction_s=" << collective_max_s
                  << " local_compute_s=" << compute_max_s
                  << " thread_reduction_s=" << thread_reduction_max_s
                  << " centroid_update_s=" << update_max_s
                  << " fit_min_s=" << fit_min_s
                  << " fit_imbalance_s=" << (fit_s - fit_min_s)
                  << " obs_per_rank=" << (n / uint64_t(ranks))
                  << " weak=" << (weak ? 1 : 0) << '\n'
                  << "source_rows=" << source_rows
                  << " source_bcast_s=" << distributed_max[0]
                  << " generation_s=" << distributed_max[1]
                  << " reduction_calls=" << reduction_calls << '\n';
        print_cluster_sizes(global_count_stats, k);
        std::cout << "label_hash=" << label_hash(all_labels) << '\n';
        std::cout << "csv_header=implementation,ranks,threads,total_workers,rows,features,k,"
                     "max_iter,iterations,fit_s,sse,converged,stopping_mode,distribution_s,"
                     "mpi_reduction_s,local_compute_s,thread_reduction_s,centroid_update_s,"
                     "fit_min_s,fit_imbalance_s,obs_per_rank,weak\n"
                  << "csv=hybrid," << ranks << ',' << threads << ',' << total_workers << ','
                  << n << ',' << D << ',' << k << ',' << options.max_iter << ',' << iterations
                  << ',' << fit_s << ',' << inertia << ',' << bool_str(converged) << ','
                  << stopping_mode << ',' << distribution_s << ',' << collective_max_s << ','
                  << compute_max_s << ',' << thread_reduction_max_s << ',' << update_max_s << ','
                  << fit_min_s << ',' << (fit_s - fit_min_s) << ','
                  << (n / uint64_t(ranks)) << ',' << (weak ? 1 : 0) << '\n';
    }
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    int init_code = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (init_code != MPI_SUCCESS) {
        std::cerr << "Error: MPI_Init_thread failed\n";
        return 1;
    }
    if (provided < MPI_THREAD_FUNNELED) {
        std::cerr << "Error: MPI_THREAD_FUNNELED support is required\n";
        MPI_Finalize();
        return 1;
    }

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    try {
        int queried = MPI_THREAD_SINGLE;
        check_mpi(MPI_Query_thread(&queried), "thread support query");
        if (queried < MPI_THREAD_FUNNELED)
            throw std::runtime_error("MPI thread support fell below MPI_THREAD_FUNNELED");
        const Options options = parse_options(argc, argv, "kmeans_hybrid");
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
