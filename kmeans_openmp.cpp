#include "kmeans_common.hpp"

#include <omp.h>

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv, "kmeans_openmp");
        if (options.weak_per_rank > 0)
            throw std::runtime_error("--weak-per-rank is an MPI-only workload mode");
        const int k = options.k;
        const int max_iter = options.max_iter;
        const uint64_t seed = options.seed;

        const auto t0 = Clock::now();
        std::vector<double> x;
        std::vector<double> c;
        uint64_t n = 0;
        double load_s = 0.0, preprocess_s = 0.0, init_s = 0.0, generation_s = 0.0;
        if (options.resampled_total > 0) {
            // Fixed-total deterministic resampling: build the logical dataset
            // from the preprocessed source rows. The same (seed, global index)
            // mapping gives every rank count exactly the same observation set,
            // and k-means++ on the source gives every rank count the same
            // initial centroids, so only the parallel decomposition changes.
            std::vector<double> source = read_data(options.data_path, options.rows);
            const auto t1 = Clock::now();
            const uint64_t source_rows = source.size() / D;
            if (uint64_t(k) > source_rows)
                throw std::runtime_error("K exceeds number of source rows");
            scale_continuous(source);
            const auto t2 = Clock::now();
            c = options.init_centroid_path.empty()
                    ? initialize(source, k, seed)
                    : read_centroids(options.init_centroid_path, k);
            const auto t3 = Clock::now();
            materialise_indices(source, source_rows, seed, 0, options.resampled_total, x);
            const auto t4 = Clock::now();
            n = options.resampled_total;
            load_s = seconds(t0, t1);
            preprocess_s = seconds(t1, t2);
            init_s = seconds(t2, t3);
            generation_s = seconds(t3, t4);
        } else {
            x = read_data(options.data_path, options.rows);
            const auto t1 = Clock::now();
            n = x.size() / D;
            if (uint64_t(k) > n) throw std::runtime_error("K exceeds number of rows");
            scale_continuous(x);
            const auto t2 = Clock::now();
            c = options.init_centroid_path.empty()
                    ? initialize(x, k, seed)
                    : read_centroids(options.init_centroid_path, k);
            const auto t3 = Clock::now();
            load_s = seconds(t0, t1);
            preprocess_s = seconds(t1, t2);
            init_s = seconds(t2, t3);
        }
        const auto fit_begin = Clock::now();

        omp_set_dynamic(0);
        std::vector<int> labels(n, -1);
        std::vector<double> sums(size_t(k) * D);
        std::vector<uint64_t> counts(k);
        const int threads = omp_get_max_threads(); // Controlled by OMP_NUM_THREADS.
        std::vector<double> local_sums(size_t(threads) * size_t(k) * D);
        std::vector<uint64_t> local_counts(size_t(threads) * size_t(k));
        std::vector<size_t> local_changed(static_cast<size_t>(threads));
        int iterations = 0;
        bool converged = false;
        double assign_accumulate_s = 0.0;
        double thread_reduction_s = 0.0;
        double centroid_update_s = 0.0;
        for (int iter = 0; iter < max_iter; ++iter) {
            std::fill(sums.begin(), sums.end(), 0.0);
            std::fill(counts.begin(), counts.end(), 0);
            std::fill(local_sums.begin(), local_sums.end(), 0.0);
            std::fill(local_counts.begin(), local_counts.end(), 0);
            std::fill(local_changed.begin(), local_changed.end(), 0);
            size_t changed = 0;
            const auto assign_start = Clock::now();
            // Centroids are read-only here; each point's label has one writer.
            // Cluster totals are private to each thread to avoid races/atomics.
#pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                double* thread_sums = &local_sums[size_t(tid) * size_t(k) * D];
                uint64_t* thread_counts = &local_counts[size_t(tid) * size_t(k)];
                size_t thread_changed = 0;
#pragma omp for schedule(static)
                for (int64_t i = 0; i < int64_t(n); ++i) {
                    const double* point = &x[size_t(i) * D];
                    int best_id = 0;
                    double best = distance2(point, c.data());
                    for (int cluster = 1; cluster < k; ++cluster) {
                        const double d = distance2(point, &c[size_t(cluster) * D]);
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
            assign_accumulate_s += seconds(assign_start, Clock::now());
            const auto reduce_start = Clock::now();
            // Implicit barrier at the end of the parallel region. Fixed thread
            // order keeps the reduction deterministic for a fixed thread count.
            for (int tid = 0; tid < threads; ++tid) {
                changed += local_changed[size_t(tid)];
                for (int cluster = 0; cluster < k; ++cluster) {
                    counts[size_t(cluster)] += local_counts[size_t(tid) * size_t(k) + size_t(cluster)];
                    for (int j = 0; j < D; ++j)
                        sums[size_t(cluster) * D + j] +=
                            local_sums[(size_t(tid) * size_t(k) + size_t(cluster)) * D + size_t(j)];
                }
            }
            thread_reduction_s += seconds(reduce_start, Clock::now());
            const auto update_start = Clock::now();
            update_centroids(c, sums, counts, k);
            centroid_update_s += seconds(update_start, Clock::now());
            iterations = iter + 1;
            if (changed == 0) {
                converged = true;
                if (!options.fixed_iterations) break;
            }
        }
        const auto fit_end = Clock::now();
        const double inertia = final_inertia(x, c, k);
        const auto t5 = Clock::now();
        if (!options.centroid_path.empty()) write_centroids(options.centroid_path, c, k);

        const double fit_s = seconds(fit_begin, fit_end);
        print_dataset_line(n, k, max_iter, seed);
        print_iteration_line(iterations, inertia);
        print_timing_line(load_s, preprocess_s, init_s, fit_s,
                          seconds(fit_end, t5), seconds(t0, t5));
        std::cout << "implementation=openmp ranks=1 threads=" << threads
                  << " total_workers=" << threads
                  << " stopping_mode=" << (options.fixed_iterations ? "fixed_iterations" : "convergence")
                  << " converged=" << bool_str(converged) << '\n';
        std::cout << "distribution_s=" << generation_s << " mpi_reduction_s=0 local_compute_s=" << assign_accumulate_s
                  << " thread_reduction_s=" << thread_reduction_s
                  << " centroid_update_s=" << centroid_update_s
                  << " fit_min_s=" << fit_s << " fit_imbalance_s=0"
                  << " obs_per_rank=" << n << " weak=0"
                  << " resampled_total=" << options.resampled_total << '\n';
        print_cluster_sizes(counts, k);
        std::cout << "label_hash=" << label_hash(labels) << '\n';
        std::cout << "csv_header=implementation,ranks,threads,total_workers,rows,features,k,"
                     "max_iter,iterations,fit_s,sse,converged,stopping_mode,distribution_s,"
                     "mpi_reduction_s,local_compute_s,thread_reduction_s,centroid_update_s,"
                     "fit_min_s,fit_imbalance_s,obs_per_rank,weak\n"
                  << "csv=openmp,1," << threads << ',' << threads << ',' << n << ',' << D << ','
                  << k << ',' << max_iter << ',' << iterations << ',' << fit_s << ','
                  << inertia << ',' << bool_str(converged) << ','
                  << (options.fixed_iterations ? "fixed_iterations" : "convergence")
                  << "," << generation_s << ",0," << assign_accumulate_s << ',' << thread_reduction_s << ','
                  << centroid_update_s << ',' << fit_s << ",0," << n << ",0\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
