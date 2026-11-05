#include "kmeans_common.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv, "kmeans_serial");
        const int k = options.k;
        const int max_iter = options.max_iter;
        const uint64_t seed = options.seed;

        const auto t0 = Clock::now();
        std::vector<double> x = read_data(options.data_path, options.rows);
        const auto t1 = Clock::now();
        const uint64_t n = x.size() / D;
        if (uint64_t(k) > n) throw std::runtime_error("K exceeds number of rows");
        scale_continuous(x);
        const auto t2 = Clock::now();
        std::vector<double> c = options.init_centroid_path.empty()
                                    ? initialize(x, k, seed)
                                    : read_centroids(options.init_centroid_path, k);
        const auto t3 = Clock::now();

        std::vector<int> labels(n, -1);
        std::vector<double> sums(size_t(k) * D);
        std::vector<uint64_t> counts(k);
        int iterations = 0;
        bool converged = false;
        // E5: the serial kernel fuses assignment and accumulation in one pass,
        // so they are reported as a single phase (assign_accumulate_s).
        double assign_accumulate_s = 0.0;
        double centroid_update_s = 0.0;
        for (int iter = 0; iter < max_iter; ++iter) {
            std::fill(sums.begin(), sums.end(), 0.0);
            std::fill(counts.begin(), counts.end(), 0);
            size_t changed = 0;
            const auto assign_start = Clock::now();
            for (uint64_t i = 0; i < n; ++i) {
                int best_id = 0;
                double best = distance2(&x[size_t(i) * D], c.data());
                for (int cluster = 1; cluster < k; ++cluster) {
                    const double d = distance2(&x[size_t(i) * D], &c[size_t(cluster) * D]);
                    if (d < best) { best = d; best_id = cluster; }
                }
                changed += labels[size_t(i)] != best_id;
                labels[size_t(i)] = best_id;
                ++counts[best_id];
                for (int j = 0; j < D; ++j) sums[size_t(best_id) * D + j] += x[size_t(i) * D + j];
            }
            assign_accumulate_s += seconds(assign_start, Clock::now());
            const auto update_start = Clock::now();
            update_centroids(c, sums, counts, k);
            centroid_update_s += seconds(update_start, Clock::now());
            iterations = iter + 1;
            if (changed == 0) {
                converged = true;
                if (!options.fixed_iterations) break;
            }
        }
        const auto t4 = Clock::now();
        const double inertia = final_inertia(x, c, k);
        const auto t5 = Clock::now();
        if (!options.centroid_path.empty()) write_centroids(options.centroid_path, c, k);

        const double fit_s = seconds(t3, t4);
        print_dataset_line(n, k, max_iter, seed);
        print_iteration_line(iterations, inertia);
        print_timing_line(seconds(t0, t1), seconds(t1, t2), seconds(t2, t3), fit_s,
                          seconds(t4, t5), seconds(t0, t5));
        std::cout << "implementation=serial ranks=1 threads=1 total_workers=1"
                  << " stopping_mode=" << (options.fixed_iterations ? "fixed_iterations" : "convergence")
                  << " converged=" << bool_str(converged) << '\n';
        std::cout << "distribution_s=0 mpi_reduction_s=0 local_compute_s=" << assign_accumulate_s
                  << " thread_reduction_s=0 centroid_update_s=" << centroid_update_s
                  << " fit_min_s=" << fit_s << " fit_imbalance_s=0"
                  << " obs_per_rank=" << n << " weak=0" << '\n';
        print_cluster_sizes(counts, k);
        std::cout << "label_hash=" << label_hash(labels) << '\n';
        std::cout << "csv_header=implementation,ranks,threads,total_workers,rows,features,k,"
                     "max_iter,iterations,fit_s,sse,converged,stopping_mode,distribution_s,"
                     "mpi_reduction_s,local_compute_s,thread_reduction_s,centroid_update_s,"
                     "fit_min_s,fit_imbalance_s,obs_per_rank,weak\n"
                  << "csv=serial,1,1,1," << n << ',' << D << ',' << k << ',' << max_iter << ','
                  << iterations << ',' << fit_s << ',' << inertia << ','
                  << bool_str(converged) << ','
                  << (options.fixed_iterations ? "fixed_iterations" : "convergence")
                  << ",0,0," << assign_accumulate_s << ",0," << centroid_update_s << ','
                  << fit_s << ",0," << n << ",0\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
