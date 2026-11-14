// E7 - CUDA K-means assignment kernel (NUS Vanda, NVIDIA A40 / sm_86).
//
// Design mirrors the validated Apple Metal experiment:
//   Stage A: the observation matrix is re-uploaded every iteration; the GPU only
//            assigns labels and the CPU accumulates and updates centroids.
//   Stage B: the observation matrix stays resident on the device; only the K x D
//            centroids are re-sent per iteration.
// Only the assignment kernel runs on the GPU. Load, preprocessing, k-means++,
// accumulation and the centroid update reuse kmeans_common.hpp unchanged, so the
// result is anchored to the double-precision CPU reference rather than to a
// re-derived implementation.
//
// Arithmetic is single precision on the device (as in Metal) while accumulation
// and centroids stay double on the host; the comparison against the CPU
// reference is therefore tolerance based, never bitwise.
//
// K=7 x D=54 floats = 1512 bytes, so the whole centroid matrix lives in
// __constant__ memory: every thread reads it through the constant cache.

#include "kmeans_common.hpp"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

constexpr int MAX_K = 32;
__constant__ float d_centroids[MAX_K * D];

__global__ void assign_labels_kernel(const float* __restrict__ x,
                                    int* __restrict__ labels,
                                    int n, int k) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* p = x + size_t(i) * D;
    float best = 1e30f;
    int best_id = 0;
    for (int c = 0; c < k; ++c) {
        const float* q = d_centroids + size_t(c) * D;
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < D; ++j) {
            const float diff = p[j] - q[j];
            sum = fmaf(diff, diff, sum);
        }
        if (sum < best) { best = sum; best_id = c; }   // strict < keeps lowest id
    }
    labels[i] = best_id;
}

static void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA failure in ") + what + ": " +
                                 cudaGetErrorString(err));
    }
}

static double host_now() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    try {
        // Device-specific flags are consumed here; the shared parser gets the rest.
        std::string mode = "full";
        std::vector<std::string> forwarded;
        forwarded.push_back(argv[0]);
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--mode" && i + 1 < argc) mode = argv[++i];
            else forwarded.push_back(a);
        }
        std::vector<char*> fargv;
        for (std::string& s : forwarded) fargv.push_back(const_cast<char*>(s.c_str()));
        const Options opt = parse_options(int(fargv.size()), fargv.data(), "kmeans_cuda");
        const int k = opt.k;
        if (k > MAX_K) throw std::runtime_error("K exceeds the compiled maximum (32)");

        const double t0 = host_now();
        std::vector<double> x = read_data(opt.data_path, opt.rows);
        const double t1 = host_now();
        const uint64_t n = x.size() / D;
        if (uint64_t(k) > n) throw std::runtime_error("K exceeds number of rows");
        scale_continuous(x);
        const double t2 = host_now();
        std::vector<double> centroids = opt.init_centroid_path.empty()
                                            ? initialize(x, k, opt.seed)
                                            : read_centroids(opt.init_centroid_path, k);
        const double t3 = host_now();
        const std::vector<double> initial_centroids = centroids;

        int device_count = 0;
        check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
        if (device_count == 0) throw std::runtime_error("No CUDA device visible on this node");
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
        char device_name[128];
        std::snprintf(device_name, sizeof(device_name), "%s", prop.name);
        for (char* p = device_name; *p; ++p) if (*p == ' ') *p = '_';

        float* d_x = nullptr;
        int* d_labels = nullptr;
        check(cudaMalloc(&d_x, size_t(n) * D * sizeof(float)), "cudaMalloc X");
        check(cudaMalloc(&d_labels, size_t(n) * sizeof(int)), "cudaMalloc labels");

        const double up_start = host_now();
        {
            std::vector<float> xf(size_t(n) * D);
            for (size_t i = 0; i < xf.size(); ++i) xf[i] = float(x[i]);
            check(cudaMemcpy(d_x, xf.data(), xf.size() * sizeof(float), cudaMemcpyHostToDevice),
                  "upload X");
        }
        const double upload_once_s = host_now() - up_start;

        std::vector<int> labels(size_t(n), -1);
        std::vector<int> host_labels(size_t(n), -1);

        auto pack_centroids = [&](const std::vector<double>& c) {
            std::vector<float> cf(size_t(k) * D);
            for (size_t i = 0; i < cf.size(); ++i) cf[i] = float(c[i]);
            check(cudaMemcpyToSymbol(d_centroids, cf.data(), cf.size() * sizeof(float)),
                  "centroid pack");
        };
        const int block = 256;
        const int grid = int((n + block - 1) / block);

        // Isolated kernel timing on the initial centroids: CUDA events, and the
        // host wall time around launch+sync (which includes dispatch overhead).
        pack_centroids(initial_centroids);
        double gpu_kernel_s = 0.0, gpu_wall_s = 0.0;
        for (int rep = 0; rep < 3; ++rep) {
            cudaEvent_t ev0, ev1;
            check(cudaEventCreate(&ev0), "event");
            check(cudaEventCreate(&ev1), "event");
            check(cudaEventRecord(ev0), "record");
            assign_labels_kernel<<<grid, block>>>(d_x, d_labels, int(n), k);
            check(cudaEventRecord(ev1), "record");
            check(cudaEventSynchronize(ev1), "sync");
            float ms = 0.f;
            check(cudaEventElapsedTime(&ms, ev0, ev1), "elapsed");
            gpu_kernel_s = ms / 1000.0;
            const double w0 = host_now();
            assign_labels_kernel<<<grid, block>>>(d_x, d_labels, int(n), k);
            check(cudaDeviceSynchronize(), "sync");
            gpu_wall_s = host_now() - w0;
            cudaEventDestroy(ev0);
            cudaEventDestroy(ev1);
        }

        // Launch floor: the same kernel over one observation.
        double dispatch_floor_s = 0.0;
        pack_centroids(initial_centroids);
        for (int rep = 0; rep < 20; ++rep) {
            const double w0 = host_now();
            assign_labels_kernel<<<1, 1>>>(d_x, d_labels, 1, k);
            check(cudaDeviceSynchronize(), "sync");
            dispatch_floor_s += (host_now() - w0) / 20.0;
        }

        // CPU reference assignment pass of identical work.
        double cpu_assign_s = 0.0;
        std::vector<int> cpu_labels;
        for (int rep = 0; rep < 3; ++rep) {
            const double w0 = host_now();
            cpu_labels.assign(size_t(n), 0);
            for (uint64_t i = 0; i < n; ++i) {
                int best_id = 0;
                double best = distance2(&x[size_t(i) * D], initial_centroids.data());
                for (int c = 1; c < k; ++c) {
                    const double d = distance2(&x[size_t(i) * D], &initial_centroids[size_t(c) * D]);
                    if (d < best) { best = d; best_id = c; }
                }
                cpu_labels[size_t(i)] = best_id;
            }
            cpu_assign_s = host_now() - w0;
        }

        // ---- fit -----------------------------------------------------------
        const bool stage_a = (mode == "assign");
        std::vector<double> sums(size_t(k) * D);
        std::vector<uint64_t> counts(static_cast<size_t>(k));
        std::vector<int> previous(size_t(n), -1);
        int iterations = 0;
        bool converged = false;
        double upload_s = 0.0, readback_s = 0.0, accumulate_s = 0.0, pack_s = 0.0;
        const double fit_start = host_now();
        for (int iter = 0; iter < opt.max_iter; ++iter) {
            std::fill(sums.begin(), sums.end(), 0.0);
            std::fill(counts.begin(), counts.end(), 0);

            if (stage_a) {
                const double w0 = host_now();
                std::vector<float> xf(size_t(n) * D);
                for (size_t i = 0; i < xf.size(); ++i) xf[i] = float(x[i]);
                check(cudaMemcpy(d_x, xf.data(), xf.size() * sizeof(float), cudaMemcpyHostToDevice),
                      "re-upload X");
                upload_s += host_now() - w0;
            }
            const double p0 = host_now();
            pack_centroids(centroids);
            pack_s += host_now() - p0;

            assign_labels_kernel<<<grid, block>>>(d_x, d_labels, int(n), k);
            check(cudaDeviceSynchronize(), "kernel sync");

            const double r0 = host_now();
            check(cudaMemcpy(host_labels.data(), d_labels, size_t(n) * sizeof(int),
                             cudaMemcpyDeviceToHost), "readback labels");
            readback_s += host_now() - r0;

            const double a0 = host_now();
            for (uint64_t i = 0; i < n; ++i) {
                const int lab = host_labels[size_t(i)];
                ++counts[size_t(lab)];
                for (int j = 0; j < D; ++j) sums[size_t(lab) * D + j] += x[size_t(i) * D + j];
            }
            accumulate_s += host_now() - a0;

            uint64_t changed = 0;
            for (uint64_t i = 0; i < n; ++i) changed += previous[size_t(i)] != host_labels[size_t(i)];
            previous = host_labels;

            update_centroids(centroids, sums, counts, k);
            iterations = iter + 1;
            if (changed == 0) {
                converged = true;
                if (!opt.fixed_iterations) break;
            }
        }
        const double fit_s = host_now() - fit_start;

        const double val_start = host_now();
        const double inertia = local_inertia(x, centroids, k);
        double label_agreement = 0.0;
        std::vector<double> reference = initial_centroids;
        {
            std::vector<int> ref_labels(size_t(n), -1);
            std::vector<double> ref_sums(size_t(k) * D);
            std::vector<uint64_t> ref_counts(static_cast<size_t>(k));
            for (int iter = 0; iter < iterations; ++iter) {
                std::fill(ref_sums.begin(), ref_sums.end(), 0.0);
                std::fill(ref_counts.begin(), ref_counts.end(), 0);
                uint64_t changed = 0;
                for (uint64_t i = 0; i < n; ++i) {
                    int best_id = 0;
                    double best = distance2(&x[size_t(i) * D], reference.data());
                    for (int c = 1; c < k; ++c) {
                        const double d = distance2(&x[size_t(i) * D], &reference[size_t(c) * D]);
                        if (d < best) { best = d; best_id = c; }
                    }
                    changed += ref_labels[size_t(i)] != best_id;
                    ref_labels[size_t(i)] = best_id;
                    ++ref_counts[size_t(best_id)];
                    for (int j = 0; j < D; ++j)
                        ref_sums[size_t(best_id) * D + j] += x[size_t(i) * D + j];
                }
                update_centroids(reference, ref_sums, ref_counts, k);
                if (changed == 0 && !opt.fixed_iterations) break;
            }
            uint64_t agree = 0;
            for (uint64_t i = 0; i < n; ++i) agree += ref_labels[size_t(i)] == host_labels[size_t(i)];
            label_agreement = double(agree) / double(n);
        }
        const double reference_inertia = local_inertia(x, reference, k);
        double centroid_max_diff = 0.0;
        for (size_t i = 0; i < size_t(k) * D; ++i)
            centroid_max_diff = std::max(centroid_max_diff, std::fabs(reference[i] - centroids[i]));
        const double validation_s = host_now() - val_start;
        const double total_s = host_now() - t0;
        const double inertia_rel = reference_inertia > 0
                                       ? std::fabs(inertia - reference_inertia) / reference_inertia
                                       : 0.0;

        print_dataset_line(n, k, opt.max_iter, opt.seed);
        print_iteration_line(iterations, inertia);
        print_timing_line(t1 - t0, t2 - t1, t3 - t2, fit_s, validation_s, total_s);
        std::cout << "implementation=cuda device=" << device_name
                  << " cc=" << prop.major << prop.minor
                  << " stage=" << (stage_a ? "A_assign" : "B_full")
                  << " rows=" << n << " features=" << D << " k=" << k
                  << " stopping_mode="
                  << (opt.fixed_iterations ? "fixed_iterations" : "convergence")
                  << " converged=" << bool_str(converged) << '\n';
        std::cout << std::fixed << std::setprecision(9)
                  << "cpu_assign_s=" << cpu_assign_s << " gpu_assign_s=" << gpu_wall_s
                  << " gpu_kernel_s=" << gpu_kernel_s << " dispatch_floor_s=" << dispatch_floor_s
                  << " upload_s=" << upload_once_s + upload_s
                  << " readback_s=" << readback_s << " accumulate_s=" << accumulate_s
                  << " centroid_pack_s=" << pack_s << '\n'
                  << "label_agreement=" << label_agreement
                  << " centroid_max_abs_diff=" << centroid_max_diff
                  << " reference_inertia=" << reference_inertia
                  << " inertia_rel_diff=" << inertia_rel << '\n';
        print_cluster_sizes(counts, k);
        std::cout << "label_hash=" << label_hash(host_labels) << '\n';
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "csv_header=implementation,stage,device,rows,features,k,max_iter,iterations,"
                     "fit_s,sse,converged,stopping_mode,cpu_assign_s,gpu_assign_s,gpu_kernel_s,"
                     "dispatch_floor_s,upload_s,readback_s,accumulate_s,centroid_pack_s,"
                     "label_agreement,centroid_max_abs_diff,inertia_rel_diff\n"
                  << "csv=cuda," << (stage_a ? "A_assign" : "B_full") << ',' << device_name << ','
                  << n << ',' << D << ',' << k << ',' << opt.max_iter << ',' << iterations << ','
                  << fit_s << ',' << inertia << ',' << bool_str(converged) << ','
                  << (opt.fixed_iterations ? "fixed_iterations" : "convergence") << ','
                  << cpu_assign_s << ',' << gpu_wall_s << ',' << gpu_kernel_s << ','
                  << dispatch_floor_s << ',' << upload_once_s + upload_s << ',' << readback_s
                  << ',' << accumulate_s << ',' << pack_s << ',' << label_agreement << ','
                  << centroid_max_diff << ',' << inertia_rel << '\n';

        cudaFree(d_x);
        cudaFree(d_labels);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
