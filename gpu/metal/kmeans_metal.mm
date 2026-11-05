// Apple Metal K-means experiment (E6). Local-only: no CUDA, no HPC scripts.
//
// Stage A ("assign"): naive per-iteration transfers. Both the observation
//   matrix and the centroids are re-uploaded every iteration and labels are
//   copied back explicitly, so the cost of avoidable traffic is visible.
// Stage B ("full"): the observation matrix stays resident on the GPU, only the
//   K x D centroids are re-uploaded per iteration, and labels are read through
//   shared (unified) memory without an explicit copy.
//
// Both stages run the same cold-start double-precision preprocessing and
// k-means++ initialization as the CPU programs; only the assignment kernel and
// the centroid storage are single precision.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "kmeans_common.hpp"
#include "kernels_source.h"

namespace {

double wall_now() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

struct GpuTiming {
    double host_s = 0.0;    // encode + commit + wait on the host
    double kernel_s = 0.0;  // GPUStartTime..GPUEndTime reported by Metal
    double upload_s = 0.0;
    double readback_s = 0.0;
    double accumulate_s = 0.0;
    double pack_s = 0.0;
    uint64_t dispatches = 0;
};

std::string sanitise(const std::string& value) {
    std::string out;
    for (char ch : value) out.push_back(ch == ' ' ? '_' : ch);
    return out;
}

int run(int argc, char** argv) {
    // Metal-specific flags are stripped before the shared parser runs.
    std::string mode = "full";
    bool validate = false;
    bool bench_overhead = false;
    std::vector<std::string> forwarded;
    forwarded.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (argument == "--validate") {
            validate = true;
        } else if (argument == "--bench-overhead") {
            bench_overhead = true;
        } else {
            forwarded.push_back(argument);
        }
    }
    if (mode != "assign" && mode != "full") {
        throw std::runtime_error("--mode must be 'assign' (Stage A) or 'full' (Stage B)");
    }
    std::vector<char*> forwarded_argv;
    forwarded_argv.reserve(forwarded.size());
    for (std::string& value : forwarded) forwarded_argv.push_back(const_cast<char*>(value.c_str()));

    const Options options =
        parse_options(int(forwarded_argv.size()), forwarded_argv.data(), "kmeans_metal");
    const int k = options.k;

    const double t0 = wall_now();
    std::vector<double> x = read_data(options.data_path, options.rows);
    const double t1 = wall_now();
    const uint64_t n = x.size() / D;
    if (uint64_t(k) > n) throw std::runtime_error("K exceeds number of rows");
    scale_continuous(x);
    const double t2 = wall_now();
    std::vector<double> centroids = options.init_centroid_path.empty()
                                        ? initialize(x, k, options.seed)
                                        : read_centroids(options.init_centroid_path, k);
    const double t3 = wall_now();
    const std::vector<double> initial_centroids = centroids;

    // ---- Metal setup ------------------------------------------------------
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil)
        throw std::runtime_error(
            "Metal unavailable: no system default Metal device on this machine");
    NSError* error = nil;
    // MTLCompileOptions defaults to fastMathEnabled = NO, i.e. strict float math.
    MTLCompileOptions* compile_options = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:kKmeansMetalSource]
                                                  options:compile_options
                                                    error:&error];
    if (library == nil) {
        throw std::runtime_error(std::string("Metal shader compilation failed: ") +
                                 [[error localizedDescription] UTF8String]);
    }
    id<MTLFunction> function = [library newFunctionWithName:@"assign_labels"];
    if (function == nil) throw std::runtime_error("Metal function assign_labels is missing");
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                error:&error];
    if (pipeline == nil) {
        throw std::runtime_error(std::string("Metal pipeline creation failed: ") +
                                 [[error localizedDescription] UTF8String]);
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    const bool unified = [device hasUnifiedMemory];

    const size_t x_elements = size_t(n) * D;
    const size_t c_elements = size_t(k) * D;
    id<MTLBuffer> x_buffer = [device newBufferWithLength:x_elements * sizeof(float)
                                               options:MTLResourceStorageModeShared];
    id<MTLBuffer> centroid_buffer = [device newBufferWithLength:c_elements * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> label_buffer = [device newBufferWithLength:size_t(n) * sizeof(int)
                                                     options:MTLResourceStorageModeShared];
    if (x_buffer == nil || centroid_buffer == nil || label_buffer == nil)
        throw std::runtime_error("Metal buffer allocation failed");

    const NSUInteger threadgroup = std::min<NSUInteger>([pipeline maxTotalThreadsPerThreadgroup], 256);
    const MTLSize threads_per_group = MTLSizeMake(threadgroup, 1, 1);

    auto encode_assign = [&](uint64_t count) -> double {
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:x_buffer offset:0 atIndex:0];
        [encoder setBuffer:centroid_buffer offset:0 atIndex:1];
        [encoder setBuffer:label_buffer offset:0 atIndex:2];
        uint32_t count32 = uint32_t(count), k32 = uint32_t(k), d32 = uint32_t(D);
        [encoder setBytes:&count32 length:sizeof(count32) atIndex:3];
        [encoder setBytes:&k32 length:sizeof(k32) atIndex:4];
        [encoder setBytes:&d32 length:sizeof(d32) atIndex:5];
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:threads_per_group];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        return command_buffer.GPUEndTime - command_buffer.GPUStartTime;
    };

    auto pack_centroids = [&](const std::vector<double>& source, GpuTiming& timing) {
        const double start = wall_now();
        float* destination = static_cast<float*>([centroid_buffer contents]);
        for (size_t i = 0; i < c_elements; ++i) destination[i] = float(source[i]);
        timing.pack_s += wall_now() - start;
    };

    GpuTiming timing;
    {
        const double start = wall_now();
        float* destination = static_cast<float*>([x_buffer contents]);
        for (size_t i = 0; i < x_elements; ++i) destination[i] = float(x[i]);
        timing.upload_s += wall_now() - start;
    }

    // ---- isolated CPU reference assignment --------------------------------
    // Same work as the GPU kernel: label everything once against the initial
    // centroids. Reported separately from any end-to-end fitting time.
    auto cpu_assign = [&](const std::vector<double>& centres) {
        std::vector<int> labels(size_t(n), 0);
        for (uint64_t i = 0; i < n; ++i) {
            int best_id = 0;
            double best = distance2(&x[size_t(i) * D], centres.data());
            for (int cluster = 1; cluster < k; ++cluster) {
                const double d = distance2(&x[size_t(i) * D], &centres[size_t(cluster) * D]);
                if (d < best) { best = d; best_id = cluster; }
            }
            labels[size_t(i)] = best_id;
        }
        return labels;
    };
    double cpu_assign_s = 0.0;
    std::vector<int> cpu_labels;
    for (int repeat = 0; repeat < 3; ++repeat) {
        const double start = wall_now();
        cpu_labels = cpu_assign(initial_centroids);
        cpu_assign_s = wall_now() - start;
    }

    // ---- isolated GPU assignment benchmark -------------------------------
    double gpu_assign_s = 0.0;
    double gpu_kernel_s = 0.0;
    {
        pack_centroids(initial_centroids, timing);
        for (int repeat = 0; repeat < 3; ++repeat) {
            const double start = wall_now();
            const double kernel = encode_assign(n);
            gpu_assign_s = wall_now() - start;
            gpu_kernel_s = kernel;
        }
    }

    double dispatch_floor_s = 0.0;
    if (bench_overhead) {
        const int repeats = 20;
        for (int repeat = 0; repeat < repeats; ++repeat) {
            const double start = wall_now();
            encode_assign(1);
            dispatch_floor_s += (wall_now() - start) / double(repeats);
        }
    }

    // ---- fit -------------------------------------------------------------
    const bool stage_a = (mode == "assign");
    std::vector<int> labels(size_t(n), -1);
    std::vector<int> previous_labels(size_t(n), -1);
    std::vector<double> sums(size_t(k) * D);
    std::vector<uint64_t> counts(static_cast<size_t>(k));
    int iterations = 0;
    bool converged = false;
    const double fit_start = wall_now();
    for (int iter = 0; iter < options.max_iter; ++iter) {
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);

        if (stage_a) {
            // Stage A: re-upload everything the kernel needs, copy labels back.
            const double upload_start = wall_now();
            float* xr = static_cast<float*>([x_buffer contents]);
            for (size_t i = 0; i < x_elements; ++i) xr[i] = float(x[i]);
            pack_centroids(centroids, timing);
            timing.upload_s += wall_now() - upload_start;
        } else {
            pack_centroids(centroids, timing);
        }

        const double dispatch_start = wall_now();
        timing.kernel_s += encode_assign(n);
        timing.host_s += wall_now() - dispatch_start;
        timing.dispatches += 1;

        uint64_t changed = 0;
        const double readback_start = wall_now();
        if (stage_a) {
            std::memcpy(labels.data(), [label_buffer contents], size_t(n) * sizeof(int));
        } else {
            const int* shared = static_cast<const int*>([label_buffer contents]);
            for (uint64_t i = 0; i < n; ++i) labels[size_t(i)] = shared[i];
        }
        timing.readback_s += wall_now() - readback_start;

        const double accumulate_start = wall_now();
        for (uint64_t i = 0; i < n; ++i) {
            const int label = labels[size_t(i)];
            ++counts[size_t(label)];
            for (int j = 0; j < D; ++j) sums[size_t(label) * D + j] += x[size_t(i) * D + j];
        }
        timing.accumulate_s += wall_now() - accumulate_start;
        for (uint64_t i = 0; i < n; ++i) changed += previous_labels[size_t(i)] != labels[size_t(i)];
        previous_labels = labels;

        update_centroids(centroids, sums, counts, k);
        iterations = iter + 1;
        if (changed == 0) {
            converged = true;
            if (!options.fixed_iterations) break;
        }
    }
    const double fit_s = wall_now() - fit_start;

    const double validation_start = wall_now();
    const double inertia = local_inertia(x, centroids, k);
    // Double-precision reference on the same initialization and iteration count.
    double label_agreement = 0.0;
    std::vector<double> reference = initial_centroids;
    {
        std::vector<int> reference_labels(size_t(n), -1);
        std::vector<double> reference_sums(size_t(k) * D);
        std::vector<uint64_t> reference_counts(static_cast<size_t>(k));
        for (int iter = 0; iter < iterations; ++iter) {
            std::fill(reference_sums.begin(), reference_sums.end(), 0.0);
            std::fill(reference_counts.begin(), reference_counts.end(), 0);
            uint64_t changed = 0;
            for (uint64_t i = 0; i < n; ++i) {
                int best_id = 0;
                double best = distance2(&x[size_t(i) * D], reference.data());
                for (int cluster = 1; cluster < k; ++cluster) {
                    const double d = distance2(&x[size_t(i) * D], &reference[size_t(cluster) * D]);
                    if (d < best) { best = d; best_id = cluster; }
                }
                changed += reference_labels[size_t(i)] != best_id;
                reference_labels[size_t(i)] = best_id;
                ++reference_counts[size_t(best_id)];
                for (int j = 0; j < D; ++j)
                    reference_sums[size_t(best_id) * D + j] += x[size_t(i) * D + j];
            }
            update_centroids(reference, reference_sums, reference_counts, k);
            if (changed == 0 && !options.fixed_iterations) break;
        }
        uint64_t agree = 0;
        for (uint64_t i = 0; i < n; ++i) agree += reference_labels[size_t(i)] == labels[size_t(i)];
        label_agreement = double(agree) / double(n);
    }
    const double reference_inertia = local_inertia(x, reference, k);
    double centroid_max_diff = 0.0;
    for (size_t i = 0; i < size_t(k) * D; ++i)
        centroid_max_diff = std::max(centroid_max_diff, std::fabs(reference[i] - centroids[i]));
    const double validation_s = wall_now() - validation_start;
    const double total_s = wall_now() - t0;

    if (!options.centroid_path.empty()) write_centroids(options.centroid_path, centroids, k);

    if (validate) {
        std::cout << std::fixed << std::setprecision(6)
                  << "validate_label_agreement=" << label_agreement << '\n';
    }
    print_dataset_line(n, k, options.max_iter, options.seed);
    print_iteration_line(iterations, inertia);
    print_timing_line(t1 - t0, t2 - t1, t3 - t2, fit_s, validation_s, total_s);
    std::cout << "implementation=metal device=" << sanitise([[device name] UTF8String])
              << " unified_memory=" << (unified ? 1 : 0)
              << " stage=" << (stage_a ? "A_assign" : "B_full")
              << " rows=" << n << " features=" << D << " k=" << k
              << " stopping_mode="
              << (options.fixed_iterations ? "fixed_iterations" : "convergence")
              << " converged=" << bool_str(converged) << '\n';
    std::cout << std::fixed << std::setprecision(9)
              << "cpu_assign_s=" << cpu_assign_s << " gpu_assign_s=" << gpu_assign_s
              << " gpu_kernel_s=" << gpu_kernel_s << " dispatch_floor_s=" << dispatch_floor_s
              << " dispatches=" << timing.dispatches << '\n'
              << "gpu_kernel_total_s=" << timing.kernel_s
              << " gpu_host_total_s=" << timing.host_s
              << " upload_s=" << timing.upload_s << " readback_s=" << timing.readback_s
              << " accumulate_s=" << timing.accumulate_s << " centroid_pack_s=" << timing.pack_s
              << '\n'
              << "label_agreement=" << label_agreement
              << " centroid_max_abs_diff=" << centroid_max_diff
              << " reference_inertia=" << reference_inertia
              << " inertia_rel_diff="
              << (reference_inertia > 0 ? std::fabs(inertia - reference_inertia) / reference_inertia
                                        : 0.0)
              << '\n';
    print_cluster_sizes(counts, k);
    std::cout << "label_hash=" << label_hash(labels) << '\n';
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "csv_header=implementation,stage,rows,features,k,max_iter,iterations,fit_s,sse,"
                 "converged,stopping_mode,cpu_assign_s,gpu_assign_s,gpu_kernel_s,"
                 "dispatch_floor_s,upload_s,readback_s,accumulate_s,centroid_pack_s,"
                 "label_agreement,centroid_max_abs_diff,inertia_rel_diff\n"
              << "csv=metal," << (stage_a ? "A_assign" : "B_full") << ',' << n << ',' << D << ','
              << k << ',' << options.max_iter << ',' << iterations << ',' << fit_s << ','
              << inertia << ',' << bool_str(converged) << ','
              << (options.fixed_iterations ? "fixed_iterations" : "convergence") << ','
              << cpu_assign_s << ',' << gpu_assign_s << ',' << gpu_kernel_s << ','
              << dispatch_floor_s << ',' << timing.upload_s << ',' << timing.readback_s << ','
              << timing.accumulate_s << ',' << timing.pack_s << ',' << label_agreement << ','
              << centroid_max_diff << ','
              << (reference_inertia > 0 ? std::fabs(inertia - reference_inertia) / reference_inertia
                                        : 0.0)
              << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        try {
            return run(argc, argv);
        } catch (const std::exception& error) {
            std::cerr << "Error: " << error.what() << '\n';
            return 1;
        }
    }
}
