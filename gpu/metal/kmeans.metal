#include <metal_stdlib>
using namespace metal;

// One GPU thread per observation. Every thread scans all K centroids and keeps
// the lowest squared-Euclidean distance, matching the CPU tie-breaking rule
// (strict "<" keeps the lowest cluster id on ties).
//
// Arithmetic is single precision: Apple GPUs do not expose IEEE double in
// Metal shaders. The comparison against the double-precision CPU reference is
// therefore tolerance based, never bitwise.
kernel void assign_labels(device const float* x        [[buffer(0)]],
                          device const float* centroids [[buffer(1)]],
                          device int* labels            [[buffer(2)]],
                          constant uint& n              [[buffer(3)]],
                          constant uint& k              [[buffer(4)]],
                          constant uint& d              [[buffer(5)]],
                          uint gid                      [[thread_position_in_grid]]) {
    if (gid >= n) return;
    device const float* p = x + gid * d;
    float best = INFINITY;
    int best_id = 0;
    for (uint c = 0; c < k; ++c) {
        device const float* q = centroids + c * d;
        float sum = 0.0f;
        for (uint j = 0; j < d; ++j) {
            const float diff = p[j] - q[j];
            sum = fma(diff, diff, sum);
        }
        if (sum < best) {
            best = sum;
            best_id = int(c);
        }
    }
    labels[gid] = best_id;
}
