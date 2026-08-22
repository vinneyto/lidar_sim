#include <metal_stdlib>
using namespace metal;

// Spatial non-maximum suppression over the already compacted SIFT candidates.
// One thread owns one candidate and keeps it only when no stronger candidate
// lies within radius pixels in the original image. Equal-score ties are broken
// by compacted-buffer index so the result is deterministic.
kernel void suppress_nearby_candidates(
    constant float *x [[buffer(0)]],
    constant float *y [[buffer(1)]],
    constant float *responses [[buffer(2)]],
    device float *kept_abs_responses [[buffer(3)]],
    constant int &candidate_count [[buffer(4)]],
    constant float &radius [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(candidate_count)) return;

    float xi = x[gid];
    float yi = y[gid];
    float score = fabs(responses[gid]);
    float radius2 = radius * radius;

    for (int j = 0; j < candidate_count; ++j) {
        if (j == int(gid)) continue;

        float other_score = fabs(responses[j]);
        if (other_score < score) continue;
        if (other_score == score && j > int(gid)) continue;

        float dx = x[j] - xi;
        float dy = y[j] - yi;
        if (dx * dx + dy * dy <= radius2) {
            kept_abs_responses[gid] = 0.0f;
            return;
        }
    }

    kept_abs_responses[gid] = score;
}
