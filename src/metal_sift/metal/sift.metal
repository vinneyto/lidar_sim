#include <metal_stdlib>
using namespace metal;

constant float PI_F = 3.14159265358979323846f;
constant int ORIENTATION_BINS = 36;
constant int MAX_LOCALIZATION_STEPS = 5;

kernel void rgb_to_gray(
    constant float *rgb [[buffer(0)]],
    device float *gray [[buffer(1)]],
    constant int &pixel_count [[buffer(2)]],
    constant int &channels [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(pixel_count)) return;
    uint base = gid * uint(channels);
    float r = rgb[base + 0];
    float g = rgb[base + 1];
    float b = rgb[base + 2];
    gray[gid] = 0.299f * r + 0.587f * g + 0.114f * b;
}

kernel void gaussian_horizontal(
    constant float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant int &width [[buffer(2)]],
    constant int &height [[buffer(3)]],
    constant float &sigma [[buffer(4)]],
    constant int &radius [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    int n = width * height;
    if (gid >= uint(n)) return;
    int x = int(gid) % width;
    int y = int(gid) / width;
    float inv_two_sigma2 = 0.5f / max(sigma * sigma, 1e-8f);
    float sum = 0.0f;
    float weight_sum = 0.0f;
    for (int dx = -radius; dx <= radius; ++dx) {
        int sx = clamp(x + dx, 0, width - 1);
        float w = exp(-float(dx * dx) * inv_two_sigma2);
        sum += w * src[y * width + sx];
        weight_sum += w;
    }
    dst[gid] = sum / max(weight_sum, 1e-12f);
}

kernel void gaussian_vertical(
    constant float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant int &width [[buffer(2)]],
    constant int &height [[buffer(3)]],
    constant float &sigma [[buffer(4)]],
    constant int &radius [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    int n = width * height;
    if (gid >= uint(n)) return;
    int x = int(gid) % width;
    int y = int(gid) / width;
    float inv_two_sigma2 = 0.5f / max(sigma * sigma, 1e-8f);
    float sum = 0.0f;
    float weight_sum = 0.0f;
    for (int dy = -radius; dy <= radius; ++dy) {
        int sy = clamp(y + dy, 0, height - 1);
        float w = exp(-float(dy * dy) * inv_two_sigma2);
        sum += w * src[sy * width + x];
        weight_sum += w;
    }
    dst[gid] = sum / max(weight_sum, 1e-12f);
}

kernel void difference_of_gaussians(
    constant float *lower [[buffer(0)]],
    constant float *upper [[buffer(1)]],
    device float *dog [[buffer(2)]],
    constant int &count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(count)) return;
    dog[gid] = upper[gid] - lower[gid];
}

kernel void downsample_half(
    constant float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant int &src_width [[buffer(2)]],
    constant int &src_height [[buffer(3)]],
    constant int &dst_width [[buffer(4)]],
    constant int &dst_height [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    int n = dst_width * dst_height;
    if (gid >= uint(n)) return;
    int x = int(gid) % dst_width;
    int y = int(gid) / dst_width;
    int sx = min(x * 2, src_width - 1);
    int sy = min(y * 2, src_height - 1);

    // This source level already has twice the octave base sigma. Sampling every
    // second pixel therefore produces the next octave at sigma0 without adding
    // another unaccounted 2x2 box blur.
    dst[gid] = src[sy * src_width + sx];
}

inline float dog_at(
    int level,
    constant float *dog0,
    constant float *dog1,
    constant float *dog2,
    constant float *dog3,
    constant float *dog4,
    int x,
    int y,
    int width) {
    int index = y * width + x;
    if (level == 0) return dog0[index];
    if (level == 1) return dog1[index];
    if (level == 2) return dog2[index];
    if (level == 3) return dog3[index];
    return dog4[index];
}

inline float gaussian_at(
    int level,
    constant float *gaussian1,
    constant float *gaussian2,
    constant float *gaussian3,
    int index) {
    if (level <= 1) return gaussian1[index];
    if (level == 2) return gaussian2[index];
    return gaussian3[index];
}

kernel void detect_extrema(
    constant float *dog0 [[buffer(0)]],
    constant float *dog1 [[buffer(1)]],
    constant float *dog2 [[buffer(2)]],
    constant float *dog3 [[buffer(3)]],
    constant float *dog4 [[buffer(4)]],
    constant float *gaussian1 [[buffer(5)]],
    constant float *gaussian2 [[buffer(6)]],
    constant float *gaussian3 [[buffer(7)]],
    device float *out_x [[buffer(8)]],
    device float *out_y [[buffer(9)]],
    device float *out_scale [[buffer(10)]],
    device float *out_response [[buffer(11)]],
    device float *out_orientation [[buffer(12)]],
    device int *counter [[buffer(13)]],
    device int *overflow [[buffer(14)]],
    constant int &width [[buffer(15)]],
    constant int &height [[buffer(16)]],
    constant float &contrast_threshold [[buffer(17)]],
    constant float &edge_threshold [[buffer(18)]],
    constant float &sigma0 [[buffer(19)]],
    constant float &scale_step [[buffer(20)]],
    constant float &coordinate_scale [[buffer(21)]],
    constant int &dog_level [[buffer(22)]],
    constant int &max_candidates [[buffer(23)]],
    uint gid [[thread_position_in_grid]]) {
    int n = width * height;
    if (gid >= uint(n)) return;
    int x = int(gid) % width;
    int y = int(gid) / width;
    if (x < 2 || y < 2 || x >= width - 2 || y >= height - 2) return;

    float value = dog_at(dog_level, dog0, dog1, dog2, dog3, dog4, x, y, width);
    if (fabs(value) < 0.5f * contrast_threshold) return;

    bool is_maximum = true;
    bool is_minimum = true;
    for (int ds = -1; ds <= 1; ++ds) {
        int level = dog_level + ds;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (ds == 0 && dx == 0 && dy == 0) continue;
                float neighbor = dog_at(
                    level, dog0, dog1, dog2, dog3, dog4,
                    x + dx, y + dy, width);
                is_maximum = is_maximum && value > neighbor;
                is_minimum = is_minimum && value < neighbor;
            }
        }
    }
    if (!is_maximum && !is_minimum) return;

    int refined_x = x;
    int refined_y = y;
    int refined_level = dog_level;
    float ox = 0.0f;
    float oy = 0.0f;
    float os = 0.0f;
    float dx = 0.0f;
    float dy = 0.0f;
    float ds = 0.0f;
    float dxx = 0.0f;
    float dyy = 0.0f;
    float dss = 0.0f;
    float dxy = 0.0f;
    float dxs = 0.0f;
    float dys = 0.0f;
    bool converged = false;

    // Lowe-style localization: if the quadratic extremum lies more than half a
    // sample away, move to that neighbouring pixel/scale and fit again. The
    // previous implementation rejected those points, which made a real feature
    // disappear whenever tiny frame-to-frame changes pushed its offset across
    // the rejection boundary.
    for (int iteration = 0; iteration < MAX_LOCALIZATION_STEPS; ++iteration) {
        value = dog_at(
            refined_level, dog0, dog1, dog2, dog3, dog4,
            refined_x, refined_y, width);

        dx = 0.5f * (
            dog_at(refined_level, dog0, dog1, dog2, dog3, dog4, refined_x + 1, refined_y, width) -
            dog_at(refined_level, dog0, dog1, dog2, dog3, dog4, refined_x - 1, refined_y, width));
        dy = 0.5f * (
            dog_at(refined_level, dog0, dog1, dog2, dog3, dog4, refined_x, refined_y + 1, width) -
            dog_at(refined_level, dog0, dog1, dog2, dog3, dog4, refined_x, refined_y - 1, width));
        ds = 0.5f * (
            dog_at(refined_level + 1, dog0, dog1, dog2, dog3, dog4, refined_x, refined_y, width) -
            dog_at(refined_level - 1, dog0, dog1, dog2, dog3, dog4, refined_x, refined_y, width));

        dxx =
            dog_at(refined_level, dog0, dog1, dog2, dog3, dog4, refined_x + 1, refined_y, width) +
            dog_at(refined_level, dog0, dog1, dog2, dog3, dog4, refined_x - 1, refined_y, width) -
            2.0f * value;
        dyy =
            dog_at(refined_level, dog0, dog1, dog2, dog3, dog4, refined_x, refined_y + 1, width) +
            dog_at(refined_level, dog0, dog1, dog2, dog3, dog4, refined_x, refined_y - 1, width) -
            2.0f * value;
        dss =
            dog_at(refined_level + 1, dog0, dog1, dog2, dog3, dog4, refined_x, refined_y, width) +
            dog_at(refined_level - 1, dog0, dog1, dog2, dog3, dog4, refined_x, refined_y, width) -
            2.0f * value;
        dxy = 0.25f * (
            dog_at(refined_level, dog0, dog1, dog2, dog3, dog4, refined_x + 1, refined_y + 1, width) -
            dog_at(refined_level, dog0, dog1, dog2, dog3, dog4, refined_x + 1, refined_y - 1, width) -
            dog_at(refined_level, dog0, dog1, dog2, dog3, dog4, refined_x - 1, refined_y + 1, width) +
            dog_at(refined_level, dog0, dog1, dog2, dog3, dog4, refined_x - 1, refined_y - 1, width));
        dxs = 0.25f * (
            dog_at(refined_level + 1, dog0, dog1, dog2, dog3, dog4, refined_x + 1, refined_y, width) -
            dog_at(refined_level + 1, dog0, dog1, dog2, dog3, dog4, refined_x - 1, refined_y, width) -
            dog_at(refined_level - 1, dog0, dog1, dog2, dog3, dog4, refined_x + 1, refined_y, width) +
            dog_at(refined_level - 1, dog0, dog1, dog2, dog3, dog4, refined_x - 1, refined_y, width));
        dys = 0.25f * (
            dog_at(refined_level + 1, dog0, dog1, dog2, dog3, dog4, refined_x, refined_y + 1, width) -
            dog_at(refined_level + 1, dog0, dog1, dog2, dog3, dog4, refined_x, refined_y - 1, width) -
            dog_at(refined_level - 1, dog0, dog1, dog2, dog3, dog4, refined_x, refined_y + 1, width) +
            dog_at(refined_level - 1, dog0, dog1, dog2, dog3, dog4, refined_x, refined_y - 1, width));

        float det_h =
            dxx * (dyy * dss - dys * dys) -
            dxy * (dxy * dss - dys * dxs) +
            dxs * (dxy * dys - dyy * dxs);
        if (fabs(det_h) < 1e-10f) return;

        float inv00 = (dyy * dss - dys * dys) / det_h;
        float inv01 = (dxs * dys - dxy * dss) / det_h;
        float inv02 = (dxy * dys - dxs * dyy) / det_h;
        float inv11 = (dxx * dss - dxs * dxs) / det_h;
        float inv12 = (dxy * dxs - dxx * dys) / det_h;
        float inv22 = (dxx * dyy - dxy * dxy) / det_h;

        ox = -(inv00 * dx + inv01 * dy + inv02 * ds);
        oy = -(inv01 * dx + inv11 * dy + inv12 * ds);
        os = -(inv02 * dx + inv12 * dy + inv22 * ds);
        if (!isfinite(ox) || !isfinite(oy) || !isfinite(os)) return;

        int step_x = ox > 0.5f ? 1 : (ox < -0.5f ? -1 : 0);
        int step_y = oy > 0.5f ? 1 : (oy < -0.5f ? -1 : 0);
        int step_s = os > 0.5f ? 1 : (os < -0.5f ? -1 : 0);
        if (step_x == 0 && step_y == 0 && step_s == 0) {
            converged = true;
            break;
        }

        refined_x += step_x;
        refined_y += step_y;
        refined_level += step_s;
        if (
            refined_x < 2 || refined_y < 2 ||
            refined_x >= width - 2 || refined_y >= height - 2 ||
            refined_level < 1 || refined_level > 3) {
            return;
        }
    }
    if (!converged) return;

    float refined_response = value + 0.5f * (dx * ox + dy * oy + ds * os);
    if (fabs(refined_response) < contrast_threshold) return;

    float spatial_det = dxx * dyy - dxy * dxy;
    if (spatial_det <= 0.0f) return;
    float trace = dxx + dyy;
    float edge_limit = ((edge_threshold + 1.0f) * (edge_threshold + 1.0f)) / edge_threshold;
    if ((trace * trace) / spatial_det >= edge_limit) return;

    float keypoint_sigma = sigma0 * pow(scale_step, float(refined_level) + os);
    float histogram[ORIENTATION_BINS];
    for (int i = 0; i < ORIENTATION_BINS; ++i) histogram[i] = 0.0f;
    float orientation_sigma = 1.5f * keypoint_sigma;
    int radius = min(16, max(1, int(round(3.0f * orientation_sigma))));
    float inv_two_orientation_sigma2 = 0.5f / max(orientation_sigma * orientation_sigma, 1e-8f);
    for (int wy = -radius; wy <= radius; ++wy) {
        int py = refined_y + wy;
        if (py <= 0 || py >= height - 1) continue;
        for (int wx = -radius; wx <= radius; ++wx) {
            int px = refined_x + wx;
            if (px <= 0 || px >= width - 1) continue;
            int center = py * width + px;
            float gx =
                gaussian_at(refined_level, gaussian1, gaussian2, gaussian3, center + 1) -
                gaussian_at(refined_level, gaussian1, gaussian2, gaussian3, center - 1);
            float gy =
                gaussian_at(refined_level, gaussian1, gaussian2, gaussian3, center + width) -
                gaussian_at(refined_level, gaussian1, gaussian2, gaussian3, center - width);
            float magnitude = sqrt(gx * gx + gy * gy);
            float angle = atan2(gy, gx);
            float normalized = (angle + PI_F) * (float(ORIENTATION_BINS) / (2.0f * PI_F));
            int bin = int(floor(normalized)) % ORIENTATION_BINS;
            if (bin < 0) bin += ORIENTATION_BINS;
            float weight = exp(-float(wx * wx + wy * wy) * inv_two_orientation_sigma2);
            histogram[bin] += weight * magnitude;
        }
    }
    int dominant_bin = 0;
    float dominant_value = histogram[0];
    for (int bin = 1; bin < ORIENTATION_BINS; ++bin) {
        if (histogram[bin] > dominant_value) {
            dominant_value = histogram[bin];
            dominant_bin = bin;
        }
    }
    float orientation =
        (float(dominant_bin) + 0.5f) * (2.0f * PI_F / float(ORIENTATION_BINS)) - PI_F;

    int index = atomic_fetch_add_explicit((device atomic_int *)counter, 1, memory_order_relaxed);
    if (index >= max_candidates) {
        atomic_fetch_add_explicit((device atomic_int *)overflow, 1, memory_order_relaxed);
        return;
    }
    out_x[index] = (float(refined_x) + ox) * coordinate_scale;
    out_y[index] = (float(refined_y) + oy) * coordinate_scale;
    out_scale[index] = keypoint_sigma * coordinate_scale;
    out_response[index] = refined_response;
    out_orientation[index] = orientation;
}
