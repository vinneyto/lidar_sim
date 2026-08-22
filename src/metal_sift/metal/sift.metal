#include <metal_stdlib>
using namespace metal;

constant float PI_F = 3.14159265358979323846f;
constant int ORIENTATION_BINS = 36;

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
    int sx1 = min(sx + 1, src_width - 1);
    int sy1 = min(sy + 1, src_height - 1);
    dst[gid] = 0.25f * (
        src[sy * src_width + sx] +
        src[sy * src_width + sx1] +
        src[sy1 * src_width + sx] +
        src[sy1 * src_width + sx1]);
}

inline float dog_at(constant float *image, int x, int y, int width) {
    return image[y * width + x];
}

kernel void detect_extrema(
    constant float *dog_prev [[buffer(0)]],
    constant float *dog_cur [[buffer(1)]],
    constant float *dog_next [[buffer(2)]],
    constant float *gaussian [[buffer(3)]],
    device float *out_x [[buffer(4)]],
    device float *out_y [[buffer(5)]],
    device float *out_scale [[buffer(6)]],
    device float *out_response [[buffer(7)]],
    device float *out_orientation [[buffer(8)]],
    device int *counter [[buffer(9)]],
    device int *overflow [[buffer(10)]],
    constant int &width [[buffer(11)]],
    constant int &height [[buffer(12)]],
    constant float &contrast_threshold [[buffer(13)]],
    constant float &edge_threshold [[buffer(14)]],
    constant float &local_sigma [[buffer(15)]],
    constant float &coordinate_scale [[buffer(16)]],
    constant int &max_candidates [[buffer(17)]],
    uint gid [[thread_position_in_grid]]) {
    int n = width * height;
    if (gid >= uint(n)) return;
    int x = int(gid) % width;
    int y = int(gid) / width;
    if (x < 2 || y < 2 || x >= width - 2 || y >= height - 2) return;

    float value = dog_cur[gid];
    if (fabs(value) < 0.5f * contrast_threshold) return;

    bool is_maximum = true;
    bool is_minimum = true;
    for (int ds = -1; ds <= 1; ++ds) {
        constant float *layer = ds < 0 ? dog_prev : (ds > 0 ? dog_next : dog_cur);
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (ds == 0 && dx == 0 && dy == 0) continue;
                float neighbor = layer[(y + dy) * width + (x + dx)];
                is_maximum = is_maximum && value > neighbor;
                is_minimum = is_minimum && value < neighbor;
            }
        }
    }
    if (!is_maximum && !is_minimum) return;

    float dx = 0.5f * (dog_at(dog_cur, x + 1, y, width) - dog_at(dog_cur, x - 1, y, width));
    float dy = 0.5f * (dog_at(dog_cur, x, y + 1, width) - dog_at(dog_cur, x, y - 1, width));
    float ds = 0.5f * (dog_next[gid] - dog_prev[gid]);

    float dxx = dog_at(dog_cur, x + 1, y, width) + dog_at(dog_cur, x - 1, y, width) - 2.0f * value;
    float dyy = dog_at(dog_cur, x, y + 1, width) + dog_at(dog_cur, x, y - 1, width) - 2.0f * value;
    float dss = dog_next[gid] + dog_prev[gid] - 2.0f * value;
    float dxy = 0.25f * (
        dog_at(dog_cur, x + 1, y + 1, width) - dog_at(dog_cur, x + 1, y - 1, width) -
        dog_at(dog_cur, x - 1, y + 1, width) + dog_at(dog_cur, x - 1, y - 1, width));
    float dxs = 0.25f * (
        dog_next[y * width + x + 1] - dog_next[y * width + x - 1] -
        dog_prev[y * width + x + 1] + dog_prev[y * width + x - 1]);
    float dys = 0.25f * (
        dog_next[(y + 1) * width + x] - dog_next[(y - 1) * width + x] -
        dog_prev[(y + 1) * width + x] + dog_prev[(y - 1) * width + x]);

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

    float ox = -(inv00 * dx + inv01 * dy + inv02 * ds);
    float oy = -(inv01 * dx + inv11 * dy + inv12 * ds);
    float os = -(inv02 * dx + inv12 * dy + inv22 * ds);
    if (fabs(ox) > 0.6f || fabs(oy) > 0.6f || fabs(os) > 0.6f) return;

    float refined_response = value + 0.5f * (dx * ox + dy * oy + ds * os);
    if (fabs(refined_response) < contrast_threshold) return;

    float spatial_det = dxx * dyy - dxy * dxy;
    if (spatial_det <= 0.0f) return;
    float trace = dxx + dyy;
    float edge_limit = ((edge_threshold + 1.0f) * (edge_threshold + 1.0f)) / edge_threshold;
    if ((trace * trace) / spatial_det >= edge_limit) return;

    float histogram[ORIENTATION_BINS];
    for (int i = 0; i < ORIENTATION_BINS; ++i) histogram[i] = 0.0f;
    float orientation_sigma = 1.5f * local_sigma;
    int radius = min(16, max(1, int(round(3.0f * orientation_sigma))));
    float inv_two_orientation_sigma2 = 0.5f / max(orientation_sigma * orientation_sigma, 1e-8f);
    for (int wy = -radius; wy <= radius; ++wy) {
        int py = y + wy;
        if (py <= 0 || py >= height - 1) continue;
        for (int wx = -radius; wx <= radius; ++wx) {
            int px = x + wx;
            if (px <= 0 || px >= width - 1) continue;
            float gx = gaussian[py * width + px + 1] - gaussian[py * width + px - 1];
            float gy = gaussian[(py + 1) * width + px] - gaussian[(py - 1) * width + px];
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
    float orientation = (float(dominant_bin) + 0.5f) * (2.0f * PI_F / float(ORIENTATION_BINS)) - PI_F;

    int index = atomic_fetch_add_explicit((device atomic_int *)counter, 1, memory_order_relaxed);
    if (index >= max_candidates) {
        atomic_fetch_add_explicit((device atomic_int *)overflow, 1, memory_order_relaxed);
        return;
    }
    out_x[index] = (float(x) + ox) * coordinate_scale;
    out_y[index] = (float(y) + oy) * coordinate_scale;
    out_scale[index] = local_sigma * exp2(os / 3.0f) * coordinate_scale;
    out_response[index] = refined_response;
    out_orientation[index] = orientation;
}
