#include <metal_stdlib>
using namespace metal;

constant float INVALID_SCORE = -3.402823466e+38f;

inline int reflect_index(int index, int size) {
    if (size <= 1) return 0;
    int period = 2 * size - 2;
    int value = index % period;
    if (value < 0) value += period;
    return value < size ? value : period - value;
}

kernel void rgb_to_gray(
    constant float *rgb [[buffer(0)]],
    device float *gray [[buffer(1)]],
    constant int &pixel_count [[buffer(2)]],
    constant int &channels [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(pixel_count)) return;
    uint base = gid * uint(channels);
    gray[gid] = 0.299f * rgb[base] + 0.587f * rgb[base + 1] + 0.114f * rgb[base + 2];
}

kernel void resize_bilinear(
    constant float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant int &src_width [[buffer(2)]],
    constant int &src_height [[buffer(3)]],
    constant int &dst_width [[buffer(4)]],
    constant int &dst_height [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    int count = dst_width * dst_height;
    if (gid >= uint(count)) return;

    int x = int(gid) % dst_width;
    int y = int(gid) / dst_width;
    float sx = (float(x) + 0.5f) * float(src_width) / float(dst_width) - 0.5f;
    float sy = (float(y) + 0.5f) * float(src_height) / float(dst_height) - 0.5f;
    sx = clamp(sx, 0.0f, float(src_width - 1));
    sy = clamp(sy, 0.0f, float(src_height - 1));

    int x0 = int(floor(sx));
    int y0 = int(floor(sy));
    int x1 = min(x0 + 1, src_width - 1);
    int y1 = min(y0 + 1, src_height - 1);
    float tx = sx - float(x0);
    float ty = sy - float(y0);

    float top = mix(src[y0 * src_width + x0], src[y0 * src_width + x1], tx);
    float bottom = mix(src[y1 * src_width + x0], src[y1 * src_width + x1], tx);
    dst[gid] = mix(top, bottom, ty);
}

kernel void gaussian_horizontal(
    constant float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant int &width [[buffer(2)]],
    constant int &height [[buffer(3)]],
    constant float &sigma [[buffer(4)]],
    constant int &radius [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    int count = width * height;
    if (gid >= uint(count)) return;
    int x = int(gid) % width;
    int y = int(gid) / width;
    float inv_two_sigma2 = 0.5f / (sigma * sigma);
    float sum = 0.0f;
    float weight_sum = 0.0f;
    for (int dx = -radius; dx <= radius; ++dx) {
        int sx = reflect_index(x + dx, width);
        float weight = exp(-float(dx * dx) * inv_two_sigma2);
        sum += weight * src[y * width + sx];
        weight_sum += weight;
    }
    dst[gid] = sum / weight_sum;
}

kernel void gaussian_vertical(
    constant float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant int &width [[buffer(2)]],
    constant int &height [[buffer(3)]],
    constant float &sigma [[buffer(4)]],
    constant int &radius [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    int count = width * height;
    if (gid >= uint(count)) return;
    int x = int(gid) % width;
    int y = int(gid) / width;
    float inv_two_sigma2 = 0.5f / (sigma * sigma);
    float sum = 0.0f;
    float weight_sum = 0.0f;
    for (int dy = -radius; dy <= radius; ++dy) {
        int sy = reflect_index(y + dy, height);
        float weight = exp(-float(dy * dy) * inv_two_sigma2);
        sum += weight * src[sy * width + x];
        weight_sum += weight;
    }
    dst[gid] = sum / weight_sum;
}

kernel void difference_of_gaussians(
    constant float *sigma1 [[buffer(0)]],
    constant float *sigma2 [[buffer(1)]],
    device float *response [[buffer(2)]],
    constant int &count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(count)) return;
    response[gid] = sigma2[gid] - sigma1[gid];
}

kernel void pyramid_blur5_horizontal(
    constant float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant int &width [[buffer(2)]],
    constant int &height [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    int count = width * height;
    if (gid >= uint(count)) return;
    int x = int(gid) % width;
    int y = int(gid) / width;
    constant float weights[5] = {1.0f, 4.0f, 6.0f, 4.0f, 1.0f};
    float sum = 0.0f;
    for (int i = -2; i <= 2; ++i) {
        int sx = reflect_index(x + i, width);
        sum += weights[i + 2] * src[y * width + sx];
    }
    dst[gid] = sum * (1.0f / 16.0f);
}

kernel void pyramid_blur5_vertical(
    constant float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant int &width [[buffer(2)]],
    constant int &height [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    int count = width * height;
    if (gid >= uint(count)) return;
    int x = int(gid) % width;
    int y = int(gid) / width;
    constant float weights[5] = {1.0f, 4.0f, 6.0f, 4.0f, 1.0f};
    float sum = 0.0f;
    for (int i = -2; i <= 2; ++i) {
        int sy = reflect_index(y + i, height);
        sum += weights[i + 2] * src[sy * width + x];
    }
    dst[gid] = sum * (1.0f / 16.0f);
}

kernel void nms15_positive(
    constant float *response [[buffer(0)]],
    device float *nms_response [[buffer(1)]],
    device int *level_counts [[buffer(2)]],
    constant int &level_index [[buffer(3)]],
    constant int &width [[buffer(4)]],
    constant int &height [[buffer(5)]],
    constant int &border [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    int count = width * height;
    if (gid >= uint(count)) return;
    int x = int(gid) % width;
    int y = int(gid) / width;

    if (x < border || y < border || x >= width - border || y >= height - border) {
        nms_response[gid] = INVALID_SCORE;
        return;
    }

    float value = response[gid];
    if (!(value > 0.0f)) {
        nms_response[gid] = INVALID_SCORE;
        return;
    }

    for (int dy = -7; dy <= 7; ++dy) {
        for (int dx = -7; dx <= 7; ++dx) {
            if (dx == 0 && dy == 0) continue;
            float neighbor = response[(y + dy) * width + (x + dx)];
            if (neighbor > value) {
                nms_response[gid] = INVALID_SCORE;
                return;
            }
        }
    }

    nms_response[gid] = value;
    atomic_fetch_add_explicit(
        (device atomic_int *)&level_counts[level_index], 1, memory_order_relaxed);
}
