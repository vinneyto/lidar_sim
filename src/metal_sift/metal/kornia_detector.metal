#include <metal_stdlib>
using namespace metal;

constant float INVALID_SCORE = -3.402823466e+38f;

// Normalized 1D Gaussian kernels matching Kornia's gaussian_blur2d kernel
// sizes for BlobDoGSingle(1.0, 1.6): 9 taps for sigma=1.0 and 13 for 1.6.
constant float GAUSS_SIGMA1[9] = {
    0.0001338306f,
    0.0044318616f,
    0.0539911274f,
    0.2419714457f,
    0.3989434694f,
    0.2419714457f,
    0.0539911274f,
    0.0044318616f,
    0.0001338306f,
};

constant float GAUSS_SIGMA2[13] = {
    0.0002203804f,
    0.0018889806f,
    0.0109555901f,
    0.0429930011f,
    0.1141598703f,
    0.2051081369f,
    0.2493480813f,
    0.2051081369f,
    0.1141598703f,
    0.0429930011f,
    0.0109555901f,
    0.0018889806f,
    0.0002203804f,
};

inline int reflect_index(int index, int size) {
    if (size <= 1) return 0;
    int period = 2 * size - 2;
    int value = index % period;
    if (value < 0) value += period;
    return value < size ? value : period - value;
}

inline float pyramid_weight(int offset) {
    int distance = abs(offset);
    if (distance == 0) return 6.0f;
    if (distance == 1) return 4.0f;
    return 1.0f;
}

inline bool inside_detection_border(int x, int y, int width, int height, int border) {
    return x >= border && y >= border && x < width - border && y < height - border;
}

kernel void rgb_to_gray(
    constant float *rgb [[buffer(0)]],
    device float *gray [[buffer(1)]],
    constant int &pixel_count [[buffer(2)]],
    constant int &channels [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(pixel_count)) return;
    uint base = gid * uint(channels);
    float r = clamp(rgb[base], 0.0f, 1.0f);
    float g = clamp(rgb[base + 1], 0.0f, 1.0f);
    float b = clamp(rgb[base + 2], 0.0f, 1.0f);
    gray[gid] = 0.299f * r + 0.587f * g + 0.114f * b;
}

// align_corners=False bilinear resize, matching Kornia's MultiResolutionDetector
// resize path for the sqrt(2) upper pyramid level and pyrdown output.
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

kernel void gaussian_sigma1_horizontal(
    constant float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant int &width [[buffer(2)]],
    constant int &height [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    int count = width * height;
    if (gid >= uint(count)) return;
    int x = int(gid) % width;
    int y = int(gid) / width;
    float sum = 0.0f;
    for (int offset = -4; offset <= 4; ++offset) {
        int sx = reflect_index(x + offset, width);
        sum += GAUSS_SIGMA1[offset + 4] * src[y * width + sx];
    }
    dst[gid] = sum;
}

kernel void gaussian_sigma1_vertical(
    constant float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant int &width [[buffer(2)]],
    constant int &height [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    int count = width * height;
    if (gid >= uint(count)) return;
    int x = int(gid) % width;
    int y = int(gid) / width;
    float sum = 0.0f;
    for (int offset = -4; offset <= 4; ++offset) {
        int sy = reflect_index(y + offset, height);
        sum += GAUSS_SIGMA1[offset + 4] * src[sy * width + x];
    }
    dst[gid] = sum;
}

kernel void gaussian_sigma2_horizontal(
    constant float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant int &width [[buffer(2)]],
    constant int &height [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    int count = width * height;
    if (gid >= uint(count)) return;
    int x = int(gid) % width;
    int y = int(gid) / width;
    float sum = 0.0f;
    for (int offset = -6; offset <= 6; ++offset) {
        int sx = reflect_index(x + offset, width);
        sum += GAUSS_SIGMA2[offset + 6] * src[y * width + sx];
    }
    dst[gid] = sum;
}

kernel void gaussian_sigma2_vertical(
    constant float *src [[buffer(0)]],
    device float *dst [[buffer(1)]],
    constant int &width [[buffer(2)]],
    constant int &height [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    int count = width * height;
    if (gid >= uint(count)) return;
    int x = int(gid) % width;
    int y = int(gid) / width;
    float sum = 0.0f;
    for (int offset = -6; offset <= 6; ++offset) {
        int sy = reflect_index(y + offset, height);
        sum += GAUSS_SIGMA2[offset + 6] * src[sy * width + x];
    }
    dst[gid] = sum;
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

// Kornia pyrdown starts with the standard separable 5x5 Gaussian pyramid
// filter [1,4,6,4,1] / 16 in each dimension, using reflect padding.
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
    float sum = 0.0f;
    for (int offset = -2; offset <= 2; ++offset) {
        int sx = reflect_index(x + offset, width);
        sum += pyramid_weight(offset) * src[y * width + sx];
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
    float sum = 0.0f;
    for (int offset = -2; offset <= 2; ++offset) {
        int sy = reflect_index(y + offset, height);
        sum += pyramid_weight(offset) * src[sy * width + x];
    }
    dst[gid] = sum * (1.0f / 16.0f);
}

// First half of a separable 15x15 max filter. Kornia zeroes a 15-pixel
// border before NMS, so samples from that removed border participate as zero,
// not as their original DoG values.
kernel void max_filter15_horizontal(
    constant float *response [[buffer(0)]],
    device float *horizontal_max [[buffer(1)]],
    constant int &width [[buffer(2)]],
    constant int &height [[buffer(3)]],
    constant int &border [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    int count = width * height;
    if (gid >= uint(count)) return;
    int x = int(gid) % width;
    int y = int(gid) / width;

    float maximum = INVALID_SCORE;
    for (int dx = -7; dx <= 7; ++dx) {
        int sx = x + dx;
        float value = 0.0f;
        if (sx >= 0 && sx < width && inside_detection_border(sx, y, width, height, border)) {
            value = response[y * width + sx];
        }
        maximum = max(maximum, value);
    }
    horizontal_max[gid] = maximum;
}

// Second max-filter pass plus the score threshold / exact-NMS decision.
// Equal-valued maxima are intentionally retained, matching max-pool-based NMS.
kernel void nms15_positive_vertical(
    constant float *response [[buffer(0)]],
    constant float *horizontal_max [[buffer(1)]],
    device float *nms_response [[buffer(2)]],
    device int *level_counts [[buffer(3)]],
    constant int &level_index [[buffer(4)]],
    constant int &width [[buffer(5)]],
    constant int &height [[buffer(6)]],
    constant int &border [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    int count = width * height;
    if (gid >= uint(count)) return;
    int x = int(gid) % width;
    int y = int(gid) / width;

    if (!inside_detection_border(x, y, width, height, border)) {
        nms_response[gid] = INVALID_SCORE;
        return;
    }

    float value = response[gid];
    if (!(value > 0.0f)) {
        nms_response[gid] = INVALID_SCORE;
        return;
    }

    float maximum = INVALID_SCORE;
    for (int dy = -7; dy <= 7; ++dy) {
        int sy = y + dy;
        float row_max = 0.0f;
        if (sy >= 0 && sy < height && inside_detection_border(x, sy, width, height, border)) {
            row_max = horizontal_max[sy * width + x];
        }
        maximum = max(maximum, row_max);
    }

    if (value < maximum) {
        nms_response[gid] = INVALID_SCORE;
        return;
    }

    nms_response[gid] = value;
    atomic_fetch_add_explicit(
        (device atomic_int *)&level_counts[level_index], 1, memory_order_relaxed);
}
