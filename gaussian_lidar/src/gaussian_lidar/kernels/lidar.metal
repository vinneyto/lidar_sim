#include <metal_stdlib>
using namespace metal;

float3 load3(device const float *values, uint index) {
    return float3(values[index * 3], values[index * 3 + 1], values[index * 3 + 2]);
}

bool hit_box(float3 o, float3 inv_d, float3 lo, float3 hi, float limit) {
    float3 t0 = (lo - o) * inv_d;
    float3 t1 = (hi - o) * inv_d;
    float3 near3 = min(t0, t1), far3 = max(t0, t1);
    float near_t = max(max(near3.x, near3.y), max(near3.z, 0.0f));
    float far_t = min(min(far3.x, far3.y), far3.z);
    return near_t <= min(far_t, limit);
}

float hit_gaussian(float3 o, float3 d, float3 center, float3 scale, float4 q,
                   float sigma_extent, float limit) {
    float3 u = q.yzw;
    float3 ro = o - center;
    // q is (w,x,y,z); apply its conjugate to enter ellipsoid-local space.
    ro = ro + 2.0f * cross(-u, cross(-u, ro) + q.x * ro);
    float3 rd = d + 2.0f * cross(-u, cross(-u, d) + q.x * d);
    ro /= (scale * sigma_extent);
    rd /= (scale * sigma_extent);
    float a = dot(rd, rd), b = dot(ro, rd), c = dot(ro, ro) - 1.0f;
    float discriminant = b * b - a * c;
    if (discriminant < 0.0f) return limit;
    float t = (-b - sqrt(discriminant)) / a;
    return t >= 0.0f && t < limit ? t : limit;
}

kernel void trace_rays(
    device const float *origins [[buffer(0)]], device const float *directions [[buffer(1)]],
    device const float *bounds_min [[buffer(2)]], device const float *bounds_max [[buffer(3)]],
    device const int *metadata [[buffer(4)]], device const int *indices [[buffer(5)]],
    device const float *centers [[buffer(6)]], device const float *scales [[buffer(7)]],
    device const float4 *rotations [[buffer(8)]], device float *hits [[buffer(9)]],
    device const int *ray_count_buffer [[buffer(10)]],
    device const float *max_distance_buffer [[buffer(11)]],
    device const float *sigma_extent_buffer [[buffer(12)]],
    uint id [[thread_position_in_grid]]) {
    uint ray_count = uint(ray_count_buffer[0]);
    if (id >= ray_count) return;
    float max_distance = max_distance_buffer[0], sigma_extent = sigma_extent_buffer[0];
    float3 o = load3(origins, id), d = load3(directions, id);
    float3 safe_d = select(copysign(float3(1e-20f), d), d, abs(d) > 1e-20f);
    float3 inv_d = 1.0f / safe_d;
    float closest = max_distance;
    uint stack[64], top = 0; stack[top++] = 0;
    while (top) {
        uint node = stack[--top];
        if (!hit_box(o, inv_d, load3(bounds_min, node), load3(bounds_max, node), closest))
            continue;
        int4 meta = int4(metadata[node * 4], metadata[node * 4 + 1],
                         metadata[node * 4 + 2], metadata[node * 4 + 3]);
        if (meta.w > 0) {
            for (int j = 0; j < meta.w; ++j) {
                uint primitive = uint(indices[meta.z + j]);
                closest = hit_gaussian(o, d, load3(centers, primitive), load3(scales, primitive),
                                       rotations[primitive], sigma_extent, closest);
            }
        } else if (top + 2 <= 64) {
            stack[top++] = uint(meta.x); stack[top++] = uint(meta.y);
        }
    }
    float4 result = closest < max_distance ? float4(o + d * closest, closest)
                                           : float4(NAN, NAN, NAN, NAN);
    hits[id * 4] = result.x; hits[id * 4 + 1] = result.y;
    hits[id * 4 + 2] = result.z; hits[id * 4 + 3] = result.w;
}
