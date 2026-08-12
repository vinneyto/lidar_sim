#include <metal_stdlib>
using namespace metal;

constant uint MAX_CANDIDATES = 128;
constant uint MAX_STACK = 64;

bool intersects(float3 o, float3 d, float3 lo, float3 hi, float near_t, float far_t) {
    float3 inv = 1.0f / d;
    float3 a = (lo-o)*inv, b = (hi-o)*inv;
    float3 mn = min(a,b), mx = max(a,b);
    return max(max(mn.x,mn.y),max(mn.z,near_t)) <= min(min(mx.x,mx.y),min(mx.z,far_t));
}

kernel void lidar_trace(
    constant packed_float3 *means [[buffer(0)]], constant packed_float3 *scales [[buffer(1)]],
    constant float4 *quats [[buffer(2)]], constant float *opacity [[buffer(3)]],
    constant packed_float3 *box_lo [[buffer(4)]], constant packed_float3 *box_hi [[buffer(5)]],
    constant int *left [[buffer(6)]], constant int *right [[buffer(7)]],
    constant int *first [[buffer(8)]], constant int *counts [[buffer(9)]],
    constant int *primitives [[buffer(10)]], constant packed_float3 *origins [[buffer(11)]],
    constant packed_float3 *directions [[buffer(12)]], device float *ranges [[buffer(13)]],
    device packed_float3 *points [[buffer(14)]], device uchar *hits [[buffer(15)]],
    device int *hit_ids [[buffer(16)]], device float *out_alpha [[buffer(17)]],
    device int *candidate_overflow [[buffer(18)]], device int *stack_overflow [[buffer(19)]],
    constant float &near_t [[buffer(20)]], constant float &far_t [[buffer(21)]],
    constant float &cutoff2 [[buffer(22)]], constant float &threshold [[buffer(23)]],
    uint ray [[thread_position_in_grid]]) {
    float ts[MAX_CANDIDATES], alphas[MAX_CANDIDATES]; int ids[MAX_CANDIDATES];
    int stack[MAX_STACK]; uint top=1, n=0; stack[0]=0;
    float3 o=float3(origins[ray]), d=float3(directions[ray]);
    while(top) {
        int node=stack[--top];
        if(!intersects(o,d,float3(box_lo[node]),float3(box_hi[node]),near_t,far_t)) continue;
        if(counts[node] == 0) {
            if(top+2 > MAX_STACK) { atomic_fetch_add_explicit((device atomic_int*)stack_overflow,1,memory_order_relaxed); break; }
            stack[top++]=left[node]; stack[top++]=right[node]; continue;
        }
        for(int p=0;p<counts[node];++p) {
            int id=primitives[first[node]+p]; float4 q=normalize(quats[id]); float3 v=o-float3(means[id]);
            // inverse rotation (w,x,y,z quaternion), followed by inverse scale
            float3 u=(v + 2.0f*cross(-q.yzw, cross(-q.yzw,v)+q.x*v))/float3(scales[id]);
            float3 ld=(d + 2.0f*cross(-q.yzw, cross(-q.yzw,d)+q.x*d))/float3(scales[id]);
            float t=-dot(u,ld)/max(dot(ld,ld),1e-20f); float3 peak=u+t*ld; float qmin=dot(peak,peak);
            if(t<near_t || t>far_t || qmin>cutoff2) continue;
            if(n==MAX_CANDIDATES) { atomic_fetch_add_explicit((device atomic_int*)candidate_overflow,1,memory_order_relaxed); continue; }
            uint at=n++; while(at>0 && ts[at-1]>t) { ts[at]=ts[at-1]; alphas[at]=alphas[at-1]; ids[at]=ids[at-1]; --at; }
            ts[at]=t; alphas[at]=opacity[id]*exp(-0.5f*qmin); ids[at]=id;
        }
    }
    float a=0.0f; hits[ray]=0; ranges[ray]=INFINITY; points[ray]=float3(NAN); hit_ids[ray]=-1;
    for(uint i=0;i<n;++i) { a += (1.0f-a)*alphas[i]; if(a>=threshold) { hits[ray]=1; ranges[ray]=ts[i]; points[ray]=o+ts[i]*d; hit_ids[ray]=ids[i]; break; } }
    out_alpha[ray]=a;
}
