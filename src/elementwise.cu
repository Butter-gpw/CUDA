#include "elementwise.cuh"
#define FLOAT4(value)(reinterpret_cast<float4*>(&value)[0])

__global__ void elementwise_add(float* a, float* b, float* c, int N){//N是元素的个数
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx<N){
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void elementwise_add_vec4(float* a, float* b, float* c, int N){
    int idx = 4*(blockDim.x * blockIdx.x + threadIdx.x);
    if(idx<N){
        float4 vec_a = FLOAT4(a[idx]);
        float4 vec_b = FLOAT4(b[idx]);
        float4 vec_c;
        vec_c.x = vec_a.x + vec_b.x;
        vec_c.y = vec_a.y + vec_b.y;
        vec_c.z = vec_a.z + vec_b.z;
        vec_c.w = vec_a.w + vec_b.w;
        FLOAT4(c[idx]) = vec_c;
    }
}