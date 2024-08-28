#pragma once
#include "reduce.cuh"
//dot product
//grid(N/128), block(128),把整个向量切分为128个数一组，一组由一个block计算，一个block128个线程
//a:Nx1,b:Nx1,y = sum(elementwise_mul(a,b))

template<const int NUM_THREADS=128>
__global__ void dot(float *a, float *b, float *y, int N){
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid; //访存的idx
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE-1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    float prod = (idx<N) ? a[idx] * b[idx] : 0.0f;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    prod = warp_reduce_sum<WARP_SIZE>(prod);
    if(lane == 0) reduce_smem[warp] = prod;
    __syncthreads();
    prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if(warp==0) prod = warp_reduce_sum<NUM_WARPS>(prod);
    if(tid==0) atomicAdd(y,prod);
}

//float4优化
template<const int NUM_THREADS=128/4>
__global__ void dot_vec4(float *a, float *b, float *y, int N){
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid)*4;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE-1) / WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];
    float4 reg_a = FLOAT4(a[idx]);//reg就是寄存器
    float4 reg_b = FLOAT4(b[idx]);

    float prod =(idx<N)? (reg_a.x * reg_b.x + reg_a.y * reg_b.y + reg_a.z * reg_b.z + reg_a.w * reg_b.w) : 0.0f;
    int warp = tid /WARP_SIZE;
    int lane = tid % WARP_SIZE;
    prod = warp_reduce_sum<WARP_SIZE>(prod);
    if(lane==0) reduce_smem[warp] = prod;
    __syncthreads();
    prod = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if(warp==0) prod = warp_reduce_sum<NUM_WARPS>(prod);
    if(tid==0) atomicAdd(y, prod);
}