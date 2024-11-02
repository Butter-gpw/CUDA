#pragma once
#include "reduce.cuh"

template<const int NUM_THREADS=128>
__global__ void softmax(float *x, float *y, float* total, int N){
    const int tid = threadIdx.x;
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE -1) /WARP_SIZE;
    __shared__ float reduce_smem[WARP_SIZE];

    float exp_val = (idx < N)? expf(x[idx]) : 0.0f;
    float sum ;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    sum = warp_reduce_sum<WARP_SIZE>(exp_val);
    if(lane==0) reduce_smem[warp] = sum;
    __syncthreads();
    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    sum = warp_reduce_sum<NUM_WARPS>(sum);
    if(tid == 0) atomicAdd(total, sum);
    __threadfence();

    if(idx < N) y[idx] = exp_val / (*total);
}

template<const int NUM_THREADS=128>
__global__ void softmax_v2(float *x, float *y, float* total, int N){
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;

    float exp_val = (idx < N) ? expf(x[idx]) : 0.0f;
    float sum = block_reduce_sum<NUM_THREADS>(exp_val);
    if(tid==0) atomicAdd(total, sum);
    __threadfence();
    if(idx < N) y[idx] = exp_val / (*total);
}

template<const int NUM_THREADS=128/4>
__global__ void softmax_v2_vec4(float *x, float *y, float* total, int N){
    const int tid = threadIdx.x;
    const int idx = 4 * (blockIdx.x* blockDim.x + tid);

    float4 reg_a = FLOAT4(x[idx]);
    float4 exp_val;
    exp_val.x = (idx < N) ? expf(reg_a.x) : 0.0f;
    exp_val.y = (idx < N) ? expf(reg_a.y) : 0.0f;
    exp_val.z = (idx < N) ? expf(reg_a.z) : 0.0f;
    exp_val.w = (idx < N) ? expf(reg_a.w) : 0.0f;

    float sum = exp_val.x + exp_val.y + exp_val.z + exp_val.w;
    sum = block_reduce_sum<NUM_THREADS>(sum);
    if(tid==0) atomicAdd(total, sum);
    __threadfence();
    if(idx <N){
        float4 reg_y;
        reg_y.x = exp_val.x/(*total);
        reg_y.y = exp_val.y/(*total);
        reg_y.z = exp_val.z/(*total);
        reg_y.w = exp_val.w/(*total);
        FLOAT4(y[idx]) = reg_y;
    }
}