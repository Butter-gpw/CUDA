#pragma once

#include <cfloat>

#define WARP_SIZE 32
// Warp Reduce Sum
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val){
#pragma unroll
    for(int mask=kWarpSize >>1;mask >=1;mask>>=1){
        val += __shfl_xor_sync(0xffffffff, val,mask); //warp reduce函数，当mask从16~1循环时， 线程i会和线程i+mask的线程交换val的值，
    }                                                             //以0号线程为例子，mask为16时，线程0中的函数返回的是线程16的val值（线程16返回0中的val值），
    return val;                                                   //mask为8时线程,线程0中返回线程8的val值。
}                                                                 //也就是线程0会和线程16，8，4，2，1交换数据，逐渐就把所有val加到了线程0中，达到sum的目的

//Warp Reduce Max
template<const int kWarpSize=WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max(float val){
#pragma unroll
    for(int mask = kWarpSize>>1;mask >=1;mask >>=1){
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val,mask));
    }
    return val;
}

//grid 1D block 1D grid(N/128), block(128)
template<const int NUM_THREADS=128>
__device__ __forceinline__ float block_reduce_sum(float val){
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE -1) / WARP_SIZE;
    int warp = threadIdx.x /WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    static __shared__ float  shared[NUM_WARPS];

    val = warp_reduce_sum<WARP_SIZE>(val);
    if(lane == 0) shared[warp]=val;
    __syncthreads();
    val = (lane<NUM_WARPS) ? shared[lane] : 0.0f;
    val = warp_reduce_sum<NUM_WARPS>(val);
    return val;
}

template<const int NUM_THREADS=128>
__device__ __forceinline__ float block_reduce_max(float val){
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE -1 )/WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    static __shared__ float shared[NUM_WARPS];

    val = warp_reduce_max<WARP_SIZE>(val);
    if(lane == 0) shared[warp] = val;
    __syncthreads();
    val = (lane < NUM_WARPS) ? shared[lane] : -FLT_MAX;
    val = warp_reduce_max<NUM_WARPS>(val);
    return val;
}