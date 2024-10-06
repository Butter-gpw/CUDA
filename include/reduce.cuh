#pragma once

#include <cfloat>
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
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

//Warp Reduce Max
template<const int kWarpSize=WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max(float val){
#pragma unroll
    for(int mask = kWarpSize>>1;mask >=1;mask >>=1){
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val,mask));
    }
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


template<const int NUM_THREADS=128> //把需要求和的数组分为128个数一组先求和再汇聚
__global__ void block_all_reduce_sum(float *a, float *y, int N){
    int tid = threadIdx.x; //块内线程号
    int idx = blockIdx.x * NUM_THREADS  + tid;//数组中数据的序号
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE-1) / WARP_SIZE; //线程束个数
    __shared__ float reduce_smem[NUM_WARPS]; //按线程束数量分配share memory
    float sum = (idx <N) ? a[idx]:0.0f; //加载数据到线程上
    int warp = tid /WARP_SIZE; //该线程属于哪个线程束
    int lane = tid % WARP_SIZE; //该线程属于线程束的哪个线程

    sum = warp_reduce_sum<WARP_SIZE>(sum);//同一个线程束内的数据先计算一个sum
    //把每个线程束的sum存到share memory中
    if(lane==0) reduce_smem[warp] = sum;

    __syncthreads();

    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if(warp==0) sum = warp_reduce_sum<WARP_SIZE>(sum); //在第一个线程束中继续计算sum
    if(tid==0) atomicAdd(y, sum);//第一个线程把该数据块的sum加到y上
}

template<const int NUM_THREADS=128/4>//block的数量是一样的，但是每个block的线程束少了
__global__ void block_all_reduce_sum_vec4(float *a, float *y, int N){//float4优化
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid)*4;//数组中数据的序号
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE-1) / WARP_SIZE; //线程束个数
    __shared__ float reduce_smem[NUM_WARPS]; //按线程束数量分配share memory

    float4 reg_a = FLOAT4(a[idx]);
    float sum = (idx < N) ? (reg_a.x+ reg_a.w+reg_a.y+reg_a.z) :0.0f;

    int warp = tid /WARP_SIZE; //该线程属于哪个线程束
    int lane = tid % WARP_SIZE; //该线程属于线程束的哪个线程

    sum = warp_reduce_sum<WARP_SIZE>(sum);//同一个线程束内的数据先计算一个sum
    //把每个线程束的sum存到share memory中
    if(lane==0) reduce_smem[warp] = sum;

    __syncthreads();

    sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.0f;
    if(warp==0) sum = warp_reduce_sum<WARP_SIZE>(sum); //在第一个线程束中继续计算sum
    if(tid==0) atomicAdd(y, sum);//第一个线程把该数据块的sum加到y上
}