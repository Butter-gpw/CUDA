#pragma once
#include "reduce.cuh"


// Layer Norm: x: NxK(K=128<1024), y': NxK, y'=x-mean(x)/std(x) each row，限定了K为128，所以每个block算的是一个token的norm
// mean(x) = sum(x)/K, 1/std(x) = rsqrtf( sum( (x-mean(x))^2 )/K ) each row
// grid(N*K/K), block(K<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g + b (g: scale, b: bias)
template<const int NUM_THREADS=128>
__global__ void layer_norm(float* x, float *y, float g, float b, int N, int K){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid*blockDim.x + tid;
    const float epsilon = 1e-5f;

    __shared__ float s_mean;
    __shared__ float s_variance;
    float value = (idx < N*K) ? x[idx] :0.0f;
    float sum = block_reduce_sum<NUM_THREADS>(value);
    if(tid==0) s_mean = sum / K;
    __syncthreads();
    float variance = (value - s_mean) * (value - s_mean);
    variance = block_reduce_sum<NUM_THREADS>(variance);
    if(tid==0) s_variance = rsqrtf(variance/ (float)K + epsilon);
    __syncthreads();
    if(idx < N * K) y[idx] = ((value- s_mean) * s_variance) * g +b;
}

template<const int NUM_THREADS=128/4>
__global__ void layer_norm_vec4(float* x, float *y, float g, float b, int N, int K){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = 4*(bid*blockDim.x + tid);
    const float epsilon = 1e-5f;

    __shared__ float s_mean;
    __shared__ float s_variance;

    float4 reg_x = FLOAT4(x[idx]);
    float value = reg_x.x + reg_x.y + reg_x.z + reg_x.w;
    float sum = block_reduce_sum<NUM_THREADS>(value);
    if(tid==0) s_mean = sum / K;
    __syncthreads();
    float variance = (reg_x.x - s_mean) * (reg_x.x - s_mean) + (reg_x.y - s_mean) * (reg_x.y - s_mean)
            + (reg_x.z - s_mean) * (reg_x.z - s_mean) + (reg_x.w - s_mean) * (reg_x.w - s_mean);
    variance = block_reduce_sum<NUM_THREADS>(variance);
    if(tid==0) s_variance = rsqrtf(variance / (float)K + epsilon);
    __syncthreads();
    float4 reg_y;
    reg_y.x = ((reg_x.x- s_mean) * s_variance) * g +b;
    reg_y.y = ((reg_x.y- s_mean) * s_variance) * g +b;
    reg_y.z = ((reg_x.z- s_mean) * s_variance) * g +b;
    reg_y.w = ((reg_x.w- s_mean) * s_variance) * g +b;
    if(idx < N * K){
        FLOAT4(y[idx])=reg_y;
    }
}


// RMS Norm: x: NxK(K=128<1024), y': NxK, y'=x/rms(x) each row
// 1/rms(x) = rsqrtf( sum(x^2)/K ) each row
// grid(N*K/K), block(K<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g (g: scale)
template<const int NUM_THREADS=128>
__global__ void rms_norm(float* x, float *y, float g, int N, int K){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + threadIdx.x;

    const float epsilon = 1e-5f;

    __shared__ float s_variance;
    float value = (idx <N * K) ? x[idx]:0.0f;
    float variance = value *value;
    variance = block_reduce_sum<NUM_THREADS>(variance);
    if(tid == 0) s_variance = rsqrtf(variance / (float) K + epsilon);
    __syncthreads();
    if(idx < N*K) y[idx] = (value * s_variance) * g;
}

// RMS Norm Vec4: x: NxK(K=128<1024), y': NxK, y'=x/rms(x) each row
// 1/rms(x) = rsqrtf( sum(x^2)/K ) each row
// grid(N*K/K), block(K/4<1024) N=batch_size*seq_len, K=hidden_size
// y=y'*g (g: scale)
template<const int NUM_THREADS=128/4>
__global__ void rms_norm_vec4(float* x, float *y, float g, int N, int K) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = (bid * blockDim.x + threadIdx.x) * 4;
    const float epsilon = 1e-5f;

    __shared__ float s_variance;
    float4 reg_x = FLOAT4(x[idx]);

    float variance = (idx < N * K) ? (reg_x.x * reg_x.x +
                                      reg_x.y * reg_x.y + reg_x.z * reg_x.z + reg_x.w + reg_x.w) : 0.0f;

    variance = block_reduce_sum<NUM_THREADS>(variance);
    if (tid == 0) s_variance = rsqrtf(variance / (float) K + epsilon);

    __syncthreads();
    float4 reg_y;
    reg_y.x = reg_x.x * s_variance * g;
    reg_y.y = reg_x.y * s_variance * g;
    reg_y.z = reg_x.z * s_variance * g;
    reg_y.w = reg_x.w * s_variance * g;
    if(idx < N * K ) FLOAT4(y[idx]) = reg_y;
}
