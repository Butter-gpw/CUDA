#pragma once
#define FLOAT4(value) (reinterpret_cast<float4*>(&value)[0])

__global__ void sigmoid(float *a, float *y, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx <N) y[idx] = 1/ (1+expf(-a[idx]));
}

__global__ void sigmoid_vec4(float *a, float *y, int N){
    int idx = 4*(blockDim.x * blockIdx.x + threadIdx.x);
    if(idx <N){
        float4 reg_a = FLOAT4(a[idx]);
        float4 reg_y;
        reg_y.x =  1/ (1+expf(-reg_a.x));
        reg_y.y =  1/ (1+expf(-reg_a.y));
        reg_y.z =  1/ (1+expf(-reg_a.z));
        reg_y.w =  1/ (1+expf(-reg_a.w));
        FLOAT4(y[idx]) = reg_y;
    }
}

__global__ void relu(float *a, float *y, int N){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N) y[idx] = fmaxf(0.0f, a[idx]);
}

__global__ void relu_vec4(float *a, float *y, int N){
    int idx = 4*(blockDim.x * blockIdx.x + threadIdx.x);
    if(idx <N){
        float4 reg_a = FLOAT4(a[idx]);
        float4 reg_y;
        reg_y.x =  fmaxf(0.0f, reg_a.x);
        reg_y.y =  fmaxf(0.0f, reg_a.y);
        reg_y.z =  fmaxf(0.0f, reg_a.z);
        reg_y.w =  fmaxf(0.0f, reg_a.w);
        FLOAT4(y[idx]) = reg_y;
    }
}

