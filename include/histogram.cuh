#pragma once
#define INT4(value)(reinterpret_cast<int4*>(&value)[0])

//a：Nx1， y histogram
__global__ void histogram(int *a, int *y, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<N) atomicAdd(&y[a[idx]],1);
}

__global__ void histogram_vec4(int *a, int *y, int N){
    int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
    if(idx<N){
        int4 reg_a = INT4(a[idx]);
        atomicAdd(&reg_a.x,1);
        atomicAdd(&reg_a.y,1);
        atomicAdd(&reg_a.z,1);
        atomicAdd(&reg_a.w,1);
    }
}