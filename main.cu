#include <cstdio>
#include <sys/time.h>
#include "gemm.cuh"
#include "utils.cuh"
#include "reduce.cuh"
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])


__global__ void warp_reduce_sum_test(float* output, float *input, int n){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n){
        float val = input[idx];
        val = warp_reduce_sum(val);
        if(threadIdx.x % WARP_SIZE==0){
            output[idx / WARP_SIZE]=val;
            printf("warpID:%d, %f\n", blockIdx.x, val);
        }
    }
}

int main(void){
    setGPU(1);
    float *input, *output,*host;
    int n=1024;
    host = (float*)malloc(n * sizeof(float));
    for(int i=0;i<n;++i){
        host[i] = i;
    }
    cudaMalloc(&input, n * sizeof(float));
    cudaMalloc(&output, (n/ WARP_SIZE)* sizeof(float));
    cudaMemcpy(input, host,n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 128;
    int numBlocks = (n + blockSize-1) /blockSize;
    warp_reduce_sum_test<<<numBlocks, blockSize>>>(output, input, n);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    cudaFree(input);
    cudaFree(output);

    return 0;
}