#include <cstdio>
#include <iostream>
#include <sys/time.h>
#include "gemm.cuh"
#include "utils.cuh"
#include "activate.cuh"
#include "softmax.cuh"
#include "norm.cuh"
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])


int main(void){
    setGPU(2);
    float *input, *output,*host;
    int n=1024;
    host = (float*)malloc(n * sizeof(float));
    for(int i=0;i<n;++i){
        host[i] = i;
    }
    float* hostSum, *deviceSum;
    hostSum = new float(0);
    cudaMalloc(&input, n * sizeof(float));
    cudaMalloc(&output, (n/ WARP_SIZE)* sizeof(float));
    cudaMalloc(&deviceSum, sizeof(float ));
    cudaMemcpy(deviceSum, hostSum,sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(input, host,n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 128;
    int numBlocks = (n + blockSize-1) /blockSize;
    block_all_reduce_sum_vec4<<<numBlocks, blockSize>>>(input, deviceSum, n);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    cudaMemcpy(hostSum, deviceSum, sizeof(float ), cudaMemcpyDeviceToHost);
    printf("sum:%f\n", *hostSum);
    cudaFree(input);
    cudaFree(output);
    cudaFree(deviceSum);
    delete hostSum;

    return 0;
}