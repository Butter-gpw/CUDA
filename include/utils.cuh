#pragma once
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include "gemm.cuh"
const int TM = 8;
const int TN = 8;
const int BLOCK_DIM_x = 16;
const int BLOCK_DIM_y = 16;
const int BM = TM * BLOCK_DIM_x;
const int BN = TN * BLOCK_DIM_y;
const int BK = 8;

void setGPU(int iDev){
    int iDeviceCount=0;
    cudaError_t err = cudaGetDeviceCount(&iDeviceCount);
    if(err != cudaSuccess || iDeviceCount==0){
        printf("No CUDA campatable GPU found\n");
        exit(-1);
    }else{
        printf("The count of GPUs is %d.\n", iDeviceCount);
    }
    err = cudaSetDevice(iDev);
    if(err!=cudaSuccess){
        printf("fail to set GPU %d for computing.\n", iDev);
        exit(-1);
    }else{
        printf("set GPU %d for computing.\n", iDev);
    }
}

cudaError_t ErrorCheck(cudaError_t error_code, const char* filename, int lineNumber){ //filename通常传入__FILE__,lineNumber通常传入__LINE__
    if(error_code != cudaSuccess){
        printf("CUDA error:\r\ncode=%d, name=%s, description=%s\r\nfile=%s, line=%d\r\n",
               error_code, cudaGetErrorName(error_code), cudaGetErrorString(error_code), filename, lineNumber);
    }
    return error_code;
}

double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}


float compare(float *hostC, float *serialC, int M, int N)
{
    float error = 0;
    for (int i = 0; i < M * N; i++)
    {
        error = fmax(error, fabs(hostC[i] - serialC[i]));
    }
    return error;
}

void hostMatrix(float *hostA, float *hostB, float *hostC, int M, int N, int K, decltype(sgemm_block) gemm)
{
    double st, ela;
    st = get_walltime();

    float *dA, *dB, *dC;
    cudaMalloc((void **)&dA, M * K * sizeof(float));
    cudaMalloc((void **)&dB, N * K * sizeof(float));
    cudaMalloc((void **)&dC, M * N * sizeof(float));

    cudaMemcpy(dA, hostA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hostB, N * K * sizeof(float), cudaMemcpyHostToDevice);

    int num_blocks_x = (M + BM - 1) / BM;
    int num_blocks_y = (N + BN - 1) / BN;
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_blocks_x, num_blocks_y, 1);
    int repeat = 20;

    gemm<<<grid_dim, block_dim>>>(dA, dB, dC, M, N, K); //计算矩阵的核函数调用一次先预热
    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < repeat; i++)
    {
        gemm<<<grid_dim, block_dim>>>(dA, dB, dC, M, N, K);//计重复计算然后计算平均时间
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time

    cudaMemcpy(hostC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    ela = get_walltime() - st;
    printf("M-K-N: %d-%d-%d\n", M, K, N);
    printf("GPU use time: %.4f second\n", ela);
    printf("kernel time: %.4f second, %.4f ms\n", ker_time / (repeat * 1000.), ker_time / repeat);
    printf("grid dim: %d, %d, %d\n", grid_dim.x, grid_dim.y, grid_dim.z);
    printf("block dim: %d, %d, %d\n", block_dim.x, block_dim.y, block_dim.z);
}