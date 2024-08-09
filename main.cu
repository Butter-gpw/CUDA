#include <cstdio>
#include <sys/time.h>
#include "gemm.cuh"
#include "utils.cuh"
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])


int main(void){
    setGPU(0);
    float *hostA, *hostB, *hostC, *serialC;
    int M = 1024;
    int K = 1024;
    int N = 1024;

    hostA = (float *)malloc(M * K * sizeof(float));
    hostB = (float *)malloc(N * K * sizeof(float));
    hostC = (float *)malloc(M * N * sizeof(float));
    serialC = (float *)malloc(M * N * sizeof(float));
    for (int i = 0; i < M * K; i++)
    {
        hostA[i] = i % 3;
    }
    for (int i = 0; i < N * K; i++)
    {
        hostB[i] = i % 3;
    }
    hostMatrix(hostA, hostB, hostC, M, N, K,sgemm_thread_tile_vec4);
    double st, ela;
    st = get_walltime();
    CPU_gemm(hostA, hostB, serialC, M, N, K);
    ela = get_walltime() - st;
    float error = compare(hostC, serialC, M, N);
    printf("CPU time:%.2f second\n", ela);
    printf("The error between CPU and GPU: %.4e\n", error);
    free(hostA);
    free(hostB);
    free(hostC);
    free(serialC);
    return 0;
}