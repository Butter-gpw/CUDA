#pragma once
void CPU_gemm(float *hostA, float *hostB, float *hostC, int M, int N, int K);
__global__ void naive_gemm(float *a, float *b, float *c, int M, int N, int K);
__global__ void sgemm_block(float* a, float* b, float* c, int M, int N, int K);

__global__ void sgemm_thread_tile_vec4(
        float* a, float* b, float* c, int M, int N, int K);