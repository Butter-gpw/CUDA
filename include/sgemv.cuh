#pragma once
#include "reduce.cuh"

//sgemv是基于warp reduce的矩阵向量乘。
//假设K是32的倍数，每个warp负责一行
//grid(M/4),block(32,4),blockDim.x=32=K,blockDim.y=4，把矩阵的行分成4组，一个block计算机组，block内分为4个warp，一个warp计算一行
//a:MxK矩阵，x:kx1,y:Mx1,计算y=a * x;
__global__ void sgemv_k32(float *a, float* x, float *y, int M, int K){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int lane = tx % WARP_SIZE;
    int m = bx * blockDim.y + ty;//行号，第几个block*block的维度加上block内的行号ty，ty是0~3取值。
    if(m<M){//一个warp负责一行，一行会按照WARP_SIZE分成很多个组进行计算
        float sum = 0.0f;//这个sum是用来加不同组中同一个lane的数据，也就是会吧同一行的0，32，64...位置的数先加起来
        int NUM_WARPS = (K + WARP_SIZE -1)/ WARP_SIZE;//K切分为32一组计算
        #pragma unroll
        for(int w=0; w<NUM_WARPS; ++w){
            int k=w * WARP_SIZE + lane;//先算不同组中同一个lane的数然后加起来
            sum += a[m * K + k] * x[k];
        }
        sum = warp_reduce_sum<WARP_SIZE>(sum);//最后使用warp reduce sum归约。
        if(lane==0) y[m]=sum;//把一行的运算结果放进y中。
    }
}


//sgemv warp k128 vec4优化
//假设K为128的倍数
//grid(M/4),block(32,4)
//a:MxK矩阵，x:kx1,y:Mx1,计算y=a * x;
__global__ void sgemv_k128_vec4(float *a, float* x, float *y, int M, int K){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int lane = tx % WARP_SIZE;
    int m = blockDim.y * bx + ty;
    if(m<M){
        float  sum=0.0f;
        int NUM_WARPS =((K + WARP_SIZE -1)/ WARP_SIZE +4 -1)/4; //一个warp处理32个float4，所以warp个数应该是K/32/4
#pragma unroll
        for(int w=0; w<NUM_WARPS; ++w){
            int k = (w*NUM_WARPS + lane)*4;
            float4 reg_x = FLOAT4(x[k]);//接收x中的一个float4
            float4 reg_a = FLOAT4(a[m*K+k]);//接收a中的一个float4
            sum += (reg_x.x * reg_a.x + reg_x.y + reg_a.y + reg_x.z*reg_a.z+reg_x.w*reg_a.w); //直接计算对应位置的内积
        }
        sum = warp_reduce_sum(sum);
        if(lane==0) y[m] = sum;
    }
}


//假设K为16，每个warp负责两行，每行是16个元素，与上面的不同
//这次是一个warp负责两行
template<const int ROW_PER_WARP=2>
__global__ void sgemv_k16(float *a, float* x, float *y, int M, int K){
    constexpr int K_WARP_SIZE = (WARP_SIZE + ROW_PER_WARP -1) / ROW_PER_WARP; //按照设定这个就是16，也就是计算一行用的线程数量
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int lane = tx % WARP_SIZE;
    int k = lane % K_WARP_SIZE;
    int m = (blockDim.y * bx +ty)*ROW_PER_WARP + lane / K_WARP_SIZE; //每一行线程对应两行的数据
    if(m<M){
        float sum = a[m*K+k] * x[k];
        sum = warp_reduce_sum<K_WARP_SIZE>(sum);
        if(k==0) y[m]=sum;
    }
}