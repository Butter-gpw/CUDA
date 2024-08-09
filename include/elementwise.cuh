#pragma once

__global__ void elementwise_add(float* a, float* b, float* c, int N);
__global__ void elementwise_add_vec4(float* a, float* b, float* c, int N);