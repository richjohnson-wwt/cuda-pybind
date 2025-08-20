// vector_add.cu

#include <cuda_runtime.h>
#include <stdexcept>

__global__
void add_kernel(const float* a, const float* b, float* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        result[i] = a[i] + b[i];
    }
}

void vector_add_cuda(const float* a, const float* b, float* result, int n) {
    float *d_a, *d_b, *d_result;
    size_t size = n * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_result, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(d_a, d_b, d_result, n);

    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}
