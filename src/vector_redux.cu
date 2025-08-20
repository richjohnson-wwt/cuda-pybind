// vector_redux.cu

#include <cuda_runtime.h>
#include <stdexcept>

__global__
void redux_kernel(const float* a, float* result, int blockSize, int segments) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < segments) {
        float temp = 0.0f;
        for (int j = 0; j < blockSize; j++) {
            temp += a[(i* blockSize) + j];
            printf("TEMP i, j, temp: %d, %d, %.2f\n", i, j, temp);
        }
        result[i] = temp;
    }
}

void vector_redux_cuda(const float* a, float* result, int blockSize, int segments) {
    float *d_a, *d_result;
    size_t size = segments * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_result, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (segments + threads - 1) / threads;
    redux_kernel<<<blocks, threads>>>(d_a, d_result, blockSize, segments);

    cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_result);
}
