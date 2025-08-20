// vector_redux.cu

#include <cuda_runtime.h>
#include <stdexcept>

__global__
void redux_kernel(const float* a, float* result, int blockSize, int segments) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < segments) {
        float temp = 0.0f;
        for (int j = 0; j < blockSize; j++) {
            int index = (i * blockSize) + j; 
            temp += a[index];
            // printf("TEMP i=%d, j=%d, index=%d, value=%.2f, sum=%.2f\n", i, j, index, a[index], temp);
        }
        result[i] = temp;
    }
}

void vector_redux_cuda(const float* a, float* result, int blockSize, int segments) {
    float *d_a, *d_result;
    size_t input_size = segments * blockSize * sizeof(float);  // Full input array size
    size_t output_size = segments * sizeof(float);            // Output array size

    cudaMalloc(&d_a, input_size);   // Allocate for full input array
    cudaMalloc(&d_result, output_size);

    cudaMemcpy(d_a, a, input_size, cudaMemcpyHostToDevice);  // Copy full input array

    int threads = 256;
    int blocks = (segments + threads - 1) / threads;
    redux_kernel<<<blocks, threads>>>(d_a, d_result, blockSize, segments);

    cudaMemcpy(result, d_result, output_size, cudaMemcpyDeviceToHost);  // Copy result array

    cudaFree(d_a);
    cudaFree(d_result);
}

