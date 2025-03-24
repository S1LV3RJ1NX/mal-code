#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_common.cuh"

// CUDA kernel for vector addition
// This kernel adds two vectors A and B and stores the result in vector C.
// Each thread computes one element of the result vector.
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    // Get global thread ID
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Check if the thread index is within the bounds of the array
    if (i < numElements) {
        // Perform the vector addition
        C[i] = A[i] + B[i];
    }
}

int main() {
    // Define the size of the vectors
    int numElements = 1000000; // Number of elements in the vectors
    size_t size = numElements * sizeof(float); // Size in bytes for the vectors
    
    // Allocate memory for host vectors
    float *h_A = (float *)malloc(size); // Host vector A
    float *h_B = (float *)malloc(size); // Host vector B
    float *h_C = (float *)malloc(size); // Host vector C (result)
    
    // Initialize input vectors with random values
    for (int i = 0; i < numElements; i++) {
        h_A[i] = rand() / (float)RAND_MAX; // Random value for vector A
        h_B[i] = rand() / (float)RAND_MAX; // Random value for vector B
    }
    
    // Device vectors
    float *d_A, *d_B, *d_C; // Device pointers for vectors A, B, and C
    
    // Allocate memory on the device for the vectors
    CUDA_CHECK(cudaMalloc(&d_A, size)); // Allocate device memory for vector A
    CUDA_CHECK(cudaMalloc(&d_B, size)); // Allocate device memory for vector B
    CUDA_CHECK(cudaMalloc(&d_C, size)); // Allocate device memory for vector C
    
    // Copy host vectors to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice)); // Copy A to device
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice)); // Copy B to device
    
    // Launch the vector addition kernel
    int threadsPerBlock = 256; // Number of threads per block
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock; // Number of blocks needed
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements); // Kernel launch
    
    // Copy result from device to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost)); // Copy result C back to host
    
    // Verify the result of the vector addition
    for (int i = 0; i < numElements; i++) {
        // Check if the result is correct within a tolerance
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i); // Print error message
            break; // Exit the loop on first failure
        }
    }
    printf("Vector addition completed successfully\n"); // Indicate successful completion
    
    // Free device memory
    cudaFree(d_A); // Free device memory for vector A
    cudaFree(d_B); // Free device memory for vector B
    cudaFree(d_C); // Free device memory for vector C
    
    // Free host memory
    free(h_A); // Free host memory for vector A
    free(h_B); // Free host memory for vector B
    free(h_C); // Free host memory for vector C
    
    return 0; // Exit the program
}