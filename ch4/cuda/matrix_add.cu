#include <stdio.h>
#include <cuda_runtime.h>
#include "cuda_common.cuh"

// CUDA kernel for matrix addition
// This kernel adds two matrices A and B and stores the result in matrix C.
// Each thread computes one element of the result matrix based on its row and column indices.
__global__ void matrixAdd(const float *A, const float *B, float *C, int width, int height) {
    // Calculate the column index for the current thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate the row index for the current thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if the calculated indices are within the bounds of the matrix
    if (col < width && row < height) {
        // Calculate the linear index for the 1D representation of the 2D matrix
        int idx = row * width + col;
        // Perform the matrix addition
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // Define the dimensions of the matrices
    int width = 1024;  // Number of columns
    int height = 1024; // Number of rows
    size_t size = width * height * sizeof(float); // Calculate the size in bytes for the matrices
    
    // Allocate memory for host matrices
    float *h_A = (float *)malloc(size); // Host matrix A
    float *h_B = (float *)malloc(size); // Host matrix B
    float *h_C = (float *)malloc(size); // Host matrix C (result)
    
    // Initialize input matrices with random values
    for (int i = 0; i < width * height; i++) {
        h_A[i] = rand() / (float)RAND_MAX; // Random value for matrix A
        h_B[i] = rand() / (float)RAND_MAX; // Random value for matrix B
    }
    
    // Device matrices
    float *d_A, *d_B, *d_C; // Device pointers for matrices A, B, and C
    
    // Allocate memory on the device for the matrices
    CUDA_CHECK(cudaMalloc(&d_A, size)); // Allocate device memory for matrix A
    CUDA_CHECK(cudaMalloc(&d_B, size)); // Allocate device memory for matrix B
    CUDA_CHECK(cudaMalloc(&d_C, size)); // Allocate device memory for matrix C
    
    // Copy host matrices to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice)); // Copy matrix A to device
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice)); // Copy matrix B to device
    
    // Define block and grid dimensions for kernel launch
    dim3 threadsPerBlock(16, 16); // Number of threads per block (16x16)
    dim3 blocksPerGrid(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x, // Calculate number of blocks in x dimension
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y  // Calculate number of blocks in y dimension
    );
    
    // Launch the matrix addition kernel
    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, width, height);
    
    // Copy the result from device to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost)); // Copy result matrix C back to host
    
    // Verify the result of the matrix addition
    bool success = true; // Flag to track verification success
    for (int i = 0; i < width * height; i++) {
        // Check if the result is correct within a tolerance
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i); // Print error message
            success = false; // Set success flag to false
            break; // Exit the loop on first failure
        }
    }
    
    // Print success message if verification passed
    if (success) {
        printf("Matrix addition completed successfully\n");
    }
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_A)); // Free device memory for matrix A
    CUDA_CHECK(cudaFree(d_B)); // Free device memory for matrix B
    CUDA_CHECK(cudaFree(d_C)); // Free device memory for matrix C
    
    // Free host memory
    free(h_A); // Free host memory for matrix A
    free(h_B); // Free host memory for matrix B
    free(h_C); // Free host memory for matrix C
    
    return 0; // Exit the program
}