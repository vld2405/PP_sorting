#include "Sortings.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

#define THREADS_PER_BLOCK 256

__global__ void odd_even_sort_kernel(int* arr, int n, int phase) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int idx = 2 * i + phase;

    if (idx + 1 < n) {
        if (arr[idx] > arr[idx + 1]) {
            int temp = arr[idx];
            arr[idx] = arr[idx + 1];
            arr[idx + 1] = temp;
        }
    }
}

double runOddEvenSort(int* h_arr, int N) {
    int* d_arr;
    size_t size = N * sizeof(int);
    cudaMalloc((void**)&d_arr, size);
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();

    int num_threads_needed = N / 2;
    int blocks = (num_threads_needed + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    for (int step = 0; step < (N + 1) / 2; step++) {
        odd_even_sort_kernel << <blocks, THREADS_PER_BLOCK >> > (d_arr, N, 0);
        cudaDeviceSynchronize();

        odd_even_sort_kernel << <blocks, THREADS_PER_BLOCK >> > (d_arr, N, 1);
        cudaDeviceSynchronize();
    }

    auto end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);

    return std::chrono::duration<double>(end - start).count();
}