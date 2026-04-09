#include "Sortings.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

#define THREADS_PER_BLOCK 256

__global__ void bitonic_sort_kernel(int* dev_values, int j, int k, int N) {
    unsigned int i, ixj;
    i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i >= N) return;

    ixj = i ^ j;

    if ((ixj > i)) {
        if ((i & k) == 0) {
            if (dev_values[i] > dev_values[ixj]) {
                int temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
        if ((i & k) != 0) {
            if (dev_values[i] < dev_values[ixj]) {
                int temp = dev_values[i];
                dev_values[i] = dev_values[ixj];
                dev_values[ixj] = temp;
            }
        }
    }
}

double runBitonicSort(int* h_arr, int N) {
    int* d_arr;
    size_t size = N * sizeof(int);
    cudaMalloc((void**)&d_arr, size);
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    auto start = std::chrono::high_resolution_clock::now();

    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j = j >> 1) {
            bitonic_sort_kernel << <blocks, THREADS_PER_BLOCK >> > (d_arr, j, k, N);
            cudaDeviceSynchronize();
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);

    std::chrono::duration<double> duration = end - start;
    return duration.count();
}