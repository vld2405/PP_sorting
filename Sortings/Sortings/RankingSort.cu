#include "Sortings.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

#define THREADS_PER_BLOCK 256


__global__ void ranking_sort_kernel(int* in_arr, int* out_arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        int my_val = in_arr[i];
        int final_position = 0;

        for (int j = 0; j < n; j++) {
            if (in_arr[j] < my_val || (in_arr[j] == my_val && j < i)) {
                final_position++;
            }
        }

        out_arr[final_position] = my_val;
    }
}

double runRankingSort(int* h_arr, int N) {
    int* d_in, * d_out;
    size_t size = N * sizeof(int);
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    cudaMemcpy(d_in, h_arr, size, cudaMemcpyHostToDevice);

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    auto start = std::chrono::high_resolution_clock::now();

    ranking_sort_kernel << <blocks, THREADS_PER_BLOCK >> > (d_in, d_out, N);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_arr, d_out, size, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);

    return std::chrono::duration<double>(end - start).count();
}