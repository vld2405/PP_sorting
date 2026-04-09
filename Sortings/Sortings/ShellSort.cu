#include "Sortings.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

#define THREADS_PER_BLOCK 256

__global__ void shell_sort_kernel(int* arr, int n, int gap, bool* changed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i + gap < n) {
        if (arr[i] > arr[i + gap]) {
            int temp = arr[i];
            arr[i] = arr[i + gap];
            arr[i + gap] = temp;
            *changed = true;
        }
    }
}

double runShellSort(int* h_arr, int N) {
    int* d_arr;
    bool* d_changed;
    bool h_changed;

    size_t size = N * sizeof(int);
    cudaMalloc((void**)&d_arr, size);
    cudaMalloc((void**)&d_changed, sizeof(bool));
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();

    for (int gap = N / 2; gap > 0; gap /= 2) {
        int blocks = ((N - gap) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        if (blocks == 0) blocks = 1;

        do {
            h_changed = false;
            cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

            shell_sort_kernel << <blocks, THREADS_PER_BLOCK >> > (d_arr, N, gap, d_changed);
            cudaDeviceSynchronize();

            cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
        } while (h_changed);
    }

    auto end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_changed);

    std::chrono::duration<double> duration = end - start;
    return duration.count();
}