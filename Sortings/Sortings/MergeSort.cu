#include "Sortings.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>

#define THREADS_PER_BLOCK 256


__global__ void merge_sort_kernel(int* arr, int* temp, int n, int width) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2 * width;

    if (i >= n) return;

    int left = i;
    int middle = i + width;
    int right = i + 2 * width;

    if (middle > n) middle = n;
    if (right > n) right = n;

    int l = left;
    int r = middle;
    int idx = left;

    while (l < middle && r < right) {
        if (arr[l] <= arr[r]) {
            temp[idx++] = arr[l++];
        }
        else {
            temp[idx++] = arr[r++];
        }
    }

    while (l < middle) temp[idx++] = arr[l++];
    while (r < right)  temp[idx++] = arr[r++];

    for (int k = left; k < right; k++) {
        arr[k] = temp[k];
    }
}

double runMergeSort(int* h_arr, int N) {
    int* d_arr, * d_temp;
    size_t size = N * sizeof(int);
    cudaMalloc((void**)&d_arr, size);
    cudaMalloc((void**)&d_temp, size);

    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();

    for (int width = 1; width < N; width *= 2) {
        int num_merges = (N + (2 * width) - 1) / (2 * width);
        int blocks = (num_merges + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        if (blocks == 0) blocks = 1;

        merge_sort_kernel << <blocks, THREADS_PER_BLOCK >> > (d_arr, d_temp, N, width);
        cudaDeviceSynchronize();
    }

    auto end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_temp);

    return std::chrono::duration<double>(end - start).count();
}