#include "Sortings.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <vector>
#include <algorithm>

#ifndef __CUDACC__
#define __syncthreads()
#endif

#define THREADS_PER_BLOCK 256


double runMergeSortHost(int* arr, int N) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> temp(N);

    for (int width = 1; width < N; width *= 2) {
        for (int i = 0; i < N; i += 2 * width) {
            int left = i;
            int mid = std::min(i + width, N);
            int right = std::min(i + 2 * width, N);

            int l = left, r = mid, idx = left;
            while (l < mid && r < right) {
                temp[idx++] = (arr[l] <= arr[r]) ? arr[l++] : arr[r++];
            }
            while (l < mid) temp[idx++] = arr[l++];
            while (r < right) temp[idx++] = arr[r++];
            for (int k = left; k < right; k++) arr[k] = temp[k];
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}


__global__ void merge_sort_kernel(int* arr, int* temp, int n, int width) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2 * width;
    if (i >= n) return;

    int left = i;
    int middle = i + width;
    int right = i + 2 * width;

    if (middle > n) middle = n;
    if (right > n) right = n;

    int l = left, r = middle, idx = left;

    while (l < middle && r < right) {
        if (arr[l] <= arr[r]) temp[idx++] = arr[l++];
        else temp[idx++] = arr[r++];
    }

    while (l < middle) temp[idx++] = arr[l++];
    while (r < right)  temp[idx++] = arr[r++];

    for (int k = left; k < right; k++) arr[k] = temp[k];
}


double runMergeSortGlobal(int* h_arr, int N) {
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


__global__ void merge_sort_shared(int* arr, int N) {
    extern __shared__ int s_arr[];
    __shared__ int s_temp[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < N) s_arr[tid] = arr[gid];
    else s_arr[tid] = INT_MAX;
    __syncthreads();

    for (int width = 1; width < blockDim.x; width *= 2) {
        s_temp[tid] = s_arr[tid];
        __syncthreads();

        if (tid % (2 * width) == 0) {
            int left = tid;
            int mid = min(tid + width, blockDim.x);
            int right = min(tid + 2 * width, blockDim.x);

            int l = left, r = mid, idx = left;
            while (l < mid && r < right) {
                if (s_temp[l] <= s_temp[r]) s_arr[idx++] = s_temp[l++];
                else                         s_arr[idx++] = s_temp[r++];
            }
            while (l < mid)   s_arr[idx++] = s_temp[l++];
            while (r < right) s_arr[idx++] = s_temp[r++];
        }
        __syncthreads();
    }

    if (gid < N) arr[gid] = s_arr[tid];
}


double runMergeSortShared(int* h_arr, int N) {
    int* d_arr, * d_temp;
    size_t size = N * sizeof(int);
    cudaMalloc((void**)&d_arr, size);
    cudaMalloc((void**)&d_temp, size);
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    size_t shared_mem_size = THREADS_PER_BLOCK * sizeof(int);

    auto start = std::chrono::high_resolution_clock::now();

    merge_sort_shared << <blocks, THREADS_PER_BLOCK, shared_mem_size >> > (d_arr, N);
    cudaDeviceSynchronize();

    for (int width = THREADS_PER_BLOCK; width < N; width *= 2) {
        int num_merges = (N + (2 * width) - 1) / (2 * width);
        int current_blocks = (num_merges + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        if (current_blocks == 0) current_blocks = 1;

        merge_sort_kernel << <current_blocks, THREADS_PER_BLOCK >> > (d_arr, d_temp, N, width);
        cudaDeviceSynchronize();
    }

    auto end = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_temp);
    return std::chrono::duration<double>(end - start).count();
}