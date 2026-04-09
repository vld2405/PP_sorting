#include "Sortings.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <utility>

#ifndef __CUDACC__
#define __syncthreads()
#endif

#define THREADS_PER_BLOCK 256

// --- HOST ---
double runOddEvenSortHost(int* arr, int N) {
    auto start = std::chrono::high_resolution_clock::now();
    bool isSorted = false;
    while (!isSorted) {
        isSorted = true;
        for (int i = 1; i < N - 1; i += 2) {
            if (arr[i] > arr[i + 1]) { std::swap(arr[i], arr[i + 1]); isSorted = false; }
        }
        for (int i = 0; i < N - 1; i += 2) {
            if (arr[i] > arr[i + 1]) { std::swap(arr[i], arr[i + 1]); isSorted = false; }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

// --- GLOBAL ---
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

double runOddEvenSortGlobal(int* h_arr, int N) {
    int* d_arr;
    size_t size = N * sizeof(int);
    cudaMalloc((void**)&d_arr, size);
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

    auto start = std::chrono::high_resolution_clock::now();
    int blocks = ((N / 2) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

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

// --- SHARED ---
__global__ void odd_even_shared(int* arr, int n) {
    extern __shared__ int s_data[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n) s_data[tid] = arr[gid];
    __syncthreads();

    for (int i = 0; i < blockDim.x / 2; i++) {
        if (tid % 2 == 0 && tid + 1 < blockDim.x && gid + 1 < n) {
            if (s_data[tid] > s_data[tid + 1]) {
                int temp = s_data[tid]; s_data[tid] = s_data[tid + 1]; s_data[tid + 1] = temp;
            }
        }
        __syncthreads();
        if (tid % 2 != 0 && tid + 1 < blockDim.x && gid + 1 < n) {
            if (s_data[tid] > s_data[tid + 1]) {
                int temp = s_data[tid]; s_data[tid] = s_data[tid + 1]; s_data[tid + 1] = temp;
            }
        }
        __syncthreads();
    }

    if (gid < n) arr[gid] = s_data[tid];
}

double runOddEvenSortShared(int* h_arr, int N) {
    int* d_arr;
    size_t size = N * sizeof(int);
    cudaMalloc((void**)&d_arr, size);
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

    int blocks_shared = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    size_t shared_mem_size = THREADS_PER_BLOCK * sizeof(int);
    int blocks_global = ((N / 2) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    auto start = std::chrono::high_resolution_clock::now();

    // 1. Sortare locală masivă în Shared Memory
    odd_even_shared << <blocks_shared, THREADS_PER_BLOCK, shared_mem_size >> > (d_arr, N);
    cudaDeviceSynchronize();

    // 2. Curățare cu Global Memory pentru elementele de la marginea blocurilor
    for (int step = 0; step < (N + 1) / 2; step++) {
        odd_even_sort_kernel << <blocks_global, THREADS_PER_BLOCK >> > (d_arr, N, 0);
        cudaDeviceSynchronize();
        odd_even_sort_kernel << <blocks_global, THREADS_PER_BLOCK >> > (d_arr, N, 1);
        cudaDeviceSynchronize();
    }

    auto end = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    return std::chrono::duration<double>(end - start).count();
}