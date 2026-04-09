#include "Sortings.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <utility> // Pentru std::swap

#define THREADS_PER_BLOCK 256

// ==========================================================
// 1. VARIANTA HOST (CPU)
// ==========================================================
double runBitonicSortHost(int* arr, int N) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int k = 2; k <= N; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int i = 0; i < N; i++) {
                int ixj = i ^ j;
                if (ixj > i) {
                    if ((i & k) == 0 && arr[i] > arr[ixj]) std::swap(arr[i], arr[ixj]);
                    if ((i & k) != 0 && arr[i] < arr[ixj]) std::swap(arr[i], arr[ixj]);
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double>(end - start).count();
}

// ==========================================================
// 2. VARIANTA GLOBAL MEMORY (GPU)
// ==========================================================
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

double runBitonicSortGlobal(int* h_arr, int N) {
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

// ==========================================================
// 3. VARIANTA SHARED MEMORY (GPU Hibrid)
// ==========================================================
__global__ void bitonic_sort_shared(int* arr, int N) {
    extern __shared__ int s_data[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < N) s_data[tid] = arr[gid];
    else s_data[tid] = INT_MAX;
    __syncthreads();

    for (int k = 2; k <= blockDim.x; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int ixj = tid ^ j;
            if (ixj > tid) {
                int val_tid = s_data[tid];
                int val_ixj = s_data[ixj];

                // REZOLVARE AICI: Folosim 'gid' în loc de 'tid'
                if ((gid & k) == 0) {
                    if (val_tid > val_ixj) { s_data[tid] = val_ixj; s_data[ixj] = val_tid; }
                }
                else {
                    if (val_tid < val_ixj) { s_data[tid] = val_ixj; s_data[ixj] = val_tid; }
                }
            }
            __syncthreads();
        }
    }
    if (gid < N) arr[gid] = s_data[tid];
}

double runBitonicSortShared(int* h_arr, int N) {
    int* d_arr;
    size_t size = N * sizeof(int);
    cudaMalloc((void**)&d_arr, size);
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    size_t shared_mem_size = THREADS_PER_BLOCK * sizeof(int); // Calcul?m octe?ii necesari

    auto start = std::chrono::high_resolution_clock::now();

    // Faza 1: Sortare rapid? pe nivel de bloc în SHARED MEMORY
    bitonic_sort_shared << <blocks, THREADS_PER_BLOCK, shared_mem_size >> > (d_arr, N);
    cudaDeviceSynchronize();

    // Faza 2: Continuarea interclas?rilor mari în GLOBAL MEMORY (k > THREADS_PER_BLOCK)
    for (int k = THREADS_PER_BLOCK * 2; k <= N; k <<= 1) {
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