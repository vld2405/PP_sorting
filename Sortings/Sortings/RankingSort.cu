#include "Sortings.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <vector>

#ifndef __CUDACC__
#define __syncthreads()
#endif

#define THREADS_PER_BLOCK 256

// --- HOST ---
double runRankingSortHost(int* arr, int N) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> out_arr(N);

    for (int i = 0; i < N; i++) {
        int rank = 0;
        for (int j = 0; j < N; j++) {
            if (arr[j] < arr[i] || (arr[j] == arr[i] && j < i)) rank++;
        }
        out_arr[rank] = arr[i];
    }

    for (int i = 0; i < N; i++) arr[i] = out_arr[i];

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

// --- GLOBAL ---
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

double runRankingSortGlobal(int* h_arr, int N) {
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

// --- SHARED ---
__global__ void ranking_sort_shared(int* in_arr, int* out_arr, int n) {
    extern __shared__ int s_tile[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    int my_val = (gid < n) ? in_arr[gid] : INT_MAX;
    int final_position = 0;

    int num_tiles = (n + blockDim.x - 1) / blockDim.x;

    for (int t = 0; t < num_tiles; t++) {
        int tile_index = t * blockDim.x + tid;
        if (tile_index < n) s_tile[tid] = in_arr[tile_index];
        else s_tile[tid] = INT_MAX;

        __syncthreads();

        if (gid < n) {
            for (int j = 0; j < blockDim.x; j++) {
                int actual_j = t * blockDim.x + j;
                if (actual_j < n) {
                    if (s_tile[j] < my_val || (s_tile[j] == my_val && actual_j < gid)) {
                        final_position++;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (gid < n) out_arr[final_position] = my_val;
}

double runRankingSortShared(int* h_arr, int N) {
    int* d_in, * d_out;
    size_t size = N * sizeof(int);
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);
    cudaMemcpy(d_in, h_arr, size, cudaMemcpyHostToDevice);

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    size_t shared_mem_size = THREADS_PER_BLOCK * sizeof(int);

    auto start = std::chrono::high_resolution_clock::now();

    ranking_sort_shared << <blocks, THREADS_PER_BLOCK, shared_mem_size >> > (d_in, d_out, N);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_arr, d_out, size, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    return std::chrono::duration<double>(end - start).count();
}