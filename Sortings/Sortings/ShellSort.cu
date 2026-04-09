#include "Sortings.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <utility>

#ifndef __CUDACC__
#define __syncthreads()
#endif

#define THREADS_PER_BLOCK 256

// ==========================================================
// --- 1. HOST ---
// ==========================================================
double runShellSortHost(int* arr, int N) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int gap = N / 2; gap > 0; gap /= 2) {
        for (int i = gap; i < N; i++) {
            int temp = arr[i];
            int j;
            for (j = i; j >= gap && arr[j - gap] > temp; j -= gap) {
                arr[j] = arr[j - gap];
            }
            arr[j] = temp;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}


// ==========================================================
// --- 2. GLOBAL ---
// ==========================================================
__global__ void shell_sort_kernel(int* arr, int n, int gap, int phase, bool* changed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Faza Par? / Impar? pentru a preveni coliziunea datelor (Race Conditions)
    if (i + gap < n && (i / gap) % 2 == phase) {
        if (arr[i] > arr[i + gap]) {
            int temp = arr[i];
            arr[i] = arr[i + gap];
            arr[i + gap] = temp;
            *changed = true;
        }
    }
}

double runShellSortGlobal(int* h_arr, int N) {
    int* d_arr;
    bool* d_changed, h_changed;
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

            shell_sort_kernel << <blocks, THREADS_PER_BLOCK >> > (d_arr, N, gap, 0, d_changed);
            cudaDeviceSynchronize();
            shell_sort_kernel << <blocks, THREADS_PER_BLOCK >> > (d_arr, N, gap, 1, d_changed);
            cudaDeviceSynchronize();

            cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
        } while (h_changed);
    }

    auto end = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_changed);
    return std::chrono::duration<double>(end - start).count();
}


// ==========================================================
// --- 3. SHARED ---
// ==========================================================
__global__ void shell_sort_shared(int* arr, int n) {
    extern __shared__ int s_arr[];
    __shared__ bool s_changed; // Variabil? sigur? partajat? pentru a verifica dac? s-a f?cut vreun swap

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n) s_arr[tid] = arr[gid];
    else s_arr[tid] = INT_MAX;
    __syncthreads();

    // Sorteaz? blocul local ca ?i cum ar fi propriul s?u array
    for (int gap = blockDim.x / 2; gap > 0; gap /= 2) {
        do {
            __syncthreads();
            if (tid == 0) s_changed = false;
            __syncthreads();

            // Faza Par? local?
            if (tid + gap < blockDim.x && (tid / gap) % 2 == 0) {
                if (s_arr[tid] > s_arr[tid + gap]) {
                    int temp = s_arr[tid]; s_arr[tid] = s_arr[tid + gap]; s_arr[tid + gap] = temp;
                    s_changed = true;
                }
            }
            __syncthreads();

            // Faza Impar? local?
            if (tid + gap < blockDim.x && (tid / gap) % 2 != 0) {
                if (s_arr[tid] > s_arr[tid + gap]) {
                    int temp = s_arr[tid]; s_arr[tid] = s_arr[tid + gap]; s_arr[tid + gap] = temp;
                    s_changed = true;
                }
            }
            __syncthreads();

        } while (s_changed);
    }

    if (gid < n) arr[gid] = s_arr[tid];
}

double runShellSortShared(int* h_arr, int N) {
    int* d_arr;
    bool* d_changed, h_changed;
    size_t size = N * sizeof(int);
    cudaMalloc((void**)&d_arr, size);
    cudaMalloc((void**)&d_changed, sizeof(bool));
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    size_t shared_mem_size = THREADS_PER_BLOCK * sizeof(int);

    auto start = std::chrono::high_resolution_clock::now();

    // PASUL 1: Pre-sortare ultra-rapid? local? în Shared Memory
    shell_sort_shared << <blocks, THREADS_PER_BLOCK, shared_mem_size >> > (d_arr, N);
    cudaDeviceSynchronize();

    // PASUL 2: Ciclul Global clasic pentru a rezolva grani?ele dintre blocuri
    // (Va rula foarte rapid deoarece vectorul e deja 99% sortat de pasul anterior)
    for (int gap = N / 2; gap > 0; gap /= 2) {
        int current_blocks = ((N - gap) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        if (current_blocks == 0) current_blocks = 1;
        do {
            h_changed = false;
            cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

            shell_sort_kernel << <current_blocks, THREADS_PER_BLOCK >> > (d_arr, N, gap, 0, d_changed);
            cudaDeviceSynchronize();
            shell_sort_kernel << <current_blocks, THREADS_PER_BLOCK >> > (d_arr, N, gap, 1, d_changed);
            cudaDeviceSynchronize();

            cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
        } while (h_changed);
    }

    auto end = std::chrono::high_resolution_clock::now();
    cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_changed);
    return std::chrono::duration<double>(end - start).count();
}