#pragma once

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <chrono>
#include <fstream>
#include <iostream>

#pragma region Metrics
// --------------------
// 1. Speedup
// S = T1 / Tp
// --------------------
double speedup(double serialTime, double parallelTime)
{
    if (parallelTime <= 0.0) return 0.0;
    return serialTime / parallelTime;
}

// --------------------
// 2. Efficiency
// E = S / P
// --------------------
double efficiency(double serialTime, double parallelTime, int processors)
{
    if (processors <= 0) return 0.0;
    return speedup(serialTime, parallelTime) / processors;
}

// --------------------
// 3. Parallel Overhead
// Overhead = P * Tp - T1
// --------------------
double overhead(double serialTime, double parallelTime, int processors)
{
    if (processors <= 0) return 0.0;
    return processors * parallelTime - serialTime;
}

// --------------------
// 4. Cost
// Cost = P * Tp
// --------------------
double cost(double parallelTime, int processors)
{
    if (processors <= 0) return 0.0;
    return processors * parallelTime;
}

// --------------------
// 5. Amdahl's Law
// S(P) = 1 / [(1 - f) + f / P]
// f = parallel fraction
// --------------------
double amdahlSpeedup(double parallelFraction, int processors)
{
    if (processors <= 0) return 0.0;
    if (parallelFraction < 0.0 || parallelFraction > 1.0) return 0.0;

    return 1.0 / ((1.0 - parallelFraction) +
        (parallelFraction / processors));
}

// --------------------
// 6. Gustafson's Law
// S(P) = P - (P - 1)(1 - f)
// --------------------
double gustafsonSpeedup(double parallelFraction, int processors)
{
    if (processors <= 0) return 0.0;
    if (parallelFraction < 0.0 || parallelFraction > 1.0) return 0.0;

    return processors - (processors - 1) * (1.0 - parallelFraction);
}

// --------------------
// 7. Strong Scaling Efficiency
// --------------------
double strongScalingEfficiency(double serialTime,
    double parallelTime,
    int processors)
{
    return efficiency(serialTime, parallelTime, processors);
}

// --------------------
// 8. Weak Scaling Efficiency
// E_weak = T1 / Tp
// --------------------
double weakScalingEfficiency(double timeOneProcessor,
    double timeManyProcessors)
{
    if (timeManyProcessors <= 0.0) return 0.0;
    return timeOneProcessor / timeManyProcessors;
}

// --------------------
// 9. Load Imbalance
// Imbalance = Tmax / Tavg
// --------------------
double loadImbalance(double maxTime, double avgTime)
{
    if (avgTime <= 0.0) return 0.0;
    return maxTime / avgTime;
}

// --------------------
// 10. Communication Time
// T = alpha + beta * n
// --------------------
double communicationTime(double latency,
    double bandwidthCost,
    double messageSize)
{
    return latency + bandwidthCost * messageSize;
}

// --------------------
// 11. Throughput
// --------------------
double throughput(int tasks, double time)
{
    if (time <= 0.0) return 0.0;
    return static_cast<double>(tasks) / time;
}
#pragma endregion

#pragma region GPUProperties

void getDeviceProperties()
{
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "GPU name: " << prop.name << "\n";
    std::cout << "Compute capability: "
        << prop.major << "." << prop.minor << "\n";

    std::cout << "Multiprocessors (SMs): "
        << prop.multiProcessorCount << "\n";

    std::cout << "Max threads per block: "
        << prop.maxThreadsPerBlock << "\n";

    std::cout << "Max threads per SM: "
        << prop.maxThreadsPerMultiProcessor << "\n";

    std::cout << "Max blocks per SM: "
        << prop.maxBlocksPerMultiProcessor << "\n";

    std::cout << "Warp size: "
        << prop.warpSize << "\n";

    std::cout << "Max block dimensions: "
        << prop.maxThreadsDim[0] << ", "
        << prop.maxThreadsDim[1] << ", "
        << prop.maxThreadsDim[2] << "\n";

    std::cout << "Max grid dimensions: "
        << prop.maxGridSize[0] << ", "
        << prop.maxGridSize[1] << ", "
        << prop.maxGridSize[2] << "\n";

    std::cout << "Shared mem per block (bytes): "
        << prop.sharedMemPerBlock << "\n";

    std::cout << "Shared mem per SM (bytes): "
        << prop.sharedMemPerMultiprocessor << "\n";

    std::cout << "Registers per block: "
        << prop.regsPerBlock << "\n";

    std::cout << "Registers per SM: "
        << prop.regsPerMultiprocessor << "\n";

    std::cout << "Global memory (bytes): "
        << prop.totalGlobalMem << "\n";

    std::cout << "L2 cache size (bytes): "
        << prop.l2CacheSize << "\n";

    std::cout << "Memory bus width (bits): "
        << prop.memoryBusWidth << "\n";

    std::cout << "Concurrent kernels: "
        << prop.concurrentKernels << "\n";

    std::cout << "Async engine count: "
        << prop.asyncEngineCount << "\n";

    std::cout << "Unified addressing: "
        << prop.unifiedAddressing << "\n";

    std::cout << "Max warps per SM: "
        << prop.maxThreadsPerMultiProcessor / prop.warpSize << "\n";

    std::cout << "Warp allocation granularity: "
        << prop.warpSize << "\n";

    std::cout << "ECC enabled: "
        << prop.ECCEnabled << "\n";
    /*
    * Error-Correcting Code (ECC).
    * ECC este o tehnologie de corectare a erorilor pentru memoria GPU-ului.
    */

    std::cout << "Managed memory: "
        << prop.managedMemory << "\n";

    std::cout << "Cooperative launch: "
        << prop.cooperativeLaunch << "\n";
}

#pragma endregion

#pragma region ReadData

std::string read_text_from_file_clasic(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Error opening file!" << std::endl;
        return "";
    }
    std::string data;
    file >> data;
    std::cout << "File size is " << data.size() << "\n";
    return data;
}

std::string read_text_from_file_fast(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate); // Open at end to get size
    if (!file) {
        std::cerr << "Error opening file!" << std::endl;
        return "";
    }
    std::streamsize size = file.tellg(); // Returns the position of the current character
    std::cout << "File size is " << size << "\n";
    file.seekg(0, std::ios::beg); // Go back to the beginning
    std::string data(size, '\0');
    file.read(&data[0], size);
    return data;
}

#pragma endregion
