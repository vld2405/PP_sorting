#include "Utils.h"
#include "Sortings.h"
#include <random>

bool isSorted(const std::vector<int>& arr) {
    for (size_t i = 1; i < arr.size(); ++i) {
        if (arr[i - 1] > arr[i]) return false;
    }
    return true;
}

int main()
{
    const long long N = 1 << 10;

    std::cout << "Se genereaza datele (" << N << " elemente)..." << std::endl;
    std::vector<int> data(N);

    std::mt19937 rng(1337);
    std::uniform_int_distribution<int> dist(0, 10000000);
    for (int i = 0; i < N; ++i) {
        data[i] = dist(rng);
    }

    std::vector<int> data_bitonic_host = data;
    std::vector<int> data_bitonic_global = data;
    std::vector<int> data_bitonic_shared = data;
    std::vector<int> data_shell_host = data;
    std::vector<int> data_shell_global = data;
    std::vector<int> data_shell_shared = data;
    std::vector<int> data_oddEven_host = data;
    std::vector<int> data_oddEven_global = data;
    std::vector<int> data_oddEven_shared = data;
    std::vector<int> data_ranking_host = data;
    std::vector<int> data_ranking_global = data;
    std::vector<int> data_ranking_shared = data;
    std::vector<int> data_merge_host = data;
    std::vector<int> data_merge_global = data;
    std::vector<int> data_merge_shared = data;

#pragma region Bitonic Sort

    std::cout << "-------------------------------------------------------\n";
    std::cout << "--- Bitonic Sort (Host/CPU Custom) ---" << std::endl;
    double bitonic_host_time = runBitonicSortHost(data_bitonic_host.data(), N);
    std::cout << "Time: " << bitonic_host_time << " s | Valid: " << (isSorted(data_bitonic_host) ? "SUCCESS" : "FAIL") << "\n";

    std::cout << "\n--- Bitonic Sort (GPU Global Memory) ---" << std::endl;
    double bitonic_global_time = runBitonicSortGlobal(data_bitonic_global.data(), N);
    std::cout << "Time: " << bitonic_global_time << " s | Valid: " << (isSorted(data_bitonic_global) ? "SUCCESS" : "FAIL") << "\n";

    std::cout << "\n--- Bitonic Sort (GPU Shared Memory) ---" << std::endl;
    double bitonic_shared_time = runBitonicSortShared(data_bitonic_shared.data(), N);
    std::cout << "Time: " << bitonic_shared_time << " s | Valid: " << (isSorted(data_bitonic_shared) ? "SUCCESS" : "FAIL") << "\n";

#pragma endregion

    std::cout << "-------------------------------------------------------\n";

#pragma region Shell Sort

    std::cout << "--- Shell Sort (Host/CPU Custom) ---" << std::endl;
    double shell_host_time = runShellSortHost(data_shell_host.data(), N);
    std::cout << "Time: " << shell_host_time << " s | Valid: " << (isSorted(data_shell_host) ? "SUCCESS" : "FAIL") << "\n";

    std::cout << "\n--- Shell Sort (GPU Global Memory) ---" << std::endl;
    double shell_global_time = runShellSortGlobal(data_shell_global.data(), N);
    std::cout << "Time: " << shell_global_time << " s | Valid: " << (isSorted(data_shell_global) ? "SUCCESS" : "FAIL") << "\n";

    std::cout << "\n--- Shell Sort (GPU Shared Memory) ---" << std::endl;
    double shell_shared_time = runShellSortShared(data_shell_shared.data(), N);
    std::cout << "Time: " << shell_shared_time << " s | Valid: " << (isSorted(data_shell_shared) ? "SUCCESS" : "FAIL") << "\n";

#pragma endregion

    std::cout << "-------------------------------------------------------\n";

#pragma region Odd-Even Sort

    std::cout << "--- Odd-Even Sort (Host/CPU Custom) ---" << std::endl;
    double oddEven_host_time = runOddEvenSortHost(data_oddEven_host.data(), N);
    std::cout << "Time: " << oddEven_host_time << " s | Valid: " << (isSorted(data_oddEven_host) ? "SUCCESS" : "FAIL") << "\n";

    std::cout << "\n--- Odd-Even Sort (GPU Global Memory) ---" << std::endl;
    double oddEven_global_time = runOddEvenSortGlobal(data_oddEven_global.data(), N);
    std::cout << "Time: " << oddEven_global_time << " s | Valid: " << (isSorted(data_oddEven_global) ? "SUCCESS" : "FAIL") << "\n";

    std::cout << "\n--- Odd-Even Sort (GPU Shared Memory) ---" << std::endl;
    double oddEven_shared_time = runOddEvenSortShared(data_oddEven_shared.data(), N);
    std::cout << "Time: " << oddEven_shared_time << " s | Valid: " << (isSorted(data_oddEven_shared) ? "SUCCESS" : "FAIL") << "\n";

#pragma endregion

    std::cout << "-------------------------------------------------------\n";

#pragma region Ranking Sort

    std::cout << "--- Ranking Sort (Host/CPU Custom) ---" << std::endl;
    double ranking_host_time = runRankingSortHost(data_ranking_host.data(), N);
    std::cout << "Time: " << ranking_host_time << " s | Valid: " << (isSorted(data_ranking_host) ? "SUCCESS" : "FAIL") << "\n";

    std::cout << "\n--- Ranking Sort (GPU Global Memory) ---" << std::endl;
    double ranking_global_time = runRankingSortGlobal(data_ranking_global.data(), N);
    std::cout << "Time: " << ranking_global_time << " s | Valid: " << (isSorted(data_ranking_global) ? "SUCCESS" : "FAIL") << "\n";

    std::cout << "\n--- Ranking Sort (GPU Shared Memory) ---" << std::endl;
    double ranking_shared_time = runRankingSortShared(data_ranking_shared.data(), N);
    std::cout << "Time: " << ranking_shared_time << " s | Valid: " << (isSorted(data_ranking_shared) ? "SUCCESS" : "FAIL") << "\n";

#pragma endregion

    std::cout << "-------------------------------------------------------\n";

#pragma region Merge Sort

    std::cout << "--- Merge Sort (Host/CPU Custom) ---" << std::endl;
    double merge_host_time = runMergeSortHost(data_merge_host.data(), N);
    std::cout << "Time: " << merge_host_time << " s | Valid: " << (isSorted(data_merge_host) ? "SUCCESS" : "FAIL") << "\n";

    std::cout << "\n--- Merge Sort (GPU Global Memory) ---" << std::endl;
    double merge_global_time = runMergeSortGlobal(data_merge_global.data(), N);
    std::cout << "Time: " << merge_global_time << " s | Valid: " << (isSorted(data_merge_global) ? "SUCCESS" : "FAIL") << "\n";

    std::cout << "\n--- Merge Sort (GPU Shared Memory) ---" << std::endl;
    double merge_shared_time = runMergeSortShared(data_merge_shared.data(), N);
    std::cout << "Time: " << merge_shared_time << " s | Valid: " << (isSorted(data_merge_shared) ? "SUCCESS" : "FAIL") << "\n";

#pragma endregion

    std::cout << "\n\n================ PERFORMANCE METRICS ================\n";
    int num_processors = 2560;

    std::cout << "[Bitonic Sort: GPU Shared vs CPU Host]\n";
    std::cout << "Speedup:    " << speedup(bitonic_host_time, bitonic_shared_time) << "x\n";
    std::cout << "Efficiency: " << efficiency(bitonic_host_time, bitonic_shared_time, num_processors) * 100 << "%\n";
    std::cout << "GPU Global vs CPU Host Speedup: " << speedup(bitonic_host_time, bitonic_global_time) << "x\n";
    
    std::cout << "-------------------------------------------------------\n";

    std::cout << "[Shell Sort: GPU Shared vs CPU Host]\n";
    std::cout << "Speedup:    " << speedup(shell_host_time, shell_shared_time) << "x\n";
    std::cout << "Efficiency: " << efficiency(shell_host_time, shell_shared_time, num_processors) * 100 << "%\n";
    std::cout << "GPU Global vs CPU Host Speedup: " << speedup(shell_host_time, shell_global_time) << "x\n";
    
    std::cout << "-------------------------------------------------------\n";

    std::cout << "[Odd-Even Sort: GPU Shared vs CPU Host]\n";
    std::cout << "Speedup:    " << speedup(oddEven_host_time, oddEven_shared_time) << "x\n";
    std::cout << "Efficiency: " << efficiency(oddEven_host_time, oddEven_shared_time, num_processors) * 100 << "%\n";
    std::cout << "GPU Global vs CPU Host Speedup: " << speedup(oddEven_host_time, oddEven_global_time) << "x\n";
    
    std::cout << "-------------------------------------------------------\n";

    std::cout << "[Merge Sort: GPU Shared vs CPU Host]\n";
    std::cout << "Speedup:    " << speedup(merge_host_time, merge_shared_time) << "x\n";
    std::cout << "Efficiency: " << efficiency(merge_host_time, merge_shared_time, num_processors) * 100 << "%\n";
    std::cout << "GPU Global vs CPU Host Speedup: " << speedup(merge_host_time, merge_global_time) << "x\n";
    
    std::cout << "-------------------------------------------------------\n";

    std::cout << "[Ranking Sort: GPU Shared vs CPU Host (pe 10k elem)]\n";
    std::cout << "Speedup:    " << speedup(ranking_host_time, ranking_shared_time) << "x\n";
    std::cout << "Efficiency: " << efficiency(ranking_host_time, ranking_shared_time, num_processors) * 100 << "%\n";
    std::cout << "GPU Global vs CPU Host Speedup: " << speedup(ranking_host_time, ranking_global_time) << "x\n";
    std::cout << "=======================================================\n";

    std::cout << "\n--- DEVICE PROPERTIES ---" << std::endl;
    getDeviceProperties();

    return 0;
}