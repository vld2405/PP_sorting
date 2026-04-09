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
    const long long N = 1 << 20;

    std::cout << "Se genereaza datele (" << N << " elemente)..." << std::endl;
    std::vector<int> data(N);

    std::mt19937 rng(1337);
    std::uniform_int_distribution<int> dist(0, 10000000);
    for (int i = 0; i < N; ++i) {
        data[i] = dist(rng);
    }

    std::vector<int> data_std = data;
    std::vector<int> data_bitonic = data;
    std::vector<int> data_shell = data;
    std::vector<int> data_oddEven = data;
    std::vector<int> data_ranking = data;
    std::vector<int> data_merge = data;

    // 1. Serial (std::sort)
    std::cout << "\n--- std::sort (CPU) ---" << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    std::sort(data_std.begin(), data_std.end());
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double>(end_cpu - start_cpu).count();
    std::cout << "Time: " << cpu_time << " s\n";

    // 2. Bitonic Sort (CUDA)
    std::cout << "\n--- Bitonic Sort (CUDA) ---" << std::endl;
    double bitonic_time = runBitonicSort(data_bitonic.data(), N);
    std::cout << "Time: " << bitonic_time << " s\n";
    std::cout << "Validare: " << (isSorted(data_bitonic) ? "SUCCES" : "ESEC") << "\n";

    // 3. Shell Sort (CUDA)
    std::cout << "\n--- Shell Sort (CUDA) ---" << std::endl;
    double shell_time = runShellSort(data_shell.data(), N);
    std::cout << "Time: " << shell_time << " s\n";
    std::cout << "Validare: " << (isSorted(data_shell) ? "SUCCES" : "ESEC") << "\n";

    // 4. Odd-Even Sort (CUDA)
    std::cout << "\n--- Odd-Even Sort (CUDA) ---" << std::endl;
    double oddEven_time = runOddEvenSort(data_oddEven.data(), N);
    std::cout << "Time: " << oddEven_time << " s\n";
    std::cout << "Validare: " << (isSorted(data_oddEven) ? "SUCCES" : "ESEC") << "\n";

    // 5. Ranking Sort (CUDA)
    std::cout << "\n--- Ranking Sort (CUDA) ---" << std::endl;
    double ranking_time = runRankingSort(data_ranking.data(), N);
    std::cout << "Time: " << ranking_time << " s\n";
    std::cout << "Validare: " << (isSorted(data_ranking) ? "SUCCES" : "ESEC") << "\n";

    // 6. Merge Sort (CUDA)
    std::cout << "\n--- Merge Sort (CUDA) ---" << std::endl;
    double merge_time = runMergeSort(data_merge.data(), N);
    std::cout << "Time: " << merge_time << " s\n";
    std::cout << "Validare: " << (isSorted(data_merge) ? "SUCCES" : "ESEC") << "\n";

    std::cout << "\n================ PERFORMANCE METRICS ================\n";
    int num_processors = 4;

    std::cout << "[Bitonic Sort vs Serial]\n";
    std::cout << "Speedup:    " << speedup(cpu_time, bitonic_time) << "x\n";
    std::cout << "Efficiency: " << efficiency(cpu_time, bitonic_time, num_processors) * 100 << "%\n";

    std::cout << "\n[Shell Sort vs Serial]\n";
    std::cout << "Speedup:    " << speedup(cpu_time, shell_time) << "x\n";
    std::cout << "Efficiency: " << efficiency(cpu_time, shell_time, num_processors) * 100 << "%\n";

    std::cout << "\n[Odd-Even Sort vs Serial]\n";
    std::cout << "Speedup:    " << speedup(cpu_time, oddEven_time) << "x\n";
    std::cout << "Efficiency: " << efficiency(cpu_time, oddEven_time, num_processors) * 100 << "%\n";

    std::cout << "\n[Ranking Sort vs Serial]\n";
    std::cout << "Speedup:    " << speedup(cpu_time, ranking_time) << "x\n";
    std::cout << "Efficiency: " << efficiency(cpu_time, ranking_time, num_processors) * 100 << "%\n";

    std::cout << "\n[Merge Sort vs Serial]\n";
    std::cout << "Speedup:    " << speedup(cpu_time, merge_time) << "x\n";
    std::cout << "Efficiency: " << efficiency(cpu_time, merge_time, num_processors) * 100 << "%\n";
    std::cout << "=====================================================\n";

    std::cout << "\n--- DEVICE PROPERTIES ---" << std::endl;
    getDeviceProperties();

    return 0;
}