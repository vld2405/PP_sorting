#include "Utils.h"

void heavyTask(int N)
{
    for (volatile int i = 0; i < N; ++i);
}

int main()
{
    const int N = 1'000'000'000;

    //goto properties;

    auto start = std::chrono::high_resolution_clock::now();
    heavyTask(N);
    //read_text_from_file_clasic("sequence1.txt");
    //read_text_from_file_fast("sequence1.txt");
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " s\n";

performance_metrics:
    // We assume we get these times. How do we evaluate them?
    double T1 = 4.0;   // serial time (s)
    double Tp = 1.2;   // parallel time (s)
    int P = 4;         // number of procs

    std::cout << "Speedup:     " << speedup(T1, Tp) << "x\n";
    std::cout << "Efficiency:  " << efficiency(T1, Tp, P) * 100 << "%\n";
    std::cout << "Overhead:    " << overhead(T1, Tp, P) << " s\n";
    std::cout << "Cost:        " << cost(Tp, P) << " s\n";

    std::cout << "Amdahl:      " << amdahlSpeedup(0.95, P) << "x\n";
    std::cout << "Gustafson:   " << gustafsonSpeedup(0.95, P) << "x\n";
    std::cout << "----------------------------------------------------\n";

properties:
    getDeviceProperties();

    return 0;
}