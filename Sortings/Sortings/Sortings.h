#pragma once
#include <vector>

double runBitonicSortHost(int* arr, int N);
double runBitonicSortGlobal(int* h_arr, int N);
double runBitonicSortShared(int* h_arr, int N);

double runShellSortHost(int* arr, int N);
double runShellSortGlobal(int* h_arr, int N);
double runShellSortShared(int* h_arr, int N);

double runOddEvenSortHost(int* arr, int N);
double runOddEvenSortGlobal(int* h_arr, int N);
double runOddEvenSortShared(int* h_arr, int N);

double runRankingSortHost(int* arr, int N);
double runRankingSortGlobal(int* h_arr, int N);
double runRankingSortShared(int* h_arr, int N);

double runMergeSortHost(int* arr, int N);
double runMergeSortGlobal(int* h_arr, int N);
double runMergeSortShared(int* h_arr, int N);