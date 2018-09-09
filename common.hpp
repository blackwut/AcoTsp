#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <iostream>
#include <iomanip>
#include <stdint.h>
#include <cstring>
#include <string>
#include <ctime>
#include <chrono>

using namespace std;

#define MAX_LEN 256
static uint64_t randomSeed = time(0);

enum ERROR {
    EXIT_ARGUMENTS_NUMBER = -1,
    EXIT_PARSE_ARG = -32,
    EXIT_LOAD_TSP_FILE,
    EXIT_EDGE_WEIGHT_TYPE,
    EXIT_EDGE_WEIGHT_FORMAT,
    EXIT_NODE_COORD_TYPE
};


// #define fitness(a, b)	aco->fitness[a * aco->nCities + b]
// #define p(a, b)			aco->p[a * aco->nCities + b]
// #define visited(a, b)	aco->visited[a * aco->nCities + b]
// #define tabu(a, b)		aco->tabu[a * aco->nCities + b]
// #define delta(a, b)     aco->delta[a * tsp->dimension + b]
// #define edges(a, b)		tsp->edges[a * tsp->dimension + b]

template <typename T>
inline T parseArg(char * arg) {
    clog << "Error: type not supported!" << endl;
    exit(EXIT_PARSE_ARG);
}

template<>
inline uint32_t parseArg(char * arg) {
    return atoi(arg);
}

template<>
inline float parseArg(char * arg) {
    return atof(arg);
}

template<>
inline double parseArg(char * arg) {
    return atof(arg);
}

template <typename T>
inline void printMatrix(string name, T * matrix, int rows, int cols) {

    cout << "**** " << name << " ****" << endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cout << setw(3) << setprecision(3) << fixed << matrix[i * cols + j] << " ";
        }
        cout << endl;
    }
}

template <typename T>
inline void printMatrixV(const string name,
                  const std::vector<T> & matrix,
                  const uint32_t rows,
                  const uint32_t cols,
                  const uint32_t precision)
{
    std::cout << "**** " << name << " ****" << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::setw(precision) << std::setprecision(precision) << std::fixed << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

template <typename T>
inline bool compareArray(const string & name, T * left, T * right, uint32_t elems) {
    for (int i = 0; i < elems; ++i) {
        if (left[i] != right[i]) {
            std::clog << "Error comparing array " << name << " at index ( " << i << " )" << std::endl;
            return false;
        }
    }
    return true;
}

template <typename T>
inline bool compareMatrix(const string & name, T * left, T * right, uint32_t rows, uint32_t cols) {
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            if (left[i * cols + j] != right[i * cols + j]) {
                std::clog << "Error comparing matrix " << name << " at index ( " << i << ", " << j << " )" << std::endl;
                return false;
            }
        }
    }
    return true;
}

static std::chrono::high_resolution_clock::time_point startPoint;
static std::chrono::high_resolution_clock::time_point endPoint;

inline void startTimer() {
    startPoint = std::chrono::high_resolution_clock::now();
}

inline void stopTimer() {
    endPoint = std::chrono::high_resolution_clock::now();
}

inline long getTimerMS() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(endPoint - startPoint).count();
}

inline long getTimerUS() {
    return std::chrono::duration_cast<std::chrono::microseconds>(endPoint - startPoint).count();
}

inline void printTimer() {
    long msec = getTimerMS();
    long usec = getTimerUS();
    cout << "Compute time: " << msec << " ms " << usec << " usec " << endl;
}

inline void stopAndPrintTimer() {
    stopTimer();
    printTimer();
}

inline void stopAndPrintTimer(string name) {
    stopTimer();
    cout << name;
    printTimer();
}

#endif
