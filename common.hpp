#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <iostream>
#include <iomanip>
#include <stdint.h>
#include <cstring>
#include <string>
#include <ctime>
#include <chrono>
#include <vector>

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
inline void printMatrix(string name, T * matrix, uint32_t rows, uint32_t cols) {

    cout << "**** " << name << " ****" << endl;
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
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
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            std::cout << std::setw(precision) << std::setprecision(precision) << std::fixed << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

template <typename T>
inline bool compareArray(const string & name, T * left, T * right, uint32_t elems) {
    for (uint32_t i = 0; i < elems; ++i) {
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

inline void printResult(const string & name,
                        const uint32_t mapWorkers,
                        const uint32_t farmWorkers,
                        const uint32_t maxEpoch,
                        const long     timeMS,
                        const long     timeUS,
                        const float    bestTourLength,
                        const bool     checkPath)
{
#define LOG_SEP " "
    std::clog << " *** "       << LOG_SEP
    << name                    << LOG_SEP
    << mapWorkers              << LOG_SEP
    << farmWorkers             << LOG_SEP
    << maxEpoch                << LOG_SEP
    << timeMS                  << LOG_SEP
    << timeUS                  << LOG_SEP
    << bestTourLength          << LOG_SEP
    << (checkPath ? "Y" : "N") << LOG_SEP
    << std::endl;
}

#endif
