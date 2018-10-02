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

#define MAX_LEN 256
static uint64_t randomSeed = time(0);

enum ERROR {
    EXIT_ARGUMENTS_NUMBER = -1,
    EXIT_PARSE_ARG = -32,
    EXIT_LOAD_TSP_FILE,
    EXIT_EDGE_WEIGHT_TYPE,
    EXIT_EDGE_WEIGHT_FORMAT,
    EXIT_NODE_COORD_TYPE,
    EXIT_RUN_FARM,
    EXIT_WAIT_FREEZING_FARM
};

template <typename T>
inline T parseArg(char * arg) {
    std::clog << "Error: type not supported!" << std::endl;
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
inline void printMatrix(const std::string name, T * matrix, uint32_t rows, uint32_t cols) {

    std::cout << "**** " << name << " ****" << std::endl;
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            std::cout << std::setw(3) << std::setprecision(3) << std::fixed << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

template <typename T>
inline void printMatrixV(const std::string name,
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
    const long msec = getTimerMS();
    const long usec = getTimerUS();
    std::cout << "Compute time: " << msec << " ms " << usec << " usec " << std::endl;
}

inline void stopAndPrintTimer() {
    stopTimer();
    printTimer();
}

inline void stopAndPrintTimer(const std::string name) {
    stopTimer();
    std::cout << name;
    printTimer();
}

template <typename T>
inline void printResult(const std::string & name,
                        const uint32_t mapWorkers,
                        const uint32_t farmWorkers,
                        const uint32_t maxEpoch,
                        const long     timeMS,
                        const long     timeUS,
                        const T        bestTourLength,
                        const T        tspTourLength,
                        const bool     checkTour)
{
#define LOG_SEP " "
    std::clog << std::fixed 
    << " *** "                                       << LOG_SEP
    << name                                          << LOG_SEP
    << mapWorkers                                    << LOG_SEP
    << farmWorkers                                   << LOG_SEP
    << maxEpoch                                      << LOG_SEP
    << timeMS                                        << LOG_SEP
    << timeUS                                        << LOG_SEP
    << bestTourLength                                << LOG_SEP
    << (checkTour ? "Y" : "N")                       << LOG_SEP
    << (bestTourLength == tspTourLength ? "Y" : "N") << LOG_SEP
    << bestTourLength                                << LOG_SEP
    << tspTourLength                                 << LOG_SEP
    << std::endl;
}

#endif
