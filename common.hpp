#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <iostream>
#include <string>
#include <cstring>
#include <iomanip>
#include <ctime>
#include <chrono>

using namespace std;

enum ERROR {
	EXIT_PARSE_INT = 32,
	EXIT_PARSE_FLOAT,
	EXIT_PARSE_STRING,
    EXIT_LOAD_TSP_FILE,
    EXIT_EDGE_WEIGHT_TYPE,
    EXIT_EDGE_WEIGHT_FORMAT,
    EXIT_NODE_COORD_TYPE
};

#define fitness(a, b)	aco->fitness[a * aco->nCities + b]
#define p(a, b)			aco->p[a * aco->nCities + b]
#define visited(a, b)	aco->visited[a * aco->nCities + b]
#define tabu(a, b)		aco->tabu[a * aco->nCities + b]

#define edges(a, b)		tsp->edges[a * tsp->dimension + b]


#define MAX_LEN 256

void intArg(int argc, char * argv[], int i, int * val) {
    if (argc > i) {
        *val = atoi(argv[i]);
        return;
    }
    clog << "Error while parsing argv[" << i << "] int value" << endl; 
    exit(EXIT_PARSE_INT);
}

template <typename T>
void floatArg(int argc, char * argv[], int i, T * val) {
    if (argc > i) {
        *val = atof(argv[i]);
        return;
    }
    clog << "Error while parsing argv[" << i << "] float value" << endl; 
    exit(EXIT_PARSE_FLOAT);
}

void stringArg(int argc, char * argv[], int i, char * val) {
    if (argc > i) {
        strcpy(val, argv[i]);
        return;
    }
    clog << "Error while parsing argv[" << i << "] string" << endl; 
    exit(EXIT_PARSE_STRING);
}

template <typename T> void printMatrix(string name, T * matrix, int rows, int cols) {

    cout << "*** " << name << " ****" << endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            cout << setw(3) << setprecision(3) << fixed << matrix[i * cols + j] << " ";
        }
        cout << endl;
    }
}

static std::chrono::high_resolution_clock::time_point startPoint;
static std::chrono::high_resolution_clock::time_point endPoint;

void startTimer() {
    startPoint = std::chrono::high_resolution_clock::now();
}

void stopTimer() {
    endPoint = std::chrono::high_resolution_clock::now();
}

long getTimerMS() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(endPoint - startPoint).count();

}

long getTimerUS() {
    return std::chrono::duration_cast<std::chrono::microseconds>(endPoint - startPoint).count();
}

void printTimer() {
    long msec = getTimerMS();
    long usec = getTimerUS();
    cout << "Compute time: " << msec << " ms " << usec << " usec " << endl;
}

void stopAndPrintTimer() {
    stopTimer();
    printTimer();
}

void stopAndPrintTimer(string name) {
    stopTimer();
    cout << name;
    printTimer();
}

#endif
