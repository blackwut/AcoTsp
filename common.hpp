#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <iomanip>
#include <ctime>
#include <chrono>

using namespace std;

#define MAX_LEN 1024

void intArg(int argc, char * argv[], int i, int * val) {
    if (argc > i) {
        *val = atoi(argv[i]);
        return;
    }
    wcerr << "Error while parsing argv[" << i << "] int value" << endl; 
    exit(-1);
}

void floatArg(int argc, char * argv[], int i, float * val) {
    if (argc > i) {
        *val = atof(argv[i]);
        return;
    }
    wcerr << "Error while parsing argv[" << i << "] float value" << endl; 
    exit(-1);
}

void stringArg(int argc, char * argv[], int i, char * val) {
    if (argc > i) {
        strcpy(val, argv[i]);
        return;
    }
    wcerr << "Error while parsing argv[" << i << "] string" << endl; 
    exit(-1);
}

template <typename T> void printMatrix(char * name, T * matrix, int rows, int cols) {

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

void printTimer() {
    long msec = std::chrono::duration_cast<std::chrono::milliseconds>(endPoint - startPoint).count();
    long usec = std::chrono::duration_cast<std::chrono::microseconds>(endPoint - startPoint).count();
    wcerr << "Computational time: " << msec << " ms " << usec << " usec " << endl;
}

void stopAndPrintTimer() {
    stopTimer();
    printTimer();
}

#endif