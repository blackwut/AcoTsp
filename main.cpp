#include <iostream>
#include <fstream>
#include <cmath>

#include "common.hpp"
#include "TSPReader.cpp"

#ifdef ACO_CPU
#include "AcoCpu.cpp"
#endif
#ifdef ACO_FF
#include "AcoFF.cpp"
#endif

using namespace std;

int main(int argc, char * argv[]) {

    char * path = (char *) malloc(MAX_LEN);
    float alpha = 4.0f;
    float beta = 2.0f;
    float q = 55.0f;
    float rho = 0.8f;
    int maxEpochs = 30;
    int nThreads = 8;

    argc--;
    argv++;
    int args = 0;
    stringArg(argc, argv, args++, path);
    floatArg(argc, argv, args++, &alpha);
    floatArg(argc, argv, args++, &beta);
    floatArg(argc, argv, args++, &q);
    floatArg(argc, argv, args++, &rho);
    intArg(argc, argv, args++, &maxEpochs);
    intArg(argc, argv, args++, &nThreads);

    __seed__ = 123;

    TSP * tsp = getTPSFromFile(path);

#ifdef ACO_CPU
    AcoCpu aco(
#endif
#ifdef ACO_FF
    AcoFF aco(
#endif
        tsp->numberOfCities, 
        tsp->numberOfCities,
        tsp->distance,
        alpha, beta, q, rho, 
        maxEpochs,
        nThreads);

    startTimer();
    aco.solve();
    stopAndPrintTimer();

    int * bestTour = aco.getBestTour();

    printMatrix("BestTour", bestTour, 1, tsp->numberOfCities);
    cout << "BestTourLen: " << aco.getBestTourLen() << endl;
    cout << (checkPathPossible(tsp, bestTour) == 1 ? "Path OK!" : "Error in the path!") << endl;

    free(path);
    free(tsp);

    return 0;
}
