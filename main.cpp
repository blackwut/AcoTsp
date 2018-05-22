#include <iostream>
#include <fstream>
#include <cmath>

#include "common.hpp"
#include "TSPReader.cpp"

#include "AcoCpu.cpp"
#include "AcoFF.cpp"

using namespace std;

int main(int argc, char * argv[]) {


    char * path = (char *) malloc(MAX_LEN);
    float alpha = 4.0f;
    float beta = 2.0f;
    float q = 55.0f;
    float rho = 0.8f;
    int maxEpochs = 30;
    int nThreads = 8;

    if (argc < 8) {
        cout << "Usage: ./acocpu file.tsp alpha beta q rho maxEpochs nThreads" << endl;
        exit(-1);
    }

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

    __seed__ = time(0);

    TSP * tsp = getTPSFromFile(path);
    int * bestTour;
    int bestTourLen;


    if (nThreads <= 1) {
        cout << "***** ACO CPU *****" << endl;
        AcoCpu aco(tsp->numberOfCities, 
                    tsp->numberOfCities,
                    tsp->distance,
                    alpha, beta, q, rho, 
                    maxEpochs,
                    nThreads);

        startTimer();
        aco.solve();
        stopAndPrintTimer();

        bestTour = aco.getBestTour();
        bestTourLen = aco.getBestTourLen();

    } else {
        cout << "***** ACO FastFlow *****" << endl;
        AcoFF aco(tsp->numberOfCities, 
                    tsp->numberOfCities,
                    tsp->distance,
                    alpha, beta, q, rho, 
                    maxEpochs,
                    nThreads);

        startTimer();
        aco.solve();
        stopAndPrintTimer();

        bestTour = aco.getBestTour();
        bestTourLen = aco.getBestTourLen();
    }
        
    printMatrix("BestTour", bestTour, 1, tsp->numberOfCities);
    cout << "BestTourLen: " << bestTourLen << endl;
    cout << (checkPathPossible(tsp, bestTour) == 1 ? "Path OK!" : "Error in the path!") << endl;

    free(path);
    free(tsp->distance);
    free(tsp);

    return 0;
}
