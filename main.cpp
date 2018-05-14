#include <iostream>
#include <fstream>
#include <cmath>

#include "common.hpp"
#include "TSPReader.cpp"
#include "Aco.cpp"

using namespace std;

int main(int argc, char * argv[]) {

    char * path = (char *) malloc(MAX_LEN);
    float alpha = 4.0f;
    float beta = 2.0f;
    float q = 55.0f;
    float rho = 0.8f;
    int maxEpochs = 30;

    argc--;
    argv++;
    int args = 0;
    stringArg(argc, argv, args++, path);
    floatArg(argc, argv, args++, &alpha);
    floatArg(argc, argv, args++, &beta);
    floatArg(argc, argv, args++, &q);
    floatArg(argc, argv, args++, &rho);
    intArg(argc, argv, args++, &maxEpochs);

    TSP * tsp = getTPSFromFile(path);

    Aco aco(
        tsp->distance, 
        tsp->numberOfCities, 
        alpha, beta, q, rho, 
        tsp->numberOfCities, 
        maxEpochs);

    startTimer();
    aco.solve();
    stopAndPrintTimer();

    aco.printBest();

    int * bestPath = aco.getBestPath();

    cout << (checkPathPossible(tsp, bestPath) == 1 ? "Path OK!" : "Error in the path!") << endl;


    free(path);
    free(tsp);

    return 0;
}
