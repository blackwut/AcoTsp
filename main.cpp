#include <iostream>
#include <cmath>

#include "AcoCpu.cpp"
// #include "AcoFF.cpp"
#include "TSP.cpp"
#include "common.hpp"

using namespace std;

#ifndef D_TYPE
#define D_TYPE float
#endif

int main(int argc, char * argv[]) {

    char * path     = new char[MAX_LEN];
    D_TYPE alpha    = 4.0f;
    D_TYPE beta     = 2.0f;
    D_TYPE q        = 55.0f;
    D_TYPE rho      = 0.8f;
    int maxEpoch    = 30;
    int mapWorkers  = 0;
    int farmWorkers = 0;  

    if (argc < 7) {
        cout << "Usage: ./acocpu file.tsp alpha beta q rho maxEpoch [mapWorkers farmWorkers]" << endl;
        exit(-1);
    }

    argc--;
    argv++;
    int args = 0;
    strArg(argc, argv, args++, path);
    fltArg(argc, argv, args++, &alpha);
    fltArg(argc, argv, args++, &beta);
    fltArg(argc, argv, args++, &q);
    fltArg(argc, argv, args++, &rho);
    intArg(argc, argv, args++, &maxEpoch);
    if (argc >= 7) intArg(argc, argv, args++, &mapWorkers);
    if (argc >= 7) intArg(argc, argv, args++, &farmWorkers);

    TSP<D_TYPE> tsp(path);

    cout << "***** ACO CPU *****" << endl;
    startTimer();
    AcoCpu<D_TYPE> acocpu(tsp, tsp.dimension, tsp.dimension, alpha, beta, q, rho, maxEpoch);
    acocpu.solve();
    stopAndPrintTimer();

    // int parallelCondition = mapWorkers > 0 && farmWorkers > 0;

    // 
    // ACO<D_TYPE> * aco = new ACO<D_TYPE>(tsp->dimension, tsp->dimension, alpha, beta, q, rho, maxEpoch, parallelCondition);

    // if (!parallelCondition) {
    //     cout << "***** ACO CPU *****" << endl;
    //     AcoCpu<D_TYPE> acocpu(aco, tsp);

    //     startTimer();
    //     acocpu.solve();
    //     stopAndPrintTimer();
    // } else {

    //     cout << "***** ACO FastFlow *****" << endl;
    //     AcoFF<D_TYPE> acoff(aco, tsp, mapWorkers, farmWorkers);
    //     startTimer();
    //     acoff.solve();
    //     stopAndPrintTimer();
    // }

    // if ( tsp->checkPath(aco->bestTour) ) {
    //   aco->printBestTour();
    // }

    printMatrixV("bestTour", acocpu.getBestTour(), 1, tsp.dimension, 0);

#define LOG_SEP " "
    clog << " *** " << LOG_SEP;
    clog << tsp.getName() << LOG_SEP;
    clog << mapWorkers << LOG_SEP;
    clog << farmWorkers << LOG_SEP;
    clog << maxEpoch << LOG_SEP;
    clog << getTimerMS() << LOG_SEP;
    clog << getTimerUS() << LOG_SEP;
    clog << acocpu.getBestTourLength() << LOG_SEP;
    clog << (tsp.checkPath(acocpu.getBestTour()) == 1 ? "Y" : "N");
    clog << endl;

  delete[] path;

  return 0;
}
