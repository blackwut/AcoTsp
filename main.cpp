#include <iostream>
#include <cmath>

#include "AcoCpu.cpp"
#include "AcoFF.cpp"
#include "TSP.cpp"
#include "Parameters.cpp"
#include "Environment.cpp"
#include "common.hpp"

using namespace std;

#ifndef D_TYPE
#define D_TYPE float
#endif

#define LOG_SEP " "

int main(int argc, char * argv[]) {

    char * path     = new char[MAX_LEN];
    D_TYPE alpha    = 1.0;
    D_TYPE beta     = 2.0;
    D_TYPE q        = 1.0;
    D_TYPE rho      = 0.5;
    uint32_t maxEpoch    = 10;
    uint32_t mapWorkers  = 0;
    uint32_t farmWorkers = 0;  

    if (argc < 7 || argc == 8) {
        cout << "Usage: ./acocpu file.tsp alpha beta q rho maxEpoch [mapWorkers farmWorkers]" << endl;
        exit(EXIT_ARGUMENTS_NUMBER);
    }

    argc--;
    argv++;
    path     = argv[0];
    alpha    = parseArg<float>   (argv[1]);
    beta     = parseArg<float>   (argv[2]);
    q        = parseArg<float>   (argv[3]);
    rho      = parseArg<float>   (argv[4]);
    maxEpoch = parseArg<uint32_t>(argv[5]);

    if (argc > 7) {
        mapWorkers  = parseArg<uint32_t>(argv[6]);
        farmWorkers = parseArg<uint32_t>(argv[7]);
    }

    TSP<D_TYPE> tsp(path);
    Parameters<D_TYPE> params(alpha, beta, q, rho, maxEpoch);

    bool parallelCondition = mapWorkers > 0 && farmWorkers > 0;

    if (!parallelCondition) {
        cout << "***** ACO CPU *****" << endl;
        Environment<D_TYPE, D_TYPE> env(tsp.getNCities(), tsp.getNCities(), tsp.getEdges());
        AcoCpu<D_TYPE, D_TYPE> acocpu(params, env);

        startTimer();
        acocpu.solve();
        stopAndPrintTimer();

        printMatrixV("bestTour", env.getBestTour(), 1, env.nCities, 0);

        clog << " *** " << LOG_SEP;
        clog << tsp.getName() << LOG_SEP;
        clog << mapWorkers << LOG_SEP;
        clog << farmWorkers << LOG_SEP;
        clog << maxEpoch << LOG_SEP;
        clog << getTimerMS() << LOG_SEP;
        clog << getTimerUS() << LOG_SEP;
        clog << env.getBestTourLength() << LOG_SEP;
        clog << ( ! tsp.checkPath(env.getBestTour()) ? "Y" : "N");
        clog << endl;
    
    } else {

        cout << "***** ACO FastFlow *****" << endl;
        Environment< D_TYPE, std::atomic<D_TYPE> > env(tsp.getNCities(), tsp.getNCities(), tsp.getEdges());
        AcoFF< D_TYPE, std::atomic<D_TYPE> > acoff(params, env, mapWorkers, farmWorkers);
        startTimer();
        acoff.solve();
        stopAndPrintTimer();

        printMatrixV("bestTour", env.getBestTour(), 1, env.nCities, 0);

        clog << " *** " << LOG_SEP;
        clog << tsp.getName() << LOG_SEP;
        clog << mapWorkers << LOG_SEP;
        clog << farmWorkers << LOG_SEP;
        clog << maxEpoch << LOG_SEP;
        clog << getTimerMS() << LOG_SEP;
        clog << getTimerUS() << LOG_SEP;
        clog << env.getBestTourLength() << LOG_SEP;
        clog << ( ! tsp.checkPath(env.getBestTour()) ? "Y" : "N");
        clog << endl;
    }

  return 0;
}
