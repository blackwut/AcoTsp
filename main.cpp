#include <iostream>
#include <cmath>

#include "AcoCpu.cpp"
#include "AcoFF.cpp"

using namespace std;

#ifndef D_TYPE
#define D_TYPE double
#endif

int main(int argc, char * argv[]) {

	char * path = new char[MAX_LEN];
    D_TYPE alpha = 4.0f;
    D_TYPE beta = 2.0f;
    D_TYPE q = 55.0f;
    D_TYPE rho = 0.8f;
    int maxEpoch = 30;
    int nThreads = 8;

    if (argc < 8) {
		cout << "Usage: ./acocpu file.tsp alpha beta q rho maxEpoch nThreads" << endl;
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
    intArg(argc, argv, args++, &maxEpoch);
    intArg(argc, argv, args++, &nThreads);

	TSP<D_TYPE> * tsp = new TSP<D_TYPE>(path);
	ACO<D_TYPE> * aco = new ACO<D_TYPE>(tsp->dimension, tsp->dimension, alpha, beta, q, rho, maxEpoch, nThreads > 1);

    if (nThreads <= 1) {
		
        cout << "***** ACO CPU *****" << endl;
        AcoCpu<D_TYPE> acocpu(aco, tsp);

        startTimer();
        acocpu.solve();
        stopAndPrintTimer();

    } else {
		
        cout << "***** ACO FastFlow *****" << endl;
		AcoFF<D_TYPE> acoff(aco, tsp, nThreads);
        startTimer();
        acoff.solve();
        stopAndPrintTimer();
    }

	if ( tsp->checkPath(aco->bestTour) ) {
		aco->printBestTour();
	}
	
#define LOG_SEP " "
	clog << " *** " << LOG_SEP;
	clog << tsp->getName() << LOG_SEP;
	clog << nThreads << LOG_SEP;
	clog << maxEpoch << LOG_SEP;
	clog << getTimerMS() << LOG_SEP;
	clog << getTimerUS() << LOG_SEP;
	clog << aco->bestTourLen << LOG_SEP;
	clog << (tsp->checkPath(aco->bestTour) == 1 ? "Y" : "N") << LOG_SEP;
	clog << endl;
	
    delete[] path;
	delete tsp;
	delete aco;

    return 0;
}
