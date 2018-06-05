#include <iostream>
#include <cmath>

#include "AcoCpu.cpp"
#include "AcoFF.cpp"

using namespace std;

int main(int argc, char * argv[]) {

    char * path = (char *) malloc(MAX_LEN);
    float alpha = 4.0f;
    float beta = 2.0f;
    float q = 55.0f;
    float rho = 0.8f;
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

	TSP<float> * tsp = new TSP<float>::TSP(path);
	ACO<float> * aco = new ACO<float>::ACO(tsp->dimension, tsp->dimension, alpha, beta, q, rho, maxEpoch);

    if (nThreads <= 1) {
		
        cout << "***** ACO CPU *****" << endl;
        AcoCpu<float> acocpu(aco, tsp);

        startTimer();
        acocpu.solve();
        stopAndPrintTimer();

    } else {
		
        cout << "***** ACO FastFlow *****" << endl;
		AcoFF<float> acoff(aco, tsp, nThreads);
        startTimer();
        acoff.solve();
        stopAndPrintTimer();
    }

	if ( tsp->checkPath(aco->bestTour) ) {
		aco->printBestTour();
	}
	
#define LOG_SEP " "
	clog << tsp->getName() << LOG_SEP;
	clog << nThreads << LOG_SEP;
	clog << maxEpoch << LOG_SEP;
	clog << getTimerMS() << LOG_SEP;
	clog << getTimerUS() << LOG_SEP;
	clog << aco->bestTourLen << LOG_SEP;
	clog << (tsp->checkPath(aco->bestTour) == 1 ? "Y" : "N") << LOG_SEP;
	clog << endl;
	
    free(path);
    free(tsp);
	free(aco);

    return 0;
}
