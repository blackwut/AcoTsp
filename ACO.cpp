#ifndef __ACO_HPP__
#define __ACO_HPP__

#include <iostream>
#include <atomic>

#include "common.hpp"

template <typename T>
class ACO {
	
private:
	
public:
	int nAnts;
	T alpha;
	T beta;
	T q;
	T rho;
	int maxEpoch;
	
	int nCities;
	int elems;
	T * eta;
	T * fitness;
	T * delta;
	atomic<T> * adelta;
	T * pheromone;
	T * p;
	
	int * visited;
	int * tabu;
	int * lengths;
	
	T bestTourLen;
	int * bestTour;
	
	ACO(int nAnts, int nCities, T alpha, T beta, T q, T rho, int maxEpoch) :
	nAnts(nAnts), alpha(alpha), beta(beta), q(q), rho(rho), maxEpoch(maxEpoch), nCities(nCities), elems(nCities * nCities) {
		
		eta = (T *) malloc(elems * sizeof(T));
		fitness = (T *) malloc(elems * sizeof(T));
		delta = (T *) malloc(elems * sizeof(T));
		adelta = (atomic<T> *) malloc(elems * sizeof(atomic<T>));
		if (adelta == NULL) cout << "porcodio" << endl;
		pheromone = (T *) malloc(elems * sizeof(T));
		
		visited = (int *) malloc(nAnts * nCities * sizeof(int));
		tabu = (int *) malloc(nAnts * nCities * sizeof(int));
		p = (T *) malloc(nAnts * nCities * sizeof(T));
		lengths = (int *) malloc(nAnts * sizeof(int));
		bestTour = (int *) malloc(nCities * sizeof(int));
	}
	
	void printBestTour() {
		cout << "Best Tour Length: " << bestTourLen << endl;
		printMatrix("Best Tour", bestTour, 1, nCities);
	}
	
	~ACO() {
		free(eta);
		free(fitness);
		free(delta);
		free(adelta);
		free(pheromone);
		free(p);
		
		free(visited);
		free(tabu);
		free(lengths);
		
		free(bestTour);
	}
};

#endif
