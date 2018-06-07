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
	int nCities;
	int elems;
	T alpha;
	T beta;
	T q;
	T rho;
	int maxEpoch;
	bool atomicDelta;
	
	T * eta;
	T * fitness;
	T * delta;
	atomic<T> * adelta;
	T * pheromone;
	T * p;
	
	int * visited;
	int * tabu;
	T * lengths;
	
	T bestTourLen;
	int * bestTour;
	
	ACO(int nAnts, int nCities, T alpha, T beta, T q, T rho, int maxEpoch, bool atomicDelta) :
	nAnts(nAnts), nCities(nCities), elems(nCities * nCities), alpha(alpha), beta(beta), q(q), rho(rho), maxEpoch(maxEpoch), atomicDelta(atomicDelta) {
		
		eta = new T[elems];
		fitness = new T[elems];
		
		if (atomicDelta)
			adelta = new atomic<T>[elems];
		else delta = new T[elems];
		
		pheromone = new T[elems];
		
		visited = new int[nAnts * nCities];
		tabu = new int[nAnts * nCities];
		p = new T[nAnts * nCities];
		lengths = new T[nAnts];
		bestTour = new int[nCities];
	}
	
	void printBestTour() {
		cout << "Best Tour Length: " << bestTourLen << endl;
		printMatrix("Best Tour", bestTour, 1, nCities);
	}
	
	~ACO() {
		delete[] eta;
		delete[] fitness;
		
		if (atomicDelta)
			delete[] adelta;
		else delete[] delta;
		
		delete[] pheromone;
		delete[] p;
		
		delete[] visited;
		delete[] tabu;
		delete[] lengths;
		
		delete[] bestTour;
	}
};

#endif
