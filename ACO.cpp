#ifndef __ACO_CPP__
#define __ACO_CPP__

#include <iostream>
#include <atomic>

#include "common.hpp"

template <typename T>
class ACO {
	
private:
	
public:
	uint32_t nAnts;
	uint32_t nCities;
	uint32_t elems;
	T alpha;
	T beta;
	T q;
	T rho;
	uint32_t maxEpoch;
	bool atomicDelta;
	
	T * eta;
	T * fitness;
	T * delta;
	atomic<T> * adelta;
	T * pheromone;
	T * p;
	
	uint8_t * visited;
	uint32_t * tabu;
	T * lengths;
	
	T bestTourLen;
	uint32_t * bestTour;
	
	ACO(uint32_t nAnts, uint32_t nCities, T alpha, T beta, T q, T rho, uint32_t maxEpoch, bool atomicDelta) :
	nAnts(nAnts), nCities(nCities), elems(nCities * nCities), alpha(alpha), beta(beta), q(q), rho(rho), maxEpoch(maxEpoch), atomicDelta(atomicDelta) {
		
		eta = new T[elems];
		fitness = new T[elems];
		
		if (atomicDelta)
			adelta = new atomic<T>[elems];
		else delta = new T[elems];
		
		pheromone = new T[elems];
		
		v = new uint8_t[nAnts * nCities];
		tabu = new uint32_t[nAnts * nCities];
		p = new T[nAnts * nCities];
		lengths = new T[nAnts];
		bestTour = new uint32_t[nCities];
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
