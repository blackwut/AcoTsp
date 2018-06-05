#include <random>
#include <climits>

#include "ACO.cpp"
#include "TSP.cpp"

using namespace std;


template <typename T>
class AcoCpu {
	
private:
	
	ACO<T> * aco;
	TSP<T> * tsp;
	
	int epoch;
	unsigned int seed = 123;
	
	T randFloat() {
		static std::mt19937 generator(seed);
		static std::uniform_real_distribution<T> distribution(0.0f, 1.0f);
		return distribution(generator);
	}
	
	void initPheromone(float initialPheromone) {
		for (int i = 0; i < aco->elems; ++i) {
			aco->pheromone[i] = initialPheromone;
		}
	}
	
	void initEta() {
		for (int i = 0; i < aco->elems; ++i) {
			aco->eta[i] = (tsp->edges[i] == 0 ? 0.0f : 1.0f / tsp->edges[i]);
		}
	}
	
	void calcfitness() {
		for (int i = 0; i < aco->elems; ++i) {
			aco->fitness[i] = pow(aco->pheromone[i], aco->alpha) * pow(aco->eta[i], aco->beta);
		}
	}
	
	void calcTour() {
		
		for (int i = 0; i < aco->nAnts * tsp->dimension; ++i) {
			aco->visited[i] = 1;
		}
		
		for (int id = 0; id < aco->nAnts; ++id) {
			
			int k = randFloat() * aco->nAnts;
			aco->visited[id * tsp->dimension + k] = 0;
			aco->tabu[id * tsp->dimension] = k;
			
			for (int s = 1; s < tsp->dimension; ++s) {
				float sum = 0.0f;
				
				int i = k;
				for (int j = 0; j < tsp->dimension; ++j) {
					sum += aco->fitness[i * tsp->dimension + j] * aco->visited[id * tsp->dimension + j];
					aco->p[id * tsp->dimension + j] = sum;
				}
				
				float r = randFloat() * sum;
				k = -1;
				for (int j = 0; j < tsp->dimension; ++j) {
					if ( k == -1 && aco->p[id * tsp->dimension +j] > r ) {
						k = j;
						break;
					}
				}
				
				if ( k == -1 ) {
					k = tsp->dimension - 1;
				}
				
				aco->visited[id * tsp->dimension + k] = 0;
				aco->tabu[id * tsp->dimension + s] = k;
			}
			
			float length = 0.0f;
			int from;
			int to;
			for (int i = 0; i < tsp->dimension - 1; ++i) {
				from = aco->tabu[id * tsp->dimension + i];
				to = aco->tabu[id * tsp->dimension + i + 1];
				length += tsp->edges[from * tsp->dimension + to];
			}
			from = aco->tabu[id * tsp->dimension + tsp->dimension - 1];
			to = aco->tabu[id * tsp->dimension];
			length += tsp->edges[from * tsp->dimension + to];
			
			aco->lengths[id] = (int)length;
		}
	}
	
	void calcBestTour() {
		for (int i = 0; i < aco->nAnts; ++i) {
			if (aco->bestTourLen > aco->lengths[i]) {
				aco->bestTourLen = aco->lengths[i];
			}
		}
		
		for (int i = 0; i < aco->nAnts; ++i) {
			if (aco->lengths[i] == aco->bestTourLen) {
				for (int j = 0; j < tsp->dimension; ++j) {
					aco->bestTour[j] = aco->tabu[i * tsp->dimension + j];
				}
				break;
			}
		}
	}
	
	void clearDelta() {
		for (int i = 0; i < tsp->dimension * tsp->dimension; ++i) {
			aco->delta[i] = 0.0f;
		}
	}
	
	void updateDelta() {
		int from;
		int to;
		for (int i = 0; i < aco->nAnts; ++i) {
			for (int j = 0; j < tsp->dimension - 1; ++j) {
				from = aco->tabu[i * tsp->dimension + j];
				to = aco->tabu[i * tsp->dimension + j + 1];
				aco->delta[from * tsp->dimension + to] += aco->q / aco->lengths[i];
			}
			from = aco->tabu[i * tsp->dimension + tsp->dimension - 1];
			to = aco->tabu[i * tsp->dimension];
			aco->delta[from * tsp->dimension + to] += aco->q / aco->lengths[i];
		}
	}
	
	void updatePheromone() {
		for (int i = 0; i < tsp->dimension * tsp->dimension; ++i) {
			aco->pheromone[i] = aco->pheromone[i] * (1 - aco->rho) + aco->delta[i];
		}
	}
	
public:
	
	AcoCpu(ACO<T> * aco, TSP<T> * tsp)
	: aco(aco), tsp(tsp)
	{
		epoch = 0;
		seed = (unsigned int) time(0);
	}
	
	void nextIteration() {
		epoch++;
		calcfitness();
		calcTour();
		calcBestTour();
		clearDelta();
		updateDelta();
		updatePheromone();
	}
	
	void solve() {
		
		aco->bestTourLen = INT_MAX;
		epoch = 0;
		
		float initialPheromone = 1.0f / tsp->dimension;
		initPheromone(initialPheromone);
		initEta();
		
		do {
			nextIteration();
		} while (epoch < aco->maxEpoch);
	}
	
	~AcoCpu(){}
};
