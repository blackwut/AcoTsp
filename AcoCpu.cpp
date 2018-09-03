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
	unsigned int seed = 435;
	
	T nextRandom() {
		static std::mt19937 generator(seed);
		static std::uniform_real_distribution<T> distribution(0.0f, 1.0f);
		return distribution(generator);
	}
	
	void initPheromone(T initialPheromone) {
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
		
		for (int i = 0; i < aco->nAnts * aco->nCities; ++i) {
			aco->visited[i] = 1;
		}
		
		for (int id = 0; id < aco->nAnts; ++id) {
			
			int k = nextRandom() * aco->nAnts;
			visited(id, k) = 0;
			tabu(id, 0) = k;
			
			for (int s = 1; s < aco->nCities; ++s) {
				
				T sum = 0.0f;
				const int i = k;
				for (int j = 0; j < aco->nCities; ++j) {
					sum += fitness(i, j) * visited(id, j);
					p(id, j) = sum;
				}
				
				const T r = nextRandom() * sum;
				k = -1;
				for (int j = 0; j < aco->nCities; ++j) {
					if ( k == -1 && p(id, j) >= r ) {
						k = j;
						break;
					}
				}
				
				if ( k == -1 ) {
					cout << "Huston we have a problem!" << endl;
					k = aco->nCities - 1;
				}
				
				visited(id, k) = 0;
				tabu(id, s) = k;
			}
			
			T length = 0.0f;
			int from;
			int to;
			for (int i = 0; i < tsp->dimension - 1; ++i) {
				from = tabu(id, i);
				to = tabu(id, i + 1);
				length += edges(from, to);
			}
			from = tabu(id, aco->nCities - 1);
			to = tabu(id, 0);
			length += edges(from, to);
			
			aco->lengths[id] = length;
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
					aco->bestTour[j] = tabu(i, j);
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
				from = tabu(i , j);
				to = tabu(i, j + 1);
				aco->delta[from * tsp->dimension + to] += aco->q / aco->lengths[i];
			}
			from = tabu(i, aco->nCities - 1);
			to = tabu(i, 0);
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
		
		T initialPheromone = 1.0f / tsp->dimension;
		initPheromone(initialPheromone);
		initEta();
		
		do {
			nextIteration();
		} while (epoch < aco->maxEpoch);
	}
	
	~AcoCpu(){}
};
