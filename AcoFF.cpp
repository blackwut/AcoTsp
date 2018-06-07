#include <random>
#include <limits>
#include <cstddef>

#include <ff/parallel_for.hpp>
#include <ff/farm.hpp>

#include "ACO.cpp"
#include "TSP.cpp"

using namespace std;
using namespace ff;

struct Emitter: ff_node_t<int> {

    ff_loadbalancer * const loadBalancer;
    int nAnts;

    Emitter(ff_loadbalancer * const loadBalancer, int nAnts) 
    : loadBalancer(loadBalancer), nAnts(nAnts) {}

    int * svc(int * s) {
        if (s == nullptr) {
            
            for (int i = 0; i < nAnts; ++i) {
				int * x = new int(i);
                ff_send_out(x);
            }
        
            loadBalancer->broadcast_task(EOS);        
            return GO_ON;
        }

        free(s);
        return GO_ON;
    }
};

template <typename T>
struct Worker: ff_node_t<int> {

    std::mt19937 * generator = NULL;
    std::uniform_real_distribution<T> * distribution = NULL;

	ACO<T> * aco = NULL;
	TSP<T> * tsp = NULL;
	
	Worker(ACO<T> * aco, TSP<T> * tsp)
	: aco(aco), tsp(tsp)
	{
		generator = new std::mt19937((unsigned int)time(0));
		distribution = new std::uniform_real_distribution<T>(0, 1);
	}

	T nextRandom() {
		return distribution->operator()(*generator);
	}
	
	T atomic_addf(atomic<T> * f, T d){
		T old = f->load(std::memory_order_consume);
		T desired = old + d;
		while (!f->compare_exchange_weak(old, desired, std::memory_order_release, std::memory_order_consume))
		{
			desired = old + d;
		}
		return desired;
	}
	
    int * svc(int * in) {

        if (in == EOS) return EOS;

        int id = *in;        

        for (int i = 0; i < aco->nCities; ++i) {
			visited(id, i) = 1;
        }

        int k = nextRandom() * aco->nCities;
		visited(id, k) = 0;
		tabu(id, 0) = k;
        
        for (int s = 1; s < aco->nCities; ++s) {
            T sum = 0;

            int i = k;
            for (int j = 0; j < aco->nCities; ++j) {
				sum += fitness(i, j) * visited(id, j);
				p(id, j) = sum;
            }

            T r = nextRandom() * sum;
            k = -1;
            for (int j = 0; j < aco->nCities; ++j) {
				if ( k == -1 && p(id, j) > r) {
                    k = j;
                    break;
                }
            }

			if (k == -1) {
				cout << "Huston we have a problem!" << endl;
				k = aco->nCities - 1;
			}
			
			visited(id, k) = 0;
			tabu(id, s) = k;
        }

        T length = 0;
        int from;
        int to;
        for (int i = 0; i < aco->nCities - 1; ++i) {
			from = tabu(id, i);
			to = tabu(id, i + 1);
			length += edges(from, to);
        }
		from = tabu(id, aco->nCities - 1);
		to = tabu(id, 0);
		length += edges(from, to);

        aco->lengths[id] = length;

        T d = aco->q / length;
        for (int i = 0; i < aco->nCities - 1; ++i) {
			from = tabu(id, i);
			to = tabu(id, i + 1);
            atomic_addf( (aco->adelta + (from * aco->nCities + to)), d );
        }
		from = tabu(id, aco->nCities - 1);
		to = tabu(id, 0);
        atomic_addf( (aco->adelta + (from * aco->nCities + to)), d );

        return in;
    }
	
	~Worker() {
		delete generator;
		delete distribution;
	}
};

template <typename T>
class AcoFF {

    private:
	
	ACO<T> * aco = NULL;
	TSP<T> * tsp = NULL;

    ff_Farm<> * farmTour = NULL;
    Emitter * emitterTour = NULL;

    ParallelForReduce<T> pfr;
	
	int epoch;

    void initPheromone(T initialPheromone) {
        pfr.parallel_for(0L, aco->elems, [&](const long i) {
            aco->pheromone[i] = initialPheromone;
        });
    }

    void initEta() {
        pfr.parallel_for(0L, aco->elems, [&](const long i) {
            aco->eta[i] = (tsp->edges[i] == 0 ? 0.0f : 1.0f / tsp->edges[i]);
        });
    }

    void calcFitness() {
        pfr.parallel_for(0L, aco->elems, [&](const long i) {
            aco->fitness[i] = pow(aco->pheromone[i], aco->alpha) * pow(aco->eta[i], aco->beta);
        });
    }

    void calcTour() {
        if (farmTour->run_then_freeze() < 0) {
            error("Running farm\n");
            exit(-1);
        }

		if (farmTour->wait_freezing() < 0) {
			error("Wait freezing farm\n");
			exit(-1);
		}
    }

    void calcBestTour() {
		T maxT = numeric_limits<T>::max();
		
		pfr.parallel_reduce(aco->bestTourLen, maxT,
							0L, aco->nAnts,
							[&](const long i, T &min) { min = (min > aco->lengths[i] ? aco->lengths[i] : min); },
							[](T &v, const T &elem) { v = (v > elem ? elem : v); });

        for (int i = 0; i < aco->nAnts; ++i) {
            if (aco->lengths[i] == aco->bestTourLen) {
                for (int j = 0; j < aco->nCities; ++j) {
                    aco->bestTour[j] = aco->tabu[i * aco->nCities + j];
                }
                break;
            }
        }
    }

    void clearDelta() {
        pfr.parallel_for(0L, aco->elems, [&](const long i) {
			aco->adelta[i] = 0;
			
		});
    }

    void updatePheromone() {
        pfr.parallel_for(0L, aco->elems, [&](const long i) {
            aco->pheromone[i] = aco->pheromone[i] * (1 - aco->rho) + aco->adelta[i];
        });
    }

    public:

    AcoFF(ACO<T> * aco, TSP<T> * tsp, int nThreads)
    : aco(aco), tsp(tsp)
	{
        farmTour = new ff_Farm<>( [&]() {
            vector< unique_ptr<ff_node> > workers;
            for(int i = 0; i < nThreads; ++i)
                workers.push_back( make_unique< Worker<T> >(aco, tsp) );
            return workers;
        }());

        emitterTour = new Emitter(farmTour->getlb(), aco->nAnts);
        farmTour->add_emitter(*emitterTour);
        farmTour->remove_collector();
        farmTour->wrap_around();
		
		epoch = 0;
    }

	void nextIteration() {
		epoch++;
		calcFitness();
		clearDelta();
		calcTour();
		calcBestTour();
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

	~AcoFF(){
        delete emitterTour;
        delete farmTour;
    }
};
