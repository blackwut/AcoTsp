#include <iostream>
#include <fstream>
#include <cmath>
#include <cfloat>
#include <chrono>
#include <random>

#include "common.hpp"
#include "TSPReader.cpp"
#include <iomanip>
#include <sstream>
#include <ff/parallel_for.hpp>
#include <ff/farm.hpp>

using namespace std;
using namespace ff;

std::mt19937 gen;
float randM() {
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    return dis(gen);
}

struct Emitter: ff_node_t<int> {

    Emitter(ff_loadbalancer * const loadBalancer, int nAnts) 
    : loadBalancer(loadBalancer), nAnts(nAnts) {}

    ff_loadbalancer * const loadBalancer;
    int nAnts;

    int * svc(int * s) {
        if (s == nullptr) { // first time
            
            for (int i = 0; i < nAnts; ++i) {
                int * x = (int *) malloc(sizeof(int));
                *x = i;
                ff_send_out(x);
            }
        
            // broadcasting the End-Of-Stream to all workers
            loadBalancer->broadcast_task(EOS);
        
            // keep me alive 
            return GO_ON;
        }

        free(s);
        return GO_ON;
    }
};

struct Worker: ff_node_t<int> {

    //Read only values
    float * distance; 
    float * fitness;


    //Ants values
    int * visited = NULL;
    float * p = NULL;

    int * tabu;
    float * delta;
    int * lengths;

    int nAnts;
    int nCities;

    Worker(float * distance, float * fitness, int * tabu, float * delta, int * lengths, int nAnts, int nCities)
    : distance(distance), fitness(fitness), tabu(tabu), delta(delta), lengths(lengths), nAnts(nAnts), nCities(nCities) {}

    int * svc(int * in) {

        if (visited == NULL) visited = (int *) malloc(nCities * sizeof(int));
        if (p == NULL) p = (float *) malloc(nCities * sizeof(float));
        if (in == EOS) { free(visited); free(p); return EOS;}

        int id = *in;

        for (int i = 0; i < nCities; ++i) {
            visited[i] = 1;
        }

        int k = randM() * nCities;
        visited[k] = 0;
        tabu[id * nCities] = k; 
        
        float sum;
        int i;
        for (int s = 1; s < nCities; ++s) {
            sum = 0.0f;

            i = k;
            for (int j = 0; j < nCities; ++j) {
                sum += fitness[i * nCities + j] * visited[j];
                p[j] = sum;
            }

            float r = randM() * sum;
            k = -1;
            for (int j = 0; j < nCities; ++j) {
                if ( k == -1 && p[j] > r) {
                    k = j;
                }
            }

            visited[k] = 0;
            tabu[id * nCities + s] = k;
        }

        float length = 0.0f;
        int from;
        int to;
        for (i = 0; i < nCities - 1; ++i) {
            from = tabu[id * nCities + i];
            to = tabu[id * nCities + i + 1];
            length += distance[from * nCities + to];
        }
        from = tabu[id * nCities + nCities - 1];
        to = tabu[id * nCities];
        length += distance[from * nCities + to];

        lengths[id] = (int)length;

        return in;
    }
};

void updateDelta(float * delta, float q, int * tabu, int * lengths, int nAnts, int nCities) {

    int from;
    int to;
    for (int i = 0; i < nAnts; ++i) {
        for (int j = 0; j < nCities - 1; ++j) {
            from = tabu[i * nCities + j];
            to = tabu[i * nCities + j + 1];
            delta[from * nCities + to] += q / lengths[i];
        }
        from = tabu[i * nCities + nCities - 1];
        to = tabu[i * nCities];
        delta[from * nCities + to] += q / lengths[i];
    }
}

int main(int argc, char * argv[]) {

    char * path = (char *) malloc(MAX_LEN);
    float alpha = 4.0f;
    float beta = 2.0f;
    float q = 55.0f;
    float rho = 0.8f;
    int maxEpochs = 30;

    argc--;
    argv++;
    int args = 0;
    stringArg(argc, argv, args++, path);
    floatArg(argc, argv, args++, &alpha);
    floatArg(argc, argv, args++, &beta);
    floatArg(argc, argv, args++, &q);
    floatArg(argc, argv, args++, &rho);
    intArg(argc, argv, args++, &maxEpochs);

    TSP * tsp = getTPSFromFile(path);

    int nCities = tsp->numberOfCities;
    int nAnts = nCities;
    long elems = (long) (nCities * nCities);

    float initialPheromone = 1.0f / nCities;

    float * distance = tsp->distance;
    float * eta = (float *) malloc(elems * sizeof(float));
    float * pheromone = (float *) malloc(elems * sizeof(float));
    float * fitness = (float *) malloc(elems * sizeof(float));
    float * delta = (float *) malloc(elems * sizeof(float));

    int * tabu = (int *) malloc(nAnts * nCities * sizeof(int));
    int * lengths = (int *) malloc(nAnts * sizeof(int));
    int minLen = INT_MAX;
    int * bestTour = (int *) malloc(nCities * sizeof(int));


    gen = std::mt19937(chrono::high_resolution_clock::now().time_since_epoch().count());

    ffTime(START_TIME);

    const size_t nWorkers = 8;


    ParallelForReduce<float> pfr(FF_AUTO, false);
    ParallelForReduce<int> pfrInt(FF_AUTO, false);

    //FARM CALCULATE TOUR
    ff_Farm<> farm( [&distance, &fitness, &tabu, &delta, &lengths, &nAnts, &nCities]() { 
        vector< unique_ptr<ff_node> > workers;
        for(size_t i = 0; i < nWorkers; ++i)
            workers.push_back( make_unique<Worker>(distance, fitness, tabu, delta, lengths, nAnts, nCities) );
        return workers;
    }());

    Emitter emitter(farm.getlb(), nAnts);
    farm.add_emitter(emitter);      // replacing the default emitter
    farm.remove_collector();  // removing the default collector
    farm.wrap_around();       // adds feedback channel between worker and emitter


    //Initialize values
    pfr.parallel_for(0L, elems, [&pheromone, &initialPheromone](const long i) {
        pheromone[i] = initialPheromone;
    });

    pfr.parallel_for(0L, elems, [&distance, &eta](const long i) {
        eta[i] = (distance[i] == 0 ? 0.0f : 1.0f / distance[i]);
    });

    int epoch = 0;
    do {
        //ACO
        pfr.parallel_for(0L, elems, [&fitness, &pheromone, &eta, &alpha, &beta](const long i) {
            fitness[i] = pow(pheromone[i], alpha) + pow(eta[i], beta);
        });

        if (farm.run_and_wait_end()<0) {
            error("running farm\n");
            return -1;
        }

        pfrInt.parallel_reduce(minLen, INT_MAX, 
                            0L, nAnts,
                            [&lengths](const long i, int &min) { min = (min > lengths[i] ? lengths[i] : min); },
                            [](int &v, const int &elem) { v = (v > elem ? elem : v); });

        for (int i = 0; i < nAnts; ++i) {
            if (lengths[i] == minLen) {
                for (int j = 0; j < nCities; ++j) {
                    bestTour[j] = tabu[i * nCities + j];
                }
                break;
            }
        }

        //Initialize Delta
        pfr.parallel_for(0L, elems, [&delta](const long i) {
            delta[i] = 0.0f;
        });

        updateDelta(delta, q, tabu, lengths, nAnts, nCities);

        //Update Pheormone
        pfr.parallel_for(0L, elems, [&pheromone, &delta, &rho](const long i) {
            pheromone[i] = pheromone[i] * (1 - rho) + delta[i];
        });

    } while (++epoch < maxEpochs);

    //farm.ffStats(cout);

    cout << (checkPathPossible(tsp, bestTour) == 1 ? "Path OK!" : "Error in the path!") << endl;
    cout << "bestPathLen: " << minLen << endl;
    cout << "CPU Path distance: " << calculatePath(tsp, bestTour) << endl;
    printMatrix("bestTour", bestTour, 1, nCities);

    std::cout << "Time: " << ffTime(STOP_TIME) << " (ms)\n";

    free(path);
    free(tsp);

    return 0;
}
