#include <ff/parallel_for.hpp>
#include <ff/farm.hpp>

#include "random.hpp"

using namespace std;
using namespace ff;

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

        int k = randFloat() * nCities;
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

            float r = randFloat() * sum;
            k = -1;
            for (int j = 0; j < nCities; ++j) {
                if ( k == -1 && p[j] > r) {
                    k = j;
                    break;
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

class AcoFF {

    private:

    ff_Farm<> * farmTour;
    Emitter * emitterTour;

    ParallelForReduce<float> * pfrFloat;
    ParallelForReduce<int> * pfrInt;
    
    int nAnts;
    int nCities;
    float * distance;
    float alpha = 1.0f;
    float beta = 1.0f;
    float q = 100.0f;
    float rho = 0.5f;   
    int maxEpoch;
    int nThreads;

    int elems;
    
    float * eta;
    float * fitness;
    float * delta;
    float * pheromone;

    int * visited;
    int * tabu;
    float * p;
    int * lengths;

    int bestTourLen;
    int * bestTour;

    void initPheromone(float initialPheromone) {
        pfrFloat->parallel_for(0L, elems, [this, &initialPheromone](const long i) {
            pheromone[i] = initialPheromone;
        });
    }

    void initEta() {
        pfrFloat->parallel_for(0L, elems, [this](const long i) {
            eta[i] = (distance[i] == 0 ? 0.0f : 1.0f / distance[i]);
        });
    }

    void calcFitness() {   
        pfrFloat->parallel_for(0L, elems, [this](const long i) {
            fitness[i] = pow(pheromone[i], alpha) + pow(eta[i], beta);
        });
    }

    void calcTour() {
        if (farmTour->run_and_wait_end()<0) {
            error("running farm\n");
            exit(-1);
        }
    }

    void calcBestTour() {      
        pfrInt->parallel_reduce(bestTourLen, INT_MAX, 
                            0L, nAnts,
                            [this](const long i, int &min) { min = (min > lengths[i] ? lengths[i] : min); },
                            [](int &v, const int &elem) { v = (v > elem ? elem : v); });

        for (int i = 0; i < nAnts; ++i) {
            if (lengths[i] == bestTourLen) {
                for (int j = 0; j < nCities; ++j) {
                    bestTour[j] = tabu[i * nCities + j];
                }
                break;
            }
        }
    }

    void clearDelta() {
        pfrFloat->parallel_for(0L, elems, [this](const long i) {
            delta[i] = 0.0f;
        });
    }

    void updateDelta() {
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

    void updatePheromone() {
        pfrFloat->parallel_for(0L, elems, [this](const long i) {
            pheromone[i] = pheromone[i] * (1 - rho) + delta[i];
        });
    }

    public:

    AcoFF(int nAnts, int nCities, float * distance, float alpha, float beta, float q, float rho, int maxEpoch, int nThreads)
    : nAnts(nAnts), nCities(nCities), distance(distance), alpha(alpha), beta(beta), q(q), rho(rho), maxEpoch(maxEpoch), nThreads(nThreads) {

        elems = nCities * nCities;
        eta = (float *) malloc(elems * sizeof(float));
        fitness = (float *) malloc(elems * sizeof(float));
        delta = (float *) malloc(elems * sizeof(float));
        pheromone = (float *) malloc(elems * sizeof(float));

        visited = (int *) malloc(nAnts * nCities * sizeof(int));
        tabu = (int *) malloc(nAnts * nCities * sizeof(int));
        p = (float *) malloc(nAnts * nCities * sizeof(float));
        lengths = (int *) malloc(nAnts * sizeof(int));
        bestTour = (int *) malloc(nCities * sizeof(int));

        pfrFloat = new ParallelForReduce<float>(FF_AUTO, false);
        pfrInt = new ParallelForReduce<int>(FF_AUTO, false);

        farmTour = new ff_Farm<>( [&distance = distance, &fitness = fitness, &tabu = tabu, &delta = delta, &lengths = lengths, &nAnts = nAnts, &nCities = nCities, &nThreads = nThreads]() { 
            vector< unique_ptr<ff_node> > workers;
            for(size_t i = 0; i < nThreads; ++i)
                workers.push_back( make_unique<Worker>(distance, fitness, tabu, delta, lengths, nAnts, nCities) );
            return workers;
        }());

        emitterTour = new Emitter(farmTour->getlb(), nAnts);
        farmTour->add_emitter(*emitterTour);      // replacing the default emitter
        farmTour->remove_collector();  // removing the default collector
        farmTour->wrap_around();       // adds feedback channel between worker and emitter
    }

    void solve() {

        bestTourLen = INT_MAX;

        float initialPheromone = 1.0f / nCities;
        initPheromone(initialPheromone);
        initEta();
        
        int epoch = 0;
        do {

            calcFitness();
            calcTour();
            calcBestTour();
            clearDelta();
            updateDelta();
            updatePheromone();

        } while (++epoch < maxEpoch);
    }

    int * getBestTour() {
        return bestTour;
    }

    int getBestTourLen() {
        return bestTourLen;
    }

    ~AcoFF(){
        delete pfrFloat;
        delete pfrInt;
        delete emitterTour;
        delete farmTour;
        free(eta);
        free(fitness);
        free(delta);
        free(pheromone);

        free(visited);
        free(tabu);
        free(p);
        free(lengths);
        free(bestTour);
    }
};
