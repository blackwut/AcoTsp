#include <cfloat>

#include "random.hpp"

using namespace std;

class AcoCpu {

    private:
        
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
        for (int i = 0; i < nCities * nCities; ++i) {
            pheromone[i] = initialPheromone;
        }
    }

    void initEta() {
        for (int i = 0; i < nCities * nCities; ++i) {
            eta[i] = (distance[i] == 0 ? 0.0f : 1.0f / distance[i]);
        }
    }

    void calcFitness() {   
        for (int i = 0; i < nCities * nCities; ++i) {
            fitness[i] = pow(pheromone[i], alpha) + pow(eta[i], beta);
        }
    }

    void calcTour() {

        for (int i = 0; i < nAnts * nCities; ++i) {
            visited[i] = 1;
        }

        for (int id = 0; id < nAnts; ++id) {

            int k = randFloat() * nCities;
            visited[id * nCities + k] = 0;
            tabu[id * nCities] = k;

            float sum;
            int i;
            for (int s = 1; s < nCities; ++s) {
                sum = 0.0f;

                i = k;
                for (int j = 0; j < nCities; ++j) {
                    sum += fitness[i * nCities + j] * visited[id * nCities + j];
                    p[id * nCities + j] = sum;
                }

                float r = randFloat() * sum;
                k = -1;
                for (int j = 0; j < nCities; ++j) {
                    if ( k == -1 && p[id * nCities +j] > r) {
                        k = j;
                        break;
                    }
                }

                visited[id * nCities + k] = 0;
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
        }
    }

    void calcBestTour() {      
        for (int i = 0; i < nAnts; ++i) {
            if (bestTourLen > lengths[i]) {
                bestTourLen = lengths[i];
            }
        }

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
        for (int i = 0; i < nCities * nCities; ++i) {
            delta[i] = 0.0f;
        }
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
        for (int i = 0; i < nCities * nCities; ++i) {
            pheromone[i] = pheromone[i] * (1 - rho) + delta[i];
        }
    }

    public:

    AcoCpu(int nAnts, int nCities, float * distance, float alpha, float beta, float q, float rho, int maxEpoch, int nThreads)
    : nAnts(nAnts), nCities(nCities), distance(distance), alpha(alpha), beta(beta), q(q), rho(rho), maxEpoch(maxEpoch) {

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

    ~AcoCpu(){
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
