#include <chrono>
#include <ctime>
#include <random>
#include <cfloat>

#include "Ant.cpp"
// #include "GPUAco.cu"

using namespace std;

#define _arc(a, b) _arc[a * _numberOfCities + b]
#define _eta(a, b) _eta[a * _numberOfCities + b]
#define _pheromone(a, b) _pheromone[a * _numberOfCities + b]
#define _delta(a, b) _delta[a * _numberOfCities + b]

class Aco {

    private:
        float _alpha = 1.0f;
        float _beta = 5.0f;
        float _q = 100.0f;
        float _rho = 0.5f;   

        int _numberOfAnts;
        int _numberOfCities;
        int _maxEpoch;

        float _bestLength = FLT_MAX;
        int * _bestPath;

        Ant ** _ants;
        float * _arc;
        float * _eta;
        float * _pheromone;
        float * _delta;

        std::mt19937 _gen;

        float rand() {
            std::uniform_real_distribution<float> dis(0.0f, 1.0f);
            return dis(_gen);
        }

        void placeAnts() {
            for (int i = 0; i < _numberOfAnts; ++i) {
                int j = floor(rand() * _numberOfCities);
                _ants[i]->visited[j] = 1;
                _ants[i]->tabu[0] = j;
                _ants[i]->next = j;
            }
        }

        void clearAnts() {
            for (int i = 0; i < _numberOfAnts; ++i) {
                clearAnt(_ants[i]);
            }
        }

        float fitness(int i, int j) {
            return pow(_pheromone(i, j), _alpha) + pow(_eta(i, j), _beta);
        }

        int selectNextForAnt(Ant * ant) {

            int i = ant->next;
            float sum = 0.0f;
            for (int j = 0; j < _numberOfCities; ++j) {
                sum += fitness(i, j) * (1.0f - ant->visited[j]);
                ant->p[j] = sum;
            }

            int k = -1;
            float r = rand() * sum;
            //Find probabilities
            // for (int j = 0; j < _numberOfCities; ++j) {
            //     ant->p[j] /= sum;
            // }

            //Select the city according with r
            for (int j = 0; j < _numberOfCities; ++j) {
                if (k == -1 && ant->p[j] > r) {
                    k = j;
                }
            }
            return k;

            // int j = 0;
            // while (1) {
            //     if ( j >= _numberOfCities ) {
            //         j = 0;
            //     }

            //     float p = fitness(i, j) * (1.0f - ant->visited[j]) / sum;
            //     float r = rand();
            //     if ( p > r ) {
            //         return j;
            //     }
            //     j++;
            // }
            // return j;
        }

        void constructTour() {
            for (int s = 1; s < _numberOfCities; ++s) {
                for (int i = 0; i < _numberOfAnts; ++i) {

                    int next = selectNextForAnt(_ants[i]);

                    _ants[i]->next = next;
                    _ants[i]->visited[next] = 1;
                    _ants[i]->tabu[s] = next;         
                    _ants[i]->length += _arc(_ants[i]->tabu[s - 1], next);
                }
            }
        }

        void wrapTour() {
            for (int i = 0; i < _numberOfAnts; ++i) {
                _ants[i]->length += _arc(_ants[i]->tabu[_numberOfCities - 1], _ants[i]->tabu[0]);
            }
        }

        void findBestTour() {
            for (int i = 0; i < _numberOfAnts; ++i) {
                if (_bestLength > _ants[i]->length) {
                    _bestLength = _ants[i]->length;

                    for (int b = 0; b < _numberOfCities; ++b){
                        _bestPath[b] = _ants[i]->tabu[b];
                    }
                }
            }
        }

        void clearDelta() {
            for (int i = 0; i < _numberOfAnts; ++i) {
                for (int j = 0; j < _numberOfCities; ++j) {
                    _delta(i, j) = 0.0f;
                }
            }
        }
        void updateDelta() {

            for (int i = 0; i < _numberOfAnts; ++i) {
                Ant * ant = _ants[i];
                for (int j = 0; j < _numberOfCities; ++j) {
                    int from = ant->tabu[j];
                    int to = ant->tabu[ (j + 1) % _numberOfCities ];
                    _delta(from, to) += _q / ant->length;
                    _delta(to, from) = _delta(from, to);
                }
            }
        }

        void updatePheromone() {
            for (int i = 0; i < _numberOfAnts; ++i) {
                for (int j = 0; j < _numberOfCities; ++j) {
                    _pheromone(i, j) *= (1.0f - _rho);
                    _pheromone(i, j) += _delta(i, j);
                }
            }
        }

    public:

        Aco(float * arc, int numberOfCities, float alpha, float beta, float q, float rho, int numberOfAnts, int maxEpoch) {

            _gen = std::mt19937(chrono::high_resolution_clock::now().time_since_epoch().count());

            _arc = arc;
            _numberOfCities = numberOfCities;
            _alpha = alpha;
            _beta = beta;
            _q = q;
            _rho = rho;
            _numberOfAnts = numberOfAnts;
            _maxEpoch = maxEpoch;

            _bestPath = (int *) malloc(_numberOfCities * sizeof(int));
        }

        void solve() {

            _ants = (Ant **) malloc(_numberOfAnts * sizeof(Ant *));
            for (int i = 0; i < _numberOfAnts; ++i) {
                _ants[i] = newAnt(_numberOfCities);
            }

            _eta = (float *) malloc(_numberOfCities * _numberOfCities * sizeof(float));
            for (int i = 0; i < _numberOfCities; ++i) {
                for (int j = 0; j < _numberOfCities; ++j) {
                    _eta(i, j) = (i == j ? 0.0f : 1.0f / _arc(i, j));
                }
            }

             _pheromone = (float *) malloc(_numberOfCities * _numberOfCities * sizeof(float));
            for (int i = 0; i < _numberOfCities; ++i) {
                for (int j = 0; j < _numberOfCities; ++j) {
                    _pheromone(i, j) = (i == j ? 0.0f : 1.0f / _numberOfCities);
                }
            }

            _delta = (float *) malloc(_numberOfCities * _numberOfCities * sizeof(float));
            for (int i = 0; i < _numberOfCities; ++i) {
                for (int j = 0; j < _numberOfCities; ++j) {
                    _delta(i, j) = 0.0f;
                }
            }

            int epoch = 0;
            do {

                placeAnts();
                constructTour();
                wrapTour();
                findBestTour();
                updateDelta();
                updatePheromone();
                clearDelta();
                clearAnts();

            } while (++epoch < _maxEpoch);
        }

        // void solveGPU() {

        //     int blockSize = 32;
        //     int elems = _numberOfCities * _numberOfCities;
        //     int numBlock = numberOfBlocks(elems, blockSize);

        //     float * distance;
        //     float * eta;
        //     cudaMallocManaged(&distance, elems * sizeof(float));
        //     cudaMallocManaged(&eta, elems * sizeof(float));

        //     for (int i = 0; i < _numberOfCities; ++i) {
        //         for (int j = 0; j < _numberOfCities; ++j) {
        //             distance[i * _numberOfCities + j] = _arc[i * _numberOfCities +j];
        //         }
        //     }

        //     calculateEta<<<numBlock, blockSize>>>(distance, eta, elems, _numberOfCities);
        //     cudaDeviceSynchronize();

        //     printFloatMatrix(eta, _numberOfCities, _numberOfCities);

        //     cudaFree(distance);
        //     cudaFree(eta);
        // }
        
        void printBest() {

            cout << "Best Tour Length: " << _bestLength << endl;

            cout << "Path: {";
            for (int i = 0; i < _numberOfCities; ++i) {
                cout << _bestPath[i] << ", ";    
            } 
            cout << _bestPath[0] << "}" << endl;
        }

        int * getBestPath() {
            return _bestPath;
        }

        ~Aco(){
            free(_bestPath);
            for (int i = 0; i < _numberOfAnts; ++i) {
                free(_ants[i]);
            }
            free(_ants);
            free(_arc);
            free(_eta);
            free(_pheromone);
            free(_delta);
        }
};
