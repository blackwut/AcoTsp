#ifndef __ACO_CPU_CPP__
#define __ACO_CPU_CPP__

#include <vector>
#include <limits>
#include <random>

#include "TSP.cpp"
#include "common.hpp"

static uint64_t randomSeed = time(0);

template <typename T>
struct Ant {
    const uint32_t nCities;
    std::vector<uint32_t> tabu;
    std::vector<uint8_t> visited;
    std::vector<T> p;
    T tourLength;

    T nextRandom() {
        static std::mt19937 generator(randomSeed);
        static std::uniform_real_distribution<T> distribution(0.0f, 1.0f);
        return distribution(generator);
    }

    Ant(const uint32_t nCities):
    nCities(nCities),
    tabu(nCities),
    visited(nCities),
    p(nCities),
    tourLength(0.0) 
    {}

    void resetVisited() {
        for (uint8_t & v : visited) {
            v = 1;
        }
    }

    void updateTourLength(const std::vector<T> & edges) {

        tourLength = 0.0;
        auto bTabu = tabu.begin();
        const auto constbTabu = bTabu;

        while ( bTabu != tabu.end() - 1 ) {
            const uint32_t from = *(bTabu++);
            const uint32_t to   = *(bTabu);
            tourLength += edges[from * nCities + to];
        }
        const uint32_t from = *(bTabu - 1);
        const uint32_t to   = *(constbTabu);
        tourLength += edges[from * nCities + to];
    }

    void constructTour(const std::vector<T> & fitness, const std::vector<T> & edges) {
        
        resetVisited();

        uint32_t k = nextRandom() * nCities;
        visited[k] = 0;
        tabu[0]    = k;

        for (uint32_t s = 1; s < nCities; ++s) {

            T sum = 0.f;

            auto bP       = p.begin();
            auto bVisited = visited.begin();
            auto bFitness = fitness.begin() + k * nCities;

            while ( bP != p.end() ) {
                auto &pVal       = *(bP++);
                const auto &visitedVal = *(bVisited++);
                const auto &fitnessVal = *(bFitness++);

                sum += fitnessVal * visitedVal;
                pVal = sum;
            }

            const T r = nextRandom() * sum;
            bP = p.begin();
            while ( bP != p.end() ) {
                const T pVal = *(bP++);
                if (pVal >= r) {
                    k          = bP - p.begin() - 1;
                    tabu[s]    = k;
                    visited[k] = false;
                    break;
                }
            }
        }

        updateTourLength(edges);
    }

    inline bool operator< (const Ant<T> & ant) const {
        return tourLength < ant.tourLength;
    }

};

template <typename T>
class AcoCpu {
    
private:
    
    TSP<T> & tsp;
    const uint32_t nAnts;
    const uint32_t nCities;
    const T alpha;
    const T beta;
    const T q;
    const T rho;
    const uint32_t maxEpoch;
    std::vector< Ant<T> > ants;
    std::vector<T> edges;
    std::vector<T> eta;
    std::vector<T> pheromone;
    std::vector<T> delta;
    std::vector<T> fitness;

    std::vector<uint32_t> bestTour;
    T bestTourLength;
    
    void initEta() {
        auto bEta   = eta.begin();
        auto bEdges = edges.begin();
        while ( bEta != eta.end() ) {
            T & etaVal = *(bEta++);
            const T & edgeVal = *(bEdges++);
            
            etaVal = (edgeVal == 0.0 ? 0.0 : 1.0 / edgeVal);
        }
    }

    void calcFitness() {

        auto bFitness   = fitness.begin();
        auto bPheromone = pheromone.begin();
        auto bEta       = eta.begin();

        while ( bFitness != fitness.end() ) {
            T & fitVal = *(bFitness++);
            const T & pheromoneVal = *(bPheromone++);
            const T & etaVal = *(bEta++);
            fitVal = pow(pheromoneVal, alpha) * pow(etaVal, beta);
        }
    }
    
    void calcTour() {
        for (Ant<T> & ant : ants) {
            ant.constructTour(fitness, edges);
        }
    }
    
    void updateBestTour() {
        const Ant<T> & bestAnt = *std::min_element(ants.begin(), ants.end());
        std::copy ( bestAnt.tabu.begin(), bestAnt.tabu.end(), bestTour.begin() );
        bestTourLength = bestAnt.tourLength;
    }
     
    void updateDelta() {

        for (T & d : delta) {
            d = 0.0;
        }

        for (Ant<T> & ant : ants) {

            const float tau = q / ant.tourLength;

            auto bTabu = ant.tabu.begin();
            const auto constbTabu = bTabu;

            while ( bTabu != ant.tabu.end() - 1) {
                const uint32_t from = *(bTabu++);
                const uint32_t to   = *(bTabu);
                delta[from * nCities + to] += tau;
            }
            const uint32_t from = *(bTabu - 1);
            const uint32_t to   = *(constbTabu);
            delta[from * nCities + to] += tau;
        }
    }
    
    void updatePheromone() {

        auto bPheromone = pheromone.begin();
        auto bDelta     = delta.begin();

        while ( bPheromone != pheromone.end() ) {
            T & pheromoneVal = *(bPheromone++);
            T & deltaVal     = *(bDelta++);

            pheromoneVal = pheromoneVal * rho + deltaVal;
        }
    }
    
public:
    
    AcoCpu(TSP<T> & tsp, uint32_t nAnts, uint32_t nCities, T alpha, T beta, T q, T rho, uint32_t maxEpoch):
    tsp(tsp),
    nAnts(nAnts),
    nCities(nCities),
    alpha(alpha),
    beta(beta),
    q(q),
    rho(1.0 - rho),
    maxEpoch(maxEpoch),
    ants(nAnts, Ant<T>(nCities)),
    edges(nCities * nCities),
    eta(nCities * nCities),
    pheromone(nCities * nCities, 1.0 / nCities),
    delta(nCities * nCities),
    fitness(nCities * nCities),
    bestTour(nCities)
    {
        bestTourLength = std::numeric_limits<T>::max();

        for (uint32_t i = 0; i < nCities * nCities; ++i) {
            edges[i] = tsp.edges[i];
        }
    }

    void solve() {
        
        uint32_t epoch = 0;
        initEta();
        
        do {
            calcFitness();
            calcTour();
            updateBestTour();
            updateDelta();
            updatePheromone();
        } while ( ++epoch < maxEpoch );
    }

    const std::vector<uint32_t> & getBestTour() const {
        return bestTour;
    }

    T getBestTourLength() {
        return bestTourLength;
    }
    
    ~AcoCpu(){}
};

#endif
