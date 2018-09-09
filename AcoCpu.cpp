#ifndef __ACO_CPU_CPP__
#define __ACO_CPU_CPP__

#include <stdint.h>
#include <vector>
#include <algorithm>

#include "Environment.cpp"
#include "Parameters.cpp"
#include "Ant.cpp"
#include "common.hpp"

template <typename T, typename D>
class AcoCpu {
    
private:
    const Parameters<T> & params;
    Environment<T, D>   & env;
    std::vector< Ant<T> > ants;
    
    void initEta(std::vector<T> & eta, 
                 const std::vector<T> & edges)
    {
        auto bEta   = eta.begin();
        auto bEdges = edges.begin();
        while ( bEta != eta.end() ) {
            T & etaVal = *(bEta++);
            const T & edgeVal = *(bEdges++);
            
            etaVal = (edgeVal == 0.0 ? 0.0 : 1.0 / edgeVal);
        }
    }

    void calcFitness(std::vector<T> & fitness,
                     const std::vector<T> & pheromone,
                     const std::vector<T> & eta,
                     const T alpha,
                     const T beta)
    {
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
    
    void calcTour(const std::vector<T> & fitness,
                  const std::vector<T> & edges)
    {
        for (Ant<T> & ant : ants) {
            ant.constructTour(fitness, edges);
        }
    }
    
    void updateBestTour(std::vector<uint32_t> & bestTour,
                        T & bestTourLength)
    {
        const Ant<T> & bestAnt = *std::min_element(ants.begin(), ants.end());
        std::copy ( bestAnt.getTabu().begin(), bestAnt.getTabu().end(), bestTour.begin() );
        bestTourLength = bestAnt.getTourLength();
    }
     
    void updateDelta(std::vector<D> & delta,
                     const uint32_t nCities,
                     const T q)
    {
        for (T & d : delta) {
            d = 0.0;
        }

        for (Ant<T> & ant : ants) {

            const float tau = q / ant.getTourLength();

            auto bTabu = ant.getTabu().begin();
            const auto constbTabu = bTabu;

            while ( bTabu != ant.getTabu().end() - 1) {
                const uint32_t from = *(bTabu++);
                const uint32_t to   = *(bTabu);
                delta[from * nCities + to] += tau;
            }
            const uint32_t from = *(bTabu);
            const uint32_t to   = *(constbTabu);
            delta[from * nCities + to] += tau;
        }
    }
    
    void updatePheromone(std::vector<T> & pheromone,
                         const std::vector<D> & delta,
                        const T rho)
    {
        auto bPheromone = pheromone.begin();
        auto bDelta     = delta.begin();

        while ( bPheromone != pheromone.end() ) {
            T & pheromoneVal = *(bPheromone++);
            const T & deltaVal     = *(bDelta++);

            pheromoneVal = pheromoneVal * rho + deltaVal;
        }
    }
    
public:
    
    AcoCpu(const Parameters<T> & params, Environment<T, D> & env):
    params(params),
    env   (env),
    ants  (env.nAnts, Ant<T>(env.nCities))
    {
        initEta(env.eta, env.edges);
    }

    void solve() {
        uint32_t epoch = 0;
        do {
            calcFitness    (env.fitness, env.pheromone, env.eta, params.alpha, params.beta);
            calcTour       (env.fitness, env.edges);
            updateBestTour (env.bestTour, env.bestTourLength);
            updateDelta    (env.delta, env.nCities, params.q);
            updatePheromone(env.pheromone, env.delta, params.rho);
        } while ( ++epoch < params.maxEpoch );
    }
    
    ~AcoCpu(){}
};

#endif
