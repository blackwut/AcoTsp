#ifndef __ANT_CPP__
#define __ANT_CPP__

#include <stdint.h>
#include <vector>
#include <random>
#include <limits>

#include "common.hpp"

template <typename T>
class Ant {

private:
    
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

    void resetVisited() {
        for (uint8_t & v : visited) {
            v = 1;
        }
    }

    void constructTabu(const std::vector<T> & fitness) {
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
                    visited[k] = 0;
                    break;
                }
            }
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
        const uint32_t from = *(bTabu);
        const uint32_t to   = *(constbTabu);
        tourLength += edges[from * nCities + to];
    }


public:

    Ant(const uint32_t nCities):
    nCities(nCities),
    tabu(nCities),
    visited(nCities),
    p(nCities),
    tourLength(std::numeric_limits<T>::max()) 
    {}

    void constructTour(const std::vector<T> & fitness, const std::vector<T> & edges) {
        resetVisited();
        constructTabu(fitness);
        updateTourLength(edges);
    }

    const T getTourLength() const {
        return tourLength;
    }

    const std::vector<uint32_t> & getTabu() const {
        return tabu;
    }

    inline bool operator< (const Ant<T> & ant) const {
        return tourLength < ant.tourLength;
    }
};

#endif
