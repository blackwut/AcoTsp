#ifndef __ENVIRONMENT_CPP__
#define __ENVIRONMENT_CPP__

#include <stdint.h>
#include <vector>
#include <limits>

template <typename T, typename D>
class Environment {

private:

public:
    const uint32_t nAnts;
    const uint32_t nCities;
    const std::vector<T> & edges;
    std::vector<T> eta;
    std::vector<T> pheromone;
    std::vector<D> delta;
    std::vector<T> fitness;
    std::vector<uint32_t> bestTour;
    T bestTourLength;

    Environment(const uint32_t nAnts, const uint32_t nCities, const std::vector<T> & edges) :
    nAnts    (nAnts),
    nCities  (nCities),
    edges    (edges),
    eta      (nCities * nCities),
    pheromone(nCities * nCities, 1.0 / nCities),
    delta    (nCities * nCities),
    fitness  (nCities * nCities),
    bestTour (nCities),
    bestTourLength(std::numeric_limits<T>::max())
    {}


    const std::vector<uint32_t> & getBestTour() const {
        return bestTour;
    }

    T getBestTourLength() {
        return bestTourLength;
    }
};

#endif
