#ifndef __RANDOM_HPP__
#define __RANDOM_HPP__

#include <random>

long __seed__ = 123;

float randFloat() {
    static std::mt19937 generator(__seed__);
    static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    return distribution(generator);
}

#endif