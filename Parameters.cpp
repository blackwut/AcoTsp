#ifndef __PARAMETERS_CPP__
#define __PARAMETERS_CPP__

#include <stdint.h>

template < typename T>
class Parameters {

private:

public:

    const T alpha;
    const T beta;
    const T q;
    const T rho;
    const uint32_t maxEpoch;

    Parameters(T alpha, T beta, T q, T rho, uint32_t maxEpoch) :
    alpha(alpha),
    beta(beta),
    q(q),
    rho(1.0 - rho),
    maxEpoch(maxEpoch)
    {
    }
};

#endif
