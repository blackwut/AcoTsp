#include <iostream>
#include <fstream>
#include <cmath>
#include <climits>

#include <curand.h>
#include <curand_kernel.h>

#include "common.hpp"
#include "ACO.cpp"
#include "TSP.cpp"
#include <float.h>

#include <cooperative_groups.h>
using namespace cooperative_groups;

#define TOUR_PRIVATE 1
#define TOUR_PRIVATE_2 2
#define TOUR_WARP 3

#define TOUR_SELECTED TOUR_WARP

__global__ 
void initCurand(curandStateXORWOW_t * state,
                const unsigned long seed,
                const uint32_t nAnts)
{
    const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if ( idx >= nAnts ) return;

    curand_init(seed, idx, 0, &state[idx]);
}

__device__ 
float randXOR(curandState * state)
{
    return (float) curand_uniform(state);
}

__global__
void initialize(const float * distance,
                float * eta,
                float * pheromone,
                float * delta,
                const float initialPheromone,
                const uint32_t rows,
                const uint32_t cols)
{    
    const uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if ( row >= rows || col >= cols ) return;

    const uint32_t idx = row * cols + col;
    const float d = distance[idx];
    if ( d == 0 ) {
        eta[idx] = 0.0f;
    } else {
        eta[idx] = 1.0f / d;
    }

    pheromone[idx] = initialPheromone;
    delta[idx] = 0.0f;
}

__global__
void calculateFitness(float * fitness,
                      const float * pheromone,
                      const float * eta,
                      const float alpha,
                      const float beta,
                      const uint32_t rows,
                      const uint32_t cols)
{
    const uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if ( row >= rows || col >= cols ) return;

    const uint32_t idx = row * cols + col;
    fitness[idx] = __powf(pheromone[idx], alpha) * __powf(eta[idx], beta);
}

__global__
void claculateTourPrivate(uint32_t * tabu, float * fitness, uint32_t rows, uint32_t cols, curandStateXORWOW_t * state)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= rows ) return;

    float p[1024];
    uint32_t visited[1024];
    float r;
    float sum;
    uint32_t j;
    int32_t k;

    for (uint32_t i = 0; i < cols; ++i) {
        visited[i] = 1;
    }

    k = cols * randXOR(state + idx);
    visited[k] = 0;
    tabu[idx * cols] = k;

    for (uint32_t s = 1; s < cols; ++s) {

        sum = 0.0f;
        const uint32_t i = k;
        for (j = 0; j < cols; ++j) {
            sum += fitness[i * cols + j] * visited[j];
            p[j] = sum;
        }

        r = randXOR(state + idx) * sum;
        k = -1;
        for (j = 0; j < cols; ++j) {
            // if ( k == -1 && p[j] > r ) {
            //     k = j;
            // }
            k += (k == -1) * (p[j] > r) * (j + 1);
        }
        visited[k] = 0;
        tabu[idx * cols + s] = k;
    }
}

__global__
void claculateTourPrivate_2(uint32_t * tabu, float * fitness, uint32_t rows, uint32_t cols, curandStateXORWOW_t * state)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= rows ) return;

    uint32_t visited[128];
    uint32_t p[128];

    for (uint32_t i = 0; i < cols; ++i) {
        visited[i] = 1;
    }

    int k = cols * randXOR(state + idx);
    visited[k] = 0;
    tabu[idx * cols] = k;

    for (uint32_t s = 1; s < cols; ++s) {

        float sum = 0.0f;
        const uint32_t i = k;
        for (uint32_t j = 0; j < cols; ++j) {
            sum += fitness[i * cols + j] * visited[j];
            p[j] = sum;
        }

        const float r = randXOR(state + idx) * sum;
        k = -1;
        for (uint32_t j = 0; j < cols; ++j) {
            k += (k == -1) * (p[j] > r) * (j + 1);
        }
        visited[k] = 0;
        tabu[idx * cols + s] = k;
    }
}

// __device__ __forceinline__ 
// float atomicMaxFloat (float * addr, float value) {
//     float old;
//     old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
//          __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

//     return old;
// }

__device__ __forceinline__
float scanTileFloat(const thread_block_tile<32> & g, float x) {
    #pragma unroll
    for( uint32_t offset = 1 ; offset < 32 ; offset <<= 1 ) {
        float y = g.shfl_up(x, offset);
        if(g.thread_rank() >= offset) x += y;
    }
    return x;
}

__device__ __forceinline__
float maxTileFloat(const thread_block_tile<32> & g, float x) {
    
    #pragma unroll
    for ( uint32_t offset = 16; offset > 0; offset >>= 1 ) {
        const float y = g.shfl_xor(x, offset);
        x = fmaxf(x, y);
    }
    return x;
}

#define TABU_SHARED_MEMORY 1

__global__
void claculateTourWarp(uint32_t * tabu,
                       const float * fitness,
                       const uint32_t rows,
                       const uint32_t cols,
                       curandStateXORWOW_t * state)
{

#if TABU_SHARED_MEMORY
    extern __shared__ uint32_t smem[];
    uint32_t * visited = (uint32_t *)  smem;
    uint32_t * tabus   = (uint32_t *) &visited[cols];
    float    * p       = (float *)    &tabus[cols];
    uint32_t * k       = (uint32_t *) &p[cols];
    float    * reduce  = (float *)    &k[1];
#else 
    extern __shared__ uint32_t smem[];
    uint32_t * visited = (uint32_t *)  smem;
    float    * p       = (float *)    &visited[cols];
    uint32_t * k       = (uint32_t *) &p[cols];
    float    * reduce  = (float *)    &k[1];
#endif

    thread_block_tile<32> tile32 = tiled_partition<32>(this_thread_block());
    const uint32_t tid = threadIdx.x;

    // initialize visited array
    for (uint32_t i = tid; i < cols; i += blockDim.x) { 
        visited[i] = 1;
    }
    __syncthreads();

    // get random starting city and update visited and tabu
    if (tid == 0) {
        *k = cols * randXOR(state + blockIdx.x);
        visited[*k] = 0;
#if TABU_SHARED_MEMORY
        tabus[0] = *k;
#else
        tabu[blockIdx.x * cols] = *k;
#endif
    }
    __syncthreads();

    

    for (uint32_t s = 1; s < cols; ++s) {
        // get city from shared memory
        const uint32_t kappa = *k;

        // update probability values
        for (uint32_t i = tid; i < cols; i += blockDim.x) { 
            p[i] = fitness[kappa * cols + i] * visited[i];
        }
        __syncthreads();

        const uint32_t numberOfBlocks = (cols + 31) / 32;

        for (uint32_t blockId = tid / 32; blockId < numberOfBlocks; blockId += blockDim.x / 32) {
            const uint32_t warpTid = tile32.thread_rank() + (blockId * 32);
            
            const float x = (warpTid < cols) ? p[warpTid] : 0.f;
            const float y = scanTileFloat(tile32, x);
            const float z = tile32.shfl(y, 31);
            if (warpTid < cols) p[warpTid] = y / z;
            if (tile32.thread_rank() == 0) reduce[blockId] = z;
        }

        __syncthreads();

        if (tid < 32) {

            uint32_t selectedBlock = 1234567890; // fake number just to be sure that will not appear somewere
            float selectedMax = -1.f;

            //TODO: verify if selectedBlock is correct when cols >= 1024
            for (uint32_t stride = 0; stride < numberOfBlocks; stride += 32) {
                const uint32_t warpTid = tid + (stride * 32);
                const float x = (warpTid < numberOfBlocks) ? reduce[warpTid] : 0.f;
                selectedMax = fmaxf(x, selectedMax);
                selectedBlock = (x == selectedMax) ? warpTid : selectedBlock;

                const float y = maxTileFloat(tile32, selectedMax);
                const uint32_t mask = tile32.ballot( x == y );
                const uint32_t maxTile = __ffs(mask) - 1;
                selectedMax = tile32.shfl(y, maxTile);
                selectedBlock = tile32.shfl(selectedBlock, maxTile);
            }

            // generate and broadcast randomFloat
            float randomFloat = -1.f;
            if (tid == 0) {
                randomFloat = randXOR(state + blockIdx.x);
            }
            randomFloat = tile32.shfl(randomFloat, 0);
            
            const uint32_t probabilityId = selectedBlock * 32 + tid;
            if (selectedBlock * 32 + tid < cols) {
                const uint32_t bitmask = tile32.ballot(randomFloat < p[probabilityId]); 
                const uint32_t selected = __ffs(bitmask) - 1;

                if (tid == selected) {
                    const uint32_t nextCity = selectedBlock * 32 + selected;
#if TABU_SHARED_MEMORY
                    tabus[s] = nextCity;
#else
                    tabu[blockIdx.x * cols + s] = nextCity;
#endif
                    visited[nextCity] = 0;
                    *k = nextCity;
                }
            }
        }
        __syncthreads();
    }
#if TABU_SHARED_MEMORY
    for (uint32_t i = tid; i < cols; i += blockDim.x) { 
        tabu[blockIdx.x * cols + i] = tabus[i];
    }
#endif
}

__global__
void calculateTourLen(const uint32_t * tabu,
                      const float * distance,
                      uint32_t * tourLen,
                      const uint32_t rows,
                      const uint32_t cols)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= rows ) return;

    float length = 0.0f;
    for (uint32_t i = 0; i < cols - 1; ++i) {
        const uint32_t from = tabu[idx * cols + i];
        const uint32_t to = tabu[idx * cols + i + 1];
        length += distance[from * cols + to];
    }

    const uint32_t from = tabu[idx * cols + cols - 1];
    const uint32_t to = tabu[idx * cols];
    tourLen[idx] = length + distance[from * cols + to];
}

__global__
void updateBest(uint32_t * bestPath,
                const uint32_t * tabu,
                const uint32_t * tourLen,
                const uint32_t rows,
                const uint32_t cols,
                uint32_t * bestPathLen,
                const uint32_t last)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= rows ) return;

    const uint32_t len = tourLen[idx];
    atomicMin(bestPathLen, len);

    __syncthreads();

    if (*bestPathLen == len) {
        for (uint32_t i = 0; i < cols; ++i) {
            bestPath[i] = tabu[idx * cols + i];
        }
    }
}

__global__
void updateDelta(float * delta,
                 const uint32_t * tabu,
                 const uint32_t * tourLen,
                 const uint32_t rows,
                 const uint32_t cols,
                 const float q)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= rows ) return;

    for (uint32_t i = 0; i < cols - 1; ++i) {
        const uint32_t from = tabu[idx * cols + i];
        const uint32_t to = tabu[idx * cols + i + 1];
        atomicAdd(delta + (from * cols + to), q / tourLen[idx]);
        // atomicAdd(delta + (to * cols + from), q / tourLen[idx]);
    }

    const uint32_t from = tabu[idx * cols + cols - 1];
    const uint32_t to = tabu[idx * cols];
    atomicAdd(delta + (from * cols + to), q / tourLen[idx]);
    // atomicAdd(delta + (to * cols + from), q / tourLen[idx]);
}

__global__
void updatePheromone(float * pheromone,
                     float * delta,
                     const uint32_t rows,
                     const uint32_t cols,
                     const float rho)
{
    const uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if ( row >= rows || col >= cols ) return;

    const uint32_t idx = row * cols + col;

    const float p = pheromone[idx];
    pheromone[idx] = p * (1.0f - rho) + delta[idx];
    delta[idx] = 0.0f;
}


uint32_t numberOfBlocks(uint32_t numberOfElements, uint32_t blockSize) {
    return (numberOfElements + blockSize - 1) / blockSize;
}

uint32_t roundWithBlockSize(uint32_t numberOfElements, uint32_t blockSize)
{
    return numberOfBlocks(numberOfElements, blockSize) * blockSize; 
}

#ifndef D_TYPE
#define D_TYPE float
#endif


int main(int argc, char * argv[]) {

	char * path = new char[MAX_LEN];
	D_TYPE alpha = 4.0f;
	D_TYPE beta = 2.0f;
	D_TYPE q = 55.0f;
	D_TYPE rho = 0.8f;
	int maxEpoch = 10;
	
	if (argc < 7) {
		cout << "Usage: ./acogpu file.tsp alpha beta q rho maxEpoch" << endl;
		exit(-1);
	}
	
    uint32_t seed = time(0);//123;
    
	argc--;
	argv++;
	uint32_t args = 0;
	strArg(argc, argv, args++, path);
	fltArg(argc, argv, args++, &alpha);
	fltArg(argc, argv, args++, &beta);
	fltArg(argc, argv, args++, &q);
	fltArg(argc, argv, args++, &rho);
	intArg(argc, argv, args++, &maxEpoch);

	TSP<D_TYPE> * tsp = new TSP<D_TYPE>(path);
	ACO<D_TYPE> * aco = new ACO<D_TYPE>(tsp->dimension, tsp->dimension, alpha, beta, q, rho, maxEpoch, 0);

    uint32_t nAnts = aco->nAnts;
    uint32_t nCities = tsp->dimension;
    float valPheromone = 1.0f / nCities;

    curandStateXORWOW_t * state;
    float * distance;
    float * eta;
    float * pheromone;
    float * fitness;
    float * delta;
    uint32_t * tabu;
    uint32_t * tourLen;
    uint32_t * bestPath;
    uint32_t * bestPathLen;

    uint32_t elems = nCities * nCities;
    cudaMallocManaged(&state,       nAnts * sizeof(curandStateXORWOW_t));
    cudaMallocManaged(&distance,    elems * sizeof(float));
    cudaMallocManaged(&eta,         elems * sizeof(float));
    cudaMallocManaged(&pheromone,   elems * sizeof(float));
    cudaMallocManaged(&fitness,     elems * sizeof(float));
    cudaMallocManaged(&delta,       elems * sizeof(float));
    cudaMallocManaged(&tabu,        nAnts * nCities * sizeof(uint32_t));
    cudaMallocManaged(&tourLen,     nAnts * sizeof(uint32_t));
    cudaMallocManaged(&bestPath,    nCities * sizeof(uint32_t));
    cudaMallocManaged(&bestPathLen, sizeof(uint32_t));

    *bestPathLen = INT_MAX;

    for (uint32_t i = 0; i < nCities; ++i) {
        for (uint32_t j = 0; j < nCities; ++j) {
            distance[i * nCities + j] = tsp->edges[i * nCities +j];
        }
    }

    dim3 dimBlockSmall(8);
    dim3 dimBlock1D(32);
    dim3 dimBlock2D(16, 16);

    dim3 gridAntSmall(numberOfBlocks(nAnts, dimBlockSmall.x));
    dim3 gridAnt1D(numberOfBlocks(nAnts, dimBlock1D.x));
    // dim3 gridAnt2D(numberOfBlocks(nAnts, dimBlock2D.y), numberOfBlocks(nCities, dimBlock2D.x));
    dim3 gridMatrix2D(numberOfBlocks(nCities, dimBlock2D.y), numberOfBlocks(nCities, dimBlock2D.x));

    startTimer();

    initCurand<<<gridAnt1D, dimBlock1D>>>(state, seed, nAnts);
    initialize<<<gridMatrix2D, dimBlock2D>>>(distance, eta, pheromone, delta, valPheromone, nCities, nCities);

    uint32_t epoch = 0;
    do {
        calculateFitness<<<gridMatrix2D, dimBlock2D>>>(fitness, pheromone, eta, alpha, beta, nCities, nCities);

        dim3 gridTour(1);
        dim3 dimBlockTour(1);

        switch (TOUR_SELECTED) {

            case TOUR_PRIVATE:
                gridTour.x = 64;
                dimBlockTour.x = (nAnts + gridTour.x - 1) / gridTour.x;
                claculateTourPrivate<<<gridTour, dimBlockTour>>>(tabu, fitness, nAnts, nCities, state);
            break;

            case TOUR_PRIVATE_2:
                gridTour.x = 64;
                dimBlockTour.x = (nAnts + gridTour.x - 1) / gridTour.x;
                claculateTourPrivate_2<<<gridTour, dimBlockTour>>>(tabu, fitness, nAnts, nCities, state);
            break;

            case TOUR_WARP:
                gridTour.x = nCities;         // number of blocks
                dimBlockTour.x = 128;         // number of threads in a block
                const uint32_t numberOfWarps = dimBlockTour.x / 32;
                // const uint32_t sharedSize = nCities       * sizeof(uint8_t)  + // visited
                //                             nCities       * sizeof(uint32_t) + // tabu
                //                             nCities       * sizeof(float)    + // p
                //                             1             * sizeof(uint32_t) + // k
                //                             numberOfWarps * sizeof(float);     //reduce

                const uint32_t sharedSize = (nAnts * 3 + 1 + numberOfWarps) * sizeof(uint32_t);

                claculateTourWarp<<<gridTour, dimBlockTour, sharedSize >>>(tabu, fitness, nAnts, nCities, state);
            break;
        }

        calculateTourLen<<<gridAnt1D, dimBlock1D>>>(tabu, distance, tourLen, nAnts, nCities);
        updateBest<<<gridAnt1D, dimBlock1D>>>(bestPath, tabu, tourLen, nAnts, nCities, bestPathLen, ((epoch + 1) == maxEpoch));
        updateDelta<<<gridAnt1D, dimBlock1D>>>(delta, tabu, tourLen, nAnts, nCities, q);
        updatePheromone<<<gridMatrix2D, dimBlock2D>>>(pheromone, delta, nCities, nCities, rho);

    } while (++epoch < maxEpoch);

    cudaDeviceSynchronize();

    stopAndPrintTimer();

    cout << (tsp->checkPath((int *)bestPath) == 1 ? "Path OK!" : "Error in the path!") << endl;
    cout << "bestPathLen: " << *bestPathLen << endl;
    cout << "CPU Path distance: " << tsp->calculatePathLen((int *)bestPath) << endl;
    printMatrix("bestPath", bestPath, 1, nCities);


    cudaFree(state);
    cudaFree(distance);
    cudaFree(eta);
    cudaFree(pheromone);
    cudaFree(fitness);
    cudaFree(delta);
    cudaFree(tabu);
    cudaFree(tourLen);
    cudaFree(bestPath);
    cudaFree(bestPathLen);

    return 0;
}
