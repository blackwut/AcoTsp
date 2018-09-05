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


#define DEBUG 0

__global__ 
void initCurand(curandStateXORWOW_t * state,
                const unsigned long seed,
                const uint32_t nAnts)
{
    const uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if ( idx >= nAnts ) return;

    curand_init(seed, idx, 0, &state[idx]);
}

__device__ __forceinline__
float randXOR(curandState * state)
{
#if DEBUG 
    return 0.5f;
#else 
    return (float) curand_uniform(state);
#endif
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

#define TABU_SHARED_MEMORY 0

__global__
void claculateTour(uint32_t * tabu,
                   const float * fitness,
                   const uint32_t rows,
                   const uint32_t cols,
                   const uint32_t strideShared,
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
        const uint32_t kappa = cols * randXOR(state + blockIdx.x);
        *k = kappa;
        visited[kappa] = 0;
#if TABU_SHARED_MEMORY
        tabus[0] = kappa;
#else
        tabu[blockIdx.x * cols] = kappa;
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
                const uint32_t warpTid = tid + stride;
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
            if (probabilityId < cols) {
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

__device__ __forceinline__
float reduceTileFloat(const thread_block_tile<32> & g, float x) {
    
    #pragma unroll
    for ( uint32_t offset = 16; offset > 0; offset >>= 1 ) {
        x += g.shfl_down(x, offset);
    }
    return x;
}

__global__
void calculateTourLen(const float * distance,
                      const uint32_t * tabu,
                      float * tourLen,
                      const uint32_t rows,
                      const uint32_t cols)
{
    __shared__ float finalLength[1];

    thread_block_tile<32> tile32 = tiled_partition<32>(this_thread_block());
    const uint32_t numberOfBlocks = (cols + 31) / 32;

    float totalLength = 0.f;
    for (uint32_t blockId = threadIdx.x / 32; blockId < numberOfBlocks; blockId += blockDim.x / 32) {
        const uint32_t warpTid = blockIdx.x * cols + tile32.thread_rank() + (blockId * 32);

        float len = 0.f;
        if ( tile32.thread_rank() + (blockId * 32) < cols - 1 ) {
            const uint32_t from = tabu[warpTid];
            const uint32_t to   = tabu[warpTid + 1];
            len  = distance[from * cols + to];
        }
        totalLength += reduceTileFloat(tile32, len);
    }

    if (threadIdx.x == 0) {
        const uint32_t from = tabu[blockIdx.x * cols + cols - 1];
        const uint32_t to   = tabu[blockIdx.x * cols];
        const float    len  = distance[from * cols + to];
        
        totalLength += len;

        finalLength[0] = 0.f;
    }

    __syncthreads();

    if (tile32.thread_rank() == 0) {
        atomicAdd(finalLength, totalLength);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        tourLen[blockIdx.x] = finalLength[0];
    }
}

__device__ __forceinline__
float minTileFloat(const thread_block_tile<32> & g, float x) {
    
    #pragma unroll
    for ( uint32_t offset = 16; offset > 0; offset >>= 1 ) {
        const float y = g.shfl_xor(x, offset);
        x = fminf(x, y);
    }
    return x;
}

__global__
void updateBest(uint32_t * bestPath,
                const uint32_t * tabu,
                const float * tourLen,
                const uint32_t rows,
                const uint32_t cols,
                float * bestPathLen,
                const uint32_t last)
{
    thread_block_tile<32> tile32 = tiled_partition<32>(this_thread_block());
    const uint32_t tid = threadIdx.x;

    uint32_t bestAnt = 1234567890; // fake number just to be sure that will not appear somewere
    float minLength = FLT_MAX;

    //TODO: verify if bestAnt is correct when cols >= 1024
    for (uint32_t stride = 0; stride < cols; stride += 32) {
        const uint32_t warpTid = tid + stride;
        const float x = (warpTid < cols) ? tourLen[warpTid] : FLT_MAX;
        minLength = fminf(x, minLength);
        bestAnt = (x == minLength) ? warpTid : bestAnt;

        const float y = minTileFloat(tile32, minLength);
        const uint32_t mask = tile32.ballot( x == y );
        const uint32_t maxTile = __ffs(mask) - 1;
        minLength = tile32.shfl(y, maxTile);
        bestAnt = tile32.shfl(bestAnt, maxTile);
    }

    for (uint32_t i = tid; i < cols; i += 32) {
        bestPath[i] = tabu[bestAnt * cols + i];
    }

    if (tid == 0) {
        bestPathLen[0] = minLength;
    }
}

__global__
void updateDelta(float * delta,
                 const uint32_t * tabu,
                 const float * tourLen,
                 const uint32_t rows,
                 const uint32_t cols,
                 const float q)
{
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= rows ) return;

    const float tau = q / tourLen[idx];

    for (uint32_t i = 0; i < cols - 1; ++i) {
        const uint32_t from = tabu[idx * cols + i];
        const uint32_t to = tabu[idx * cols + i + 1];
        atomicAdd(delta + (from * cols + to), tau);
        // atomicAdd(delta + (to * cols + from), q / tourLen[idx]);
    }

    const uint32_t from = tabu[idx * cols + cols - 1];
    const uint32_t to = tabu[idx * cols];
    atomicAdd(delta + (from * cols + to), tau);
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

	argc--;
	argv++;
    path     = argv[0];
    alpha    = parseArg<float>   (argv[1]);
    beta     = parseArg<float>   (argv[2]);
    q        = parseArg<float>   (argv[3]);
    rho      = parseArg<float>   (argv[4]);
    maxEpoch = parseArg<uint32_t>(argv[5]);

	TSP<D_TYPE> * tsp = new TSP<D_TYPE>(path);
	ACO<D_TYPE> * aco = new ACO<D_TYPE>(tsp->dimension, tsp->dimension, alpha, beta, q, rho, maxEpoch, 0);

#if DEBUG
    const uint32_t seed         = 123;
#else 
    const uint32_t seed         = time(0);
#endif
    const uint32_t nAnts        = aco->nAnts;
    const uint32_t nCities      = tsp->dimension;
    const float    valPheromone = 1.0f / nCities;

    curandStateXORWOW_t * state;
    float * distance;
    float * eta;
    float * pheromone;
    float * fitness;
    float * delta;
    uint32_t * tabu;
    float * tourLen;
    uint32_t * bestPath;
    float * bestPathLen;

    uint32_t elems = nCities * nCities;
    cudaMallocManaged(&state,       nAnts * sizeof(curandStateXORWOW_t));
    cudaMallocManaged(&distance,    elems * sizeof(float));
    cudaMallocManaged(&eta,         elems * sizeof(float));
    cudaMallocManaged(&pheromone,   elems * sizeof(float));
    cudaMallocManaged(&fitness,     elems * sizeof(float));
    cudaMallocManaged(&delta,       elems * sizeof(float));
    cudaMallocManaged(&tabu,        nAnts * nCities * sizeof(uint32_t));
    cudaMallocManaged(&tourLen,     nAnts * sizeof(float));
    cudaMallocManaged(&bestPath,    nCities * sizeof(uint32_t));
    cudaMallocManaged(&bestPathLen, sizeof(float));

    uint32_t totalMemory = (2 * nAnts + elems * 5 + nAnts * nCities + nCities) * sizeof(uint32_t);
    std::cout << " *** totalMemory **** \t" << (totalMemory / 1024.f) / 1024.f << "MB" << std::endl;


    *bestPathLen = INT_MAX;

    for (uint32_t i = 0; i < nCities; ++i) {
        for (uint32_t j = 0; j < nCities; ++j) {
            distance[i * nCities + j] = tsp->edges[i * nCities +j];
        }
    }

    dim3 dimBlock1D(32);
    dim3 dimBlock2D(16, 16);

    dim3 gridAnt1D(numberOfBlocks(nAnts, dimBlock1D.x));
    // dim3 gridAnt2D(numberOfBlocks(nAnts, dimBlock2D.y), numberOfBlocks(nCities, dimBlock2D.x));
    dim3 gridMatrix2D(numberOfBlocks(nCities, dimBlock2D.y), numberOfBlocks(nCities, dimBlock2D.x));

    startTimer();

    initCurand<<<gridAnt1D, dimBlock1D>>>(state, seed, nAnts);
    initialize<<<gridMatrix2D, dimBlock2D>>>(distance, eta, pheromone, delta, valPheromone, nCities, nCities);

    const uint32_t visitedStride = ((nCities + 31) / 32) * 32;

    const dim3 tourGrid(nCities);         // number of blocks
    const dim3 tourDimBlock(128);         // number of threads in a block
    const uint32_t tourWarps = (nCities + 31) / 32;//tourDimBlock.x / 32;
    const uint32_t tourShared = nCities   * sizeof(uint32_t) + // visited
#if TABU_SHARED_MEMORY
                                nCities   * sizeof(uint32_t) + // tabu
#endif
                                nCities   * sizeof(float)    + // p
                                1         * sizeof(uint32_t) + // k
                                tourWarps * sizeof(float);     //reduce

    std::cout << " *** sharedMemory **** \t" << (tourShared / 1024.f) << "KB" << std::endl;


    const dim3 lenGrid(nAnts);
    const dim3 lenDimBlock(64);
    const uint32_t lenShared = lenDimBlock.x / 32 * sizeof(float);
    uint32_t epoch = 0;
    do {
        calculateFitness <<<gridMatrix2D, dimBlock2D>>>(fitness, pheromone, eta, alpha, beta, nCities, nCities);
        claculateTour    <<<tourGrid,     tourDimBlock, tourShared >>>(tabu, fitness, nAnts, nCities, visitedStride, state);
        calculateTourLen <<<lenGrid,      lenDimBlock,  lenShared  >>>(distance, tabu, tourLen, nAnts, nCities);
        updateBest       <<<1, 32>>>(bestPath, tabu, tourLen, nAnts, nCities, bestPathLen, ((epoch + 1) == maxEpoch));
        updateDelta      <<<gridAnt1D,    dimBlock1D>>>(delta, tabu, tourLen, nAnts, nCities, q);
        updatePheromone  <<<gridMatrix2D, dimBlock2D>>>(pheromone, delta, nCities, nCities, rho);
    } while (++epoch < maxEpoch);

    cudaDeviceSynchronize();

    stopAndPrintTimer();

    // printMatrix("tourLen", tourLen, 1, nAnts);
    cout << (tsp->checkPath(bestPath) == 1 ? "Path OK!" : "Error in the path!") << endl;
    cout << "bestPathLen: " << *bestPathLen << endl;
    cout << "CPU Path distance: " << tsp->calculatePathLen(bestPath) << endl;
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
