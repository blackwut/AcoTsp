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

#include <thread>
#include <chrono>
#include <assert.h>

#include <cooperative_groups.h>
using namespace cooperative_groups;


#define cudaCheck(ans) do { cudaAssert((ans), __FILE__, __LINE__); } while(0)
inline void cudaAssert(cudaError_t code, const char * file, uint32_t line, bool abort = true)
{
    if (code != cudaSuccess) {
        std::clog <<  "cudaErrorAssert: "<< cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort) { 
            exit(code);
        }
    }
}

#define DEBUG 1

__global__ 
void initCurand(curandStateXORWOW_t * state,
                const uint64_t seed,
                const uint32_t elems)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint32_t c = tid; c < elems; c += gridDim.x * blockDim.x) {
        curand_init(seed, c, 0, &state[c]);
    }
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
void initEta(float * eta,
             const float * distance,
             const uint32_t rows,
             const uint32_t cols)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = tid; c < cols; c += gridDim.x * blockDim.x) {
            const uint32_t id = r * cols + c;
            const float d = distance[id];
#if DEBUG
            eta[id] = (d == 0.f) ? 0.f : 0.1f;
#else 
            eta[id] = (d == 0.f) ? 0.f : 1.f / d;
#endif
        }
    }
}

__global__
void initPheromone(float * pheromone,
                   const float initialValue,
                   const uint32_t rows,
                   const uint32_t cols)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = tid; c < cols; c += gridDim.x * blockDim.x) {
            const uint32_t id = r * cols + c;
            pheromone[id] = initialValue;
        }
    }
}

__global__
void initDelta(float * delta,
               const uint32_t rows,
               const uint32_t cols)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = tid; c < cols; c += gridDim.x * blockDim.x) {
        const uint32_t id = r * cols + c;
            delta[id] = 0.f;
        }
    }
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
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = tid; c < cols; c += gridDim.x * blockDim.x) {
            const uint32_t id = r * cols + c;
            const float p = pheromone[id];
            const float e = eta[id];
            fitness[id] = __powf(p, alpha) * __powf(e, beta);
        }
    }
}

#define FULL_MASK 0xFFFFFFFF
__device__ __forceinline__
float scanWarpFloat(const uint32_t tid, float x) {
    #pragma unroll
    for( uint32_t offset = 1 ; offset < 32 ; offset <<= 1 ) {
        const float y = __shfl_up_sync(FULL_MASK, x, offset);
        if(tid >= offset) x += y;
    }
    return x;
}

__global__
void claculateTour(uint32_t * tabu,
                   const float * fitness,
                   const uint32_t rows,
                   const uint32_t cols,
                   const uint32_t alignedCols,
                   curandStateXORWOW_t * state)
{
    extern __shared__ uint32_t smem[];
    float    * p = (float *)     smem;
    uint32_t * k = (uint32_t *) &p[alignedCols];
    uint8_t  * v = (uint8_t *)  &k[1]; 

    const uint32_t tid = threadIdx.x;

    // initialize visited array
    for (uint32_t i = tid; i < alignedCols; i += 32) { 
        v[i] = 1;
    }
    __syncwarp();

    // get random starting city and update visited and tabu
    if (tid == 0) {
#if DEBUG
        const uint32_t kappa = 0;
#else
        const uint32_t kappa = cols * randXOR(state + blockIdx.x);
#endif
        *k = kappa;
        v[kappa] = 0;
        tabu[blockIdx.x * alignedCols] = kappa;
    }
    

    for (uint32_t s = 1; s < cols; ++s) {
        __syncwarp(); // sync warp once for tabu initialization and then for *k value update
        // get city from shared memory
        const uint32_t kappa = *k;

        // update probability values
        for (uint32_t pid = tid; pid < alignedCols; pid += 32) {
            p[pid] = fitness[kappa * alignedCols + pid] * v[pid];
        }
        __syncwarp();

        float sum = 0.f;
        for (uint32_t pid = tid; pid < alignedCols; pid += 32) {
            const float x = p[pid];
            const float y = sum + scanWarpFloat(tid, x);
            p[pid] = y;
            sum = __shfl_sync(FULL_MASK, y, 31);
            // printf("%d) %f - %f - %f\n", pid, x, y, sum);
        }

        __syncwarp();

        // generate and broadcast randomFloat
        float randomFloat = -1.f;
        if (tid == 0) {
            randomFloat = randXOR(state + blockIdx.x);
        }
        randomFloat = __shfl_sync(FULL_MASK, randomFloat, 0);

        const float probability = randomFloat * sum;
        for (uint32_t pid = tid; pid < alignedCols; pid += 32) {
            
            const float prevP = (pid == 0 ? 0.f : p[pid - 1]);
            const float currP = p[pid];
            const float magicProbability = (prevP - probability) * (currP - probability);
            const uint32_t ballotMask = __ballot_sync(FULL_MASK, magicProbability <= 0.f);
            const uint32_t winner = __ffs(ballotMask);

            if (winner > 0) {
                if (tid == winner - 1) {
                    tabu[blockIdx.x * alignedCols + s] = pid;
                    v[pid]= 0;
                    *k = pid;
                }
                break;
            }
        }
    }
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
                      const uint32_t cols,
                      const uint32_t realCols)
{
    __shared__ float finalLength[1];

    thread_block_tile<32> tile32 = tiled_partition<32>(this_thread_block());
    const uint32_t numberOfBlocks = (cols + 31) / 32;

    float totalLength = 0.f;
    for (uint32_t blockId = threadIdx.x / 32; blockId < numberOfBlocks; blockId += blockDim.x / 32) {
        const uint32_t warpTid = blockIdx.x * cols + tile32.thread_rank() + (blockId * 32);

        float len = 0.f;
        if (tile32.thread_rank() + (blockId * 32) < realCols - 1) {
            const uint32_t from = tabu[warpTid];
            const uint32_t to   = tabu[warpTid + 1];
            len  = distance[from * cols + to];
        }
        totalLength += reduceTileFloat(tile32, len);
    }

    if (threadIdx.x == 0) {
        const uint32_t from = tabu[blockIdx.x * cols + realCols - 1];
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
                const uint32_t realCols,
                float * bestPathLen)
{
    thread_block_tile<32> tile32 = tiled_partition<32>(this_thread_block());
    const uint32_t tid = threadIdx.x;

    uint32_t bestAnt = 1234567890; // fake number just to be sure that will not appear somewere
    float minLength = FLT_MAX;

    //TODO: verify if bestAnt is correct when cols >= 1024
    for (uint32_t stride = 0; stride < cols; stride += 32) {
        const uint32_t warpTid = tid + stride;
        const float x = (warpTid < realCols) ? tourLen[warpTid] : FLT_MAX; //TODO: find a way to avoid realCols parameter
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
                 const uint32_t realCols,
                 const float q)
{
    extern __shared__ uint32_t tabus[];
    const uint32_t tid = threadIdx.x;

    for (uint32_t i = tid; i < realCols; i += blockDim.x) { //TODO: verifica se c'è un degrado di prestazioni se invece di cols usi realCols in tutto il kernel
        tabus[i] = tabu[blockIdx.x * cols + i];
    }
    __syncthreads();

    const float tau = q / tourLen[blockIdx.x];

    for (uint32_t i = tid; i < realCols - 1; i += blockDim.x) { 
        const uint32_t from = tabus[i];
        const uint32_t to   = tabus[i + 1];
        atomicAdd(delta + (from * cols + to), tau);
    }

    if (tid == 0) {
        const uint32_t from = tabus[realCols - 1];
        const uint32_t to   = tabus[0];
        atomicAdd(delta + (from * cols + to), tau);
    }
}

__global__
void updatePheromone(float * pheromone,
                     float * delta,
                     const uint32_t rows,
                     const uint32_t cols,
                     const float rho)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = tid; c < cols; c += gridDim.x * blockDim.x) {
            const uint32_t id = r * cols + c;
            const float p = pheromone[id];
            pheromone[id] = p * (1.f  - rho) + delta[id];
            delta[id] = 0.f;
        }
    }
}

uint32_t numberOfBlocks(uint32_t elems, uint32_t blockSize) {
    return (elems + blockSize - 1) / blockSize;
}

uint32_t roundWithBlockSize(uint32_t elems, uint32_t blockSize)
{
    return numberOfBlocks(elems, blockSize) * blockSize; 
}

#ifndef D_TYPE
#define D_TYPE float
#endif


int main(int argc, char * argv[]) {

	char * path = new char[MAX_LEN];
	D_TYPE alpha = 1.0f;
	D_TYPE beta = 2.0f;
	D_TYPE q = 1.0f;
	D_TYPE rho = 0.5f;
	int maxEpoch = 1;
	
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
    const uint64_t seed         = 123;
#else 
    const uint64_t seed         = time(0);
#endif
    const uint32_t nAnts        = aco->nAnts;
    const uint32_t nCities      = tsp->dimension;
    const float    valPheromone = 1.0f / nCities;

    curandStateXORWOW_t * randState;
    float * distance;
    float * eta;
    float * pheromone;
    float * fitness;
    float * delta;
    uint32_t * tabu;
    float * tourLen;
    uint32_t * bestPath;
    float * bestPathLen;

    const uint32_t alignedAnts = ((nAnts + 31) / 32) * 32;
    const uint32_t alignedCities = ((nCities + 31) / 32) * 32;

    std::cout << "alignedAnts:   " << alignedAnts   << std::endl;
    std::cout << "alignedCities: " << alignedCities << std::endl;


    const uint32_t randStateElems = alignedAnts;
    const uint32_t distanceElems  = nCities * alignedCities;
    const uint32_t etaElems       = nCities * alignedCities;
    const uint32_t pheromoneElems = nCities * alignedCities;
    const uint32_t fitnessElems   = nCities * alignedCities;
    const uint32_t deltaElems     = nCities * alignedCities;
    const uint32_t tabuElems      = nAnts   * alignedCities;
    const uint32_t tourLenElems  = alignedAnts;
    const uint32_t bestPathElems  = alignedCities;

    cudaCheck( cudaMallocManaged(&randState,   randStateElems * sizeof(curandStateXORWOW_t)) );
    cudaCheck( cudaMallocManaged(&distance,    distanceElems  * sizeof(float))               );
    cudaCheck( cudaMallocManaged(&eta,         etaElems       * sizeof(float))               );
    cudaCheck( cudaMallocManaged(&pheromone,   pheromoneElems * sizeof(float))               );
    cudaCheck( cudaMallocManaged(&fitness,     fitnessElems   * sizeof(float))               );
    cudaCheck( cudaMallocManaged(&delta,       deltaElems     * sizeof(float))               );
    cudaCheck( cudaMallocManaged(&tabu,        tabuElems      * sizeof(uint32_t))            );
    cudaCheck( cudaMallocManaged(&tourLen,     tourLenElems   * sizeof(float))               );
    cudaCheck( cudaMallocManaged(&bestPath,    bestPathElems  * sizeof(uint32_t))            );
    cudaCheck( cudaMallocManaged(&bestPathLen, sizeof(float))                                );

    const uint32_t totalMemory = (randStateElems * sizeof(float)    +
                                  distanceElems  * sizeof(float)    +
                                  etaElems       * sizeof(float)    +
                                  pheromoneElems * sizeof(float)    +
                                  fitnessElems   * sizeof(float)    +
                                  deltaElems     * sizeof(float)    +
                                  tabuElems      * sizeof(uint32_t) +
                                  tourLenElems   * sizeof(float)    +
                                  bestPathElems  * sizeof(uint32_t) +
                                  1              * sizeof(float));

    std::cout << " **** ACO TSP totalMemory **** \tMB " << (totalMemory / 1024.f) / 1024.f<< std::endl;

    *bestPathLen = FLT_MAX;

    for (uint32_t i = 0; i < nCities; ++i) {
        for (uint32_t j = 0; j < alignedCities; ++j) {
            const uint32_t edgeId = i * nCities + j;
            distance[i * alignedCities + j] = (j < nCities) ? tsp->edges[edgeId] : 0.f;
        }
    }

#if DEBUG
    float    * _distance  = (float *)    malloc(distanceElems  * sizeof(float));
    float    * _eta       = (float *)    malloc(etaElems       * sizeof(float));
    float    * _pheromone = (float *)    malloc(pheromoneElems * sizeof(float));
    float    * _fitness   = (float *)    malloc(fitnessElems   * sizeof(float));
    float    * _delta     = (float *)    malloc(deltaElems     * sizeof(float));
    uint32_t * _tabu      = (uint32_t *) malloc(tabuElems      * sizeof(uint32_t));
    float    * _tourLen   = (float *)    malloc(tourLenElems   * sizeof(float));
    uint32_t * _bestPath  = (uint32_t *) malloc(bestPathElems  * sizeof(uint32_t));
    float _bestPathLen;

    for (uint32_t i = 0; i < nCities; ++i) {
        for (uint32_t j = 0; j < alignedCities; ++j) {
            const uint32_t edgeId = i * nCities + j;
            const uint32_t id = i * alignedCities +j;
            const float d = (j < nCities) ? tsp->edges[edgeId] : 0.f;
            _distance[id] = d;
            _eta[id] = (d == 0.f) ? 0.f : 1.f / d;
            _pheromone[id] = valPheromone;
            _delta[id] = 0;
        }
        _tourLen[i] = 0.f;
        _bestPath[i] = 0;
    }
    _bestPathLen = FLT_MAX;

    for (uint32_t i = 0; i < nCities; ++i) {
        for (uint32_t j = 0; j < alignedCities; ++j) {
             const uint32_t id = i * alignedCities +j;
            _fitness[id] = powf(_pheromone[id], alpha) * powf(_eta[id], beta);
        }
    }

    uint32_t * v = new uint32_t[nCities];
    float * p = new float[nCities];

    for (uint32_t i = 0; i < nAnts; ++i) {

        for (uint32_t j = 0; j < nCities; ++j) {
            v[j] = 1;
            p[j] = 0.f;
        }

        uint32_t k = 0.5 * nCities;
        v[k] = 0;
        _tabu[i * alignedCities] = k;

        for (uint32_t s = 1; s < nCities; ++s) {
            
            float sum = 0.f;
            for (uint32_t j = 0; j < nCities; ++j) {
                sum += _fitness[k * alignedCities + j] * v[j];
                p[j] = sum;
            }

            const uint32_t r = 0.5 * sum;

            for (uint32_t j = 0; j < nCities; ++j) {
                const float prevP = (j == 0) ? 0.f : p[j - 1];
                const float currP = p[j];
                const float magicNumber = (prevP - r) * (currP - r);
                if (magicNumber <= 0.f) {
                    k = j;
                    _tabu[i * alignedCities + s] = k;
                    v[k]= 0;
                    break;
                }
            }
        }
    }
    delete[] v;
    delete[] p;

    for (uint32_t i = 0; i < nAnts; ++i) {
        float len = 0.f;
        for (uint32_t j = 0; j < nCities; ++j) {
            const uint32_t from = _tabu[i * alignedCities + j];
            const uint32_t to = _tabu[i * alignedCities + j + 1];
            len += _distance[from * alignedCities + to];
        }
        const uint32_t from = _tabu[i * alignedCities + nCities - 1];
        const uint32_t to = _tabu[i * alignedCities];
        len += _distance[from * alignedCities + to];

        _tourLen[i] = len;
    }

    _bestPathLen = _tourLen[0];
    uint32_t _maxId = 0;
    for (uint32_t i = 1; i < nAnts; ++i) {
        if (_tourLen[i] > _bestPathLen) {
            _bestPathLen = _tourLen[i];
            _maxId = i;
        }
    }

    for (uint32_t i = 0; i < nCities; ++i) {
        _bestPath[i] = _tabu[_maxId * alignedCities + i];
    }

    std::cout << "_bestPathLen: " << _bestPathLen << std::endl;
    printMatrix("BEST_PATH", _bestPath, 1, nCities);

#endif

    dim3 dimBlock1D(32);
    dim3 dimBlock2D(16, 16);

    dim3 gridAnt1D(numberOfBlocks(nAnts, dimBlock1D.x));
    dim3 gridMatrix2D(numberOfBlocks(nCities, dimBlock2D.y), numberOfBlocks(nCities, dimBlock2D.x));

    startTimer();

    const dim3 initRandBlock(32);
    const dim3 initRandGrid( numberOfBlocks(randStateElems, initRandBlock.x) );

    const dim3 initEtaBlock(32);
    const dim3 initEtaGrid( numberOfBlocks(etaElems, initEtaBlock.x) );

    const dim3 initPheroBlock(32);
    const dim3 initPheroGrid( numberOfBlocks(pheromoneElems, initPheroBlock.x) );

    const dim3 initDeltaBlock(32);
    const dim3 initDeltaGrid( numberOfBlocks(deltaElems, initDeltaBlock.x) );

    initCurand   <<<initRandGrid,  initRandBlock >>>(randState, seed, alignedAnts);
    cudaCheck( cudaGetLastError() );
    initEta      <<<initEtaGrid,   initEtaBlock  >>>(eta, distance, nCities, alignedCities);
    cudaCheck( cudaGetLastError() );
    initPheromone<<<initPheroGrid, initPheroBlock>>>(pheromone, valPheromone, nCities, alignedCities);
    cudaCheck( cudaGetLastError() );
    initDelta    <<<initDeltaGrid, initDeltaBlock>>>(delta, nCities, alignedCities);
    cudaCheck( cudaGetLastError() );

    const dim3 fitBlock(32);
    const dim3 fitGrid( numberOfBlocks(fitnessElems, fitBlock.x) );

    const dim3 tourGrid(nAnts); // number of blocks
    const dim3 tourBlock(32); // number of threads in a block
    const uint32_t tourShared  = alignedCities  * sizeof(float)    + // p
                                 1              * sizeof(uint32_t) + // k
                                 alignedCities  * sizeof(uint8_t);   // v
    std::cout << " **** TOUR   sharedMemory **** \tKB " << (tourShared / 1024.f) << std::endl;

    const dim3 lenGrid(nAnts);
    const dim3 lenBlock(64);
    const uint32_t lenShared = lenBlock.x / 32 * sizeof(float);
    std::cout << " **** LEN    sharedMemory **** \tKB " << (lenShared / 1024.f)  << std::endl;

    const dim3 bestGrid(1);
    const dim3 bestBlock(32);

    const dim3 deltaGrid(nAnts);
    const dim3 deltaBlock(32);
    const uint32_t deltaShared = nCities * sizeof(uint32_t);
    std::cout << " **** DELTA  sharedMemory **** \tKB " << (deltaShared / 1024.f) << std::endl;

    const dim3 pheroBlock(32);
    const dim3 pheroGrid( numberOfBlocks(pheromoneElems, pheroBlock.x) );

    uint32_t epoch = 0;
    do {
        // initDelta        <<<initDeltaGrid, initDeltaBlock         >>>(delta, nCities, alignedCities);
        calculateFitness <<<fitGrid,       fitBlock               >>>(fitness, pheromone, eta, alpha, beta, nCities, alignedCities);
        cudaCheck( cudaGetLastError() );
        claculateTour    <<<tourGrid,      tourBlock,  tourShared >>>(tabu, fitness, nAnts, nCities, alignedCities, randState);
        cudaCheck( cudaGetLastError() );
        calculateTourLen <<<lenGrid,       lenBlock,   lenShared  >>>(distance, tabu, tourLen, nAnts, alignedCities, nCities);
        cudaCheck( cudaGetLastError() );
        updateBest       <<<bestGrid,      bestBlock              >>>(bestPath, tabu, tourLen, nAnts, alignedCities, nCities, bestPathLen);
        cudaCheck( cudaGetLastError() );
        updateDelta      <<<deltaGrid,     deltaBlock, deltaShared>>>(delta, tabu, tourLen, nAnts, alignedCities, nCities, q);
        cudaCheck( cudaGetLastError() );
        updatePheromone  <<<pheroGrid,     pheroBlock             >>>(pheromone, delta, nCities, alignedCities, rho);
        cudaCheck( cudaGetLastError() );
    } while (++epoch < maxEpoch);

    cudaDeviceSynchronize();
    
    std::this_thread::sleep_for (std::chrono::seconds(1));

#if DEBUG

    compareArray("distance",  _distance , distance,  distanceElems);
    compareArray("eta",       _eta      , eta,       etaElems);
    compareArray("pheromone", _pheromone, pheromone, pheromoneElems);
    compareArray("fitness",   _fitness  , fitness,   fitnessElems);
    compareArray("delta",     _delta    , delta,     deltaElems);
    compareArray("tabu",      _tabu     , tabu,      tabuElems);
    compareArray("tourLen",   _tourLen  , tourLen,   tourLenElems);
    compareArray("bestPath",  _bestPath , bestPath,  bestPathElems);

#endif
    // printMatrix("eta", eta, nCities, alignedCities);
    // printMatrix("pheromone_", pheromone, nCities, alignedCities);
    // printMatrix("delta", delta, nCities, alignedCities);
    // printMatrix("fitness", fitness, nCities, alignedCities);
    // printMatrix("tabu", tabu, nAnts, nCities);

    stopAndPrintTimer();
    cout << (tsp->checkPath(bestPath) == 1 ? "Path OK!" : "Error in the path!") << endl;
    cout << "bestPathLen: " << *bestPathLen << endl;
    cout << "CPU Path distance: " << tsp->calculatePathLen(bestPath) << endl;
    printMatrix("bestPath", bestPath, 1, nCities);


    cudaFree(randState);
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
