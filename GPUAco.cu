#include <iostream>
#include <fstream>
#include <float.h>
#include <cmath>
#include <climits>
#include <thread>
#include <chrono>

#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
using namespace cooperative_groups;

#include "common.hpp"
#include "TSP.cpp"

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
    return (float) curand_uniform(state);
}

__global__
void initEta(float * eta,
             const float * edges,
             const uint32_t rows,
             const uint32_t cols)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = tid; c < cols; c += gridDim.x * blockDim.x) {
            const uint32_t id = r * cols + c;
            const float d = edges[id];
            eta[id] = (d == 0.0) ? 0.0 : 1.0 / d;
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
            delta[id] = 0.0;
        }
    }
}

__global__
void calcFitness(float * fitness,
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
void calcTour(uint32_t * tabu,
              const float * fitness,
              const uint32_t rows,
              const uint32_t cols,
              const uint32_t alignedCols,
              curandStateXORWOW_t * state)
{
    extern __shared__ uint32_t smem[];
    uint8_t * v = (uint8_t *) smem;
    float   * p = (float *)   &v[alignedCols];

    // Vector type aliases
    const float4 * _f4 = (float4 *) fitness;
    uchar4 * _v4 = (uchar4 *) v;
    float4 * _p4 = (float4 *) p;
    const uint32_t cols4 = alignedCols / 4;


    const uint32_t tid = threadIdx.x;

    for (uint32_t ant = blockIdx.x; ant < rows; ant += gridDim.x) {

        for (uint32_t i = tid; i < cols4; i += 32) {
            _v4[i] = make_uchar4(1, 1, 1, 1);
        }
        __syncwarp();

        uint32_t kappa = 12345678;
        if (tid == 0) {
            kappa = cols * randXOR(state + ant);
            v[kappa] = 0;
            tabu[ant * alignedCols] = kappa;
        }
        kappa = __shfl_sync(FULL_MASK, kappa, 0);


        for (uint32_t s = 1; s < cols; ++s) {
            float sum = 0.0f;
            for (uint32_t pid = tid; pid < cols4; pid += 32) {
                const float4 f4 = _f4[kappa * cols4 + pid];
                const uchar4 v4 = _v4[pid];
                const float4 p4 = make_float4(f4.x * v4.x,
                                              f4.y * v4.y,
                                              f4.z * v4.z,
                                              f4.w * v4.w);

                const float xP4 = p4.x + p4.y + p4.z + p4.w;
                const float y   = sum + scanWarpFloat(tid, xP4);
                _p4[pid] = make_float4( y - p4.y - p4.z - p4.w,
                                        y        - p4.z - p4.w,
                                        y               - p4.w,
                                        y );
                sum = __shfl_sync(FULL_MASK, y, 31);
            }

            float randomFloat = -1.0;
            if (tid == 0) {
                randomFloat = randXOR(state + ant);
            }
            randomFloat = __shfl_sync(FULL_MASK, randomFloat, 0);

            const float probability = randomFloat * sum;
            uint32_t l = 0;
            uint32_t r = (cols + 31) / 32 - 1;

            while ( l < r ){
                const uint32_t m = (l + r) / 2;
                const uint32_t pid = (m * 32) + tid;
                const uint32_t ballotMask = __ballot_sync(FULL_MASK,  probability <= p[pid]);
                const uint32_t ntid = __popc(ballotMask);

                if (ntid == 0) {
                    l = m + 1;
                } else {
                    r = m;
                } 
            }

            const uint32_t pid = (l * 32) + tid;
            const uint32_t ballotMask = __ballot_sync(FULL_MASK,  probability <= p[pid]);
            const uint32_t winner = __ffs(ballotMask) - 1;
            if (tid == winner) {
                kappa = pid;
                tabu[ant * alignedCols + s] = pid;
                v[pid]= 0;
            }
            kappa = __shfl_sync(FULL_MASK, kappa, winner);
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
void calcTourLength(float * tourLength,
                    const float * edges,
                    const uint32_t * tabu,
                    const uint32_t rows,
                    const uint32_t cols,
                    const uint32_t realCols)
{
    __shared__ float finalLength[1];

    thread_block_tile<32> tile32 = tiled_partition<32>(this_thread_block());
    const uint32_t numberOfBlocks = (cols + 31) / 32;

    float totalLength = 0.0;
    for (uint32_t blockId = threadIdx.x / 32; blockId < numberOfBlocks; blockId += blockDim.x / 32) {
        const uint32_t warpTid = blockIdx.x * cols + tile32.thread_rank() + (blockId * 32);

        float length = 0.0;
        if (tile32.thread_rank() + (blockId * 32) < realCols - 1) {
            const uint32_t from = tabu[warpTid];
            const uint32_t to   = tabu[warpTid + 1];
            length = edges[from * cols + to];
        }
        totalLength += reduceTileFloat(tile32, length);
    }

    if (threadIdx.x == 0) {
        const uint32_t from = tabu[blockIdx.x * cols + realCols - 1];
        const uint32_t to   = tabu[blockIdx.x * cols];
        totalLength += edges[from * cols + to];
        finalLength[0] = 0.0;
    }
    __syncthreads();

    if (tile32.thread_rank() == 0) {
        atomicAdd(finalLength, totalLength);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        tourLength[blockIdx.x] = finalLength[0];
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
void updateBestTour(uint32_t * bestTour,
                    float * bestTourLength,
                    const uint32_t * tabu,
                    const float * tourLength,
                    const uint32_t rows,
                    const uint32_t cols)
{
    thread_block_tile<32> tile32 = tiled_partition<32>(this_thread_block());
    const uint32_t tid = threadIdx.x;

    uint32_t bestAnt = 1234567890;
    float minLength = FLT_MAX;

    for (uint32_t stride = 0; stride < cols; stride += 32) {
        const uint32_t warpTid = tid + stride;
        const float x = tourLength[warpTid];
        minLength = fminf(x, minLength);
        bestAnt = (x == minLength) ? warpTid : bestAnt;

        const float y = minTileFloat(tile32, minLength);
        const uint32_t mask = tile32.ballot( x == y );
        const uint32_t maxTile = __ffs(mask) - 1;
        minLength = tile32.shfl(y, maxTile);
        bestAnt = tile32.shfl(bestAnt, maxTile);
    }

    for (uint32_t i = tid; i < cols; i += 32) {
        bestTour[i] = tabu[bestAnt * cols + i];
    }

    if (tid == 0) {
        bestTourLength[0] = minLength;
    }
}

__global__
void updateDelta(float * delta,
                 const uint32_t * tabu,
                 const float * tourLenght,
                 const uint32_t rows,
                 const uint32_t cols,
                 const uint32_t realCols,
                 const float q)
{
    extern __shared__ uint32_t tabus[];
    const uint32_t tid = threadIdx.x;

    for (uint32_t i = tid; i < cols; i += blockDim.x) {
        tabus[i] = tabu[blockIdx.x * cols + i];
    }
    __syncthreads();

    const float tau = q / tourLenght[blockIdx.x];

    for (uint32_t i = tid; i < realCols - 1; i += blockDim.x) { 
        const uint32_t from = tabus[i];
        const uint32_t to   = tabus[i + 1];
        atomicAdd(delta + (from * cols + to), tau);
        atomicAdd(delta + (to * cols + from), tau);
    }

    if (tid == 0) {
        const uint32_t from = tabus[realCols - 1];
        const uint32_t to   = tabus[0];
        atomicAdd(delta + (from * cols + to), tau);
        atomicAdd(delta + (to * cols + from), tau);
    }
}

__global__
void updatePheromone(float * pheromone,
                     const float * delta,
                     const uint32_t rows,
                     const uint32_t cols,
                     const float rho)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = tid; c < cols; c += gridDim.x * blockDim.x) {
            const uint32_t id = r * cols + c;
            const float p = pheromone[id];
            pheromone[id] = p * rho + delta[id];
        }
    }
}

inline uint32_t divUp(const uint32_t elems, const uint32_t div) {
    return (elems + div - 1) / div;
}

inline uint32_t numberOfBlocks(const uint32_t elems, const uint32_t blockSize) {
    return divUp(elems, blockSize);
}

inline uint32_t alignToWarp4(const uint32_t elems) {
    return numberOfBlocks(elems, 128) * 128;
}

int main(int argc, char * argv[]) {

	char * path = new char[MAX_LEN];
	float alpha = 1.0f;
	float beta = 2.0f;
	float q = 1.0f;
	float rho = 0.5f;
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

    TSP<float> tsp(path);

    const uint64_t seed         = time(0);
    const uint32_t nAnts        = tsp.getNCities();
    const uint32_t nCities      = tsp.getNCities();
    const float    valPheromone = 1.0f / nCities;

    curandStateXORWOW_t * randState;
    float * edges;
    float * eta;
    float * pheromone;
    float * fitness;
    float * delta;
    uint32_t * tabu;
    float * tourLength;
    uint32_t * bestTour;
    float * bestTourLength;

    const uint32_t alignedAnts = alignToWarp4(nAnts);
    const uint32_t alignedCities = alignToWarp4(nCities);

    const uint32_t randStateRows  = alignedAnts;
    const uint32_t randStateCols  = 1;
    const uint32_t edgesRows      = nCities;
    const uint32_t edgesCols      = alignedCities;
    const uint32_t etaRows        = nCities;
    const uint32_t etaCols        = alignedCities;
    const uint32_t pheromoneRows  = nCities;
    const uint32_t pheromoneCols  = alignedCities;
    const uint32_t fitnessRows    = nCities;
    const uint32_t fitnessCols    = alignedCities;
    const uint32_t deltaRows      = nCities;
    const uint32_t deltaCols      = alignedCities;
    const uint32_t tabuRows       = nAnts;
    const uint32_t tabuCols       = alignedCities;
    const uint32_t tourLengthRows = alignedAnts;
    const uint32_t tourLengthCols = 1;
    const uint32_t bestTourRows   = alignedCities;
    const uint32_t bestTourCols   = 1;

    const uint32_t randStateElems  = randStateRows  * randStateCols;
    const uint32_t edgesElems      = edgesRows      * edgesCols;
    const uint32_t etaElems        = etaRows        * etaCols;
    const uint32_t pheromoneElems  = pheromoneRows  * pheromoneCols;
    const uint32_t fitnessElems    = fitnessRows    * fitnessCols;
    const uint32_t deltaElems      = deltaRows      * deltaCols;
    const uint32_t tabuElems       = tabuRows       * tabuCols;
    const uint32_t tourLengthElems = tourLengthRows * tourLengthCols;
    const uint32_t bestTourElems   = bestTourRows   * bestTourCols;

    cudaCheck( cudaMallocManaged(&randState,      randStateElems  * sizeof(curandStateXORWOW_t)) );
    cudaCheck( cudaMallocManaged(&edges,          edgesElems      * sizeof(float))               );
    cudaCheck( cudaMallocManaged(&eta,            etaElems        * sizeof(float))               );
    cudaCheck( cudaMallocManaged(&pheromone,      pheromoneElems  * sizeof(float))               );
    cudaCheck( cudaMallocManaged(&fitness,        fitnessElems    * sizeof(float))               );
    cudaCheck( cudaMallocManaged(&delta,          deltaElems      * sizeof(float))               );
    cudaCheck( cudaMallocManaged(&tabu,           tabuElems       * sizeof(uint32_t))            );
    cudaCheck( cudaMallocManaged(&tourLength,     tourLengthElems * sizeof(float))               );
    cudaCheck( cudaMallocManaged(&bestTour,       bestTourElems   * sizeof(uint32_t))            );
    cudaCheck( cudaMallocManaged(&bestTourLength, sizeof(float))                                 );

    const uint32_t totalMemory = (randStateElems  * sizeof(float)    +
                                  edgesElems      * sizeof(float)    +
                                  etaElems        * sizeof(float)    +
                                  pheromoneElems  * sizeof(float)    +
                                  fitnessElems    * sizeof(float)    +
                                  deltaElems      * sizeof(float)    +
                                  tabuElems       * sizeof(uint32_t) +
                                  tourLengthElems * sizeof(float)    +
                                  bestTourElems   * sizeof(uint32_t) +
                                  1               * sizeof(float));

    std::cout << " **** ACO TSP totalMemory **** \tMB " << (totalMemory / 1024.f) / 1024.f<< std::endl;

    rho = 1.0 - rho;

    *bestTourLength = FLT_MAX;
    for (uint32_t i = 0; i < tourLengthElems; ++i) {
        tourLength[i] = FLT_MAX;
    }

    const std::vector<float> & tspEdges = tsp.getEdges();
    for (uint32_t i = 0; i < nCities; ++i) {
        for (uint32_t j = 0; j < alignedCities; ++j) {
            const uint32_t alignedId = i * alignedCities + j;
            const uint32_t id = i * nCities + j;
            edges[alignedId] = (j < nCities) ? tspEdges[id] : 0.0;
        }
    }

    // Curand 
    const dim3 initRandBlock(128);
    const dim3 initRandGrid( numberOfBlocks(randStateElems, initRandBlock.x) );
    initCurand <<< initRandGrid, initRandBlock >>>(randState, seed, alignedAnts);
    cudaCheck( cudaGetLastError() );
    // Eta
    const dim3 initEtaBlock(128);
    const dim3 initEtaGrid( numberOfBlocks(etaCols, initEtaBlock.x) );
    initEta <<<initEtaGrid, initEtaBlock >>>(eta, edges, etaRows, etaCols);
    cudaCheck( cudaGetLastError() );
    // Pheromone
    const dim3 initPheroBlock(128);
    const dim3 initPheroGrid( numberOfBlocks(pheromoneCols, initPheroBlock.x) );
    initPheromone <<<initPheroGrid, initPheroBlock>>> (pheromone, valPheromone, pheromoneRows, pheromoneCols);
    cudaCheck( cudaGetLastError() );
    

    startTimer();
    uint32_t epoch = 0;
    do {
        // Delta
        const dim3 initDeltaBlock(128);
        const dim3 initDeltaGrid( numberOfBlocks(deltaCols, initDeltaBlock.x) );
        initDelta <<<initDeltaGrid, initDeltaBlock>>> (delta, deltaRows, deltaCols);
        cudaCheck( cudaGetLastError() );

        // Fitness
        const dim3 fitBlock(128);
        const dim3 fitGrid( numberOfBlocks(fitnessCols, fitBlock.x) );
        calcFitness <<<fitGrid, fitBlock >>> (fitness, pheromone, eta, alpha, beta, fitnessRows, fitnessCols);
        cudaCheck( cudaGetLastError() );

        // Tour
        const dim3 tourGrid( divUp(nAnts, 8) ); // max 8 due to vector type of order 4 and warpsize (32)
        const dim3 tourBlock(32);
        const uint32_t tourShared  = alignedCities  * sizeof(uint8_t) +   // v
                                     alignedCities  * sizeof(float); // p
        calcTour <<<tourGrid, tourBlock, tourShared>>> (tabu, fitness, nAnts, tabuRows, tabuCols, randState);
        cudaCheck( cudaGetLastError() );

        // TourLength
        const dim3 lenGrid(nAnts);
        const dim3 lenBlock(32);
        const uint32_t lenShared = lenBlock.x / 32 * sizeof(float);
        calcTourLength <<<lenGrid, lenBlock, lenShared>>> (tourLength, edges, tabu, nAnts, alignedCities, nCities);
        cudaCheck( cudaGetLastError() );

        // Update best
        const dim3 bestGrid(1);
        const dim3 bestBlock(32);
        updateBestTour <<<bestGrid, bestBlock>>> (bestTour, bestTourLength, tabu, tourLength, nAnts, alignedCities);
        cudaCheck( cudaGetLastError() );
        
        // Update Delta
        const dim3 deltaGrid(nAnts);
        const dim3 deltaBlock(32);
        const uint32_t deltaShared = alignedCities * sizeof(uint32_t);
        updateDelta <<<deltaGrid, deltaBlock, deltaShared>>> (delta, tabu, tourLength, nAnts, alignedCities, nCities, q);
        cudaCheck( cudaGetLastError() );

        // Update Pheromone
        const dim3 pheroBlock(32);
        const dim3 pheroGrid( numberOfBlocks(pheromoneCols, pheroBlock.x) );
        updatePheromone <<<pheroGrid, pheroBlock>>> (pheromone, delta, pheromoneRows, pheromoneCols, rho);
        cudaCheck( cudaGetLastError() );
    } while (++epoch < maxEpoch);

    cudaDeviceSynchronize();

    stopAndPrintTimer();
    printMatrix("bestTour", bestTour, 1, nCities);
    printResult(tsp.getName(), 0, 0, maxEpoch, getTimerMS(), getTimerUS(), *bestTourLength, tsp.checkPath(bestTour));

    if ( *bestTourLength != tsp.calculatePathLength(bestTour) ) {
        std::cout << "Huston we have a problem!" << std::endl;
    }

    cudaFree(randState);
    cudaFree(edges);
    cudaFree(eta);
    cudaFree(pheromone);
    cudaFree(fitness);
    cudaFree(delta);
    cudaFree(tabu);
    cudaFree(tourLength);
    cudaFree(bestTour);
    cudaFree(bestTourLength);

    return 0;
}
