#include <iostream>
#include <fstream>
#include <float.h>
#include <cmath>
#include <climits>
#include <thread>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>
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
        std::cout <<  "cudaErrorAssert: "<< cudaGetErrorString(code) << " " << file << " " << line << std::endl;
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
            if ( d == 0.0 ) {
                eta[id] = 0.0;
            } else {
                eta[id] = __powf(d, -2.0);
            }
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
            fitness[id] = __powf(p, alpha) * e;//__powf(e, beta);
        }
    }
}

__device__ __forceinline__
float scanTileFloat(const thread_block_tile<32> & g, float x) {
    #pragma unroll
    for( uint32_t offset = 1 ; offset < 32 ; offset <<= 1 ) {
        const float y = g.shfl_up(x, offset);
        if(g.thread_rank() >= offset) x += y;
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

    thread_block_tile<32> tile32 = tiled_partition<32>(this_thread_block());
    const uint32_t tid    = threadIdx.x;
    const uint32_t tileId = tid / 32;
    const uint32_t tiles  = blockDim.x / 32;

    extern __shared__ uint32_t smem[];
    uint8_t * v = (uint8_t *) smem;
    float   * p = (float *)   &v[alignedCols * tiles];
    const uint32_t cols4 = alignedCols / 4;


    for (uint32_t ant = tileId + (blockIdx.x * tiles); ant < rows; ant += (gridDim.x * tiles) ) {

        for (uint32_t i = tile32.thread_rank(); i < cols4; i += warpSize) {
            const uint32_t idx = tileId * cols4 + i;
            reinterpret_cast<uchar4 *>(v)[idx] = make_uchar4(1, 1, 1, 1);
        }
        tile32.sync();

        uint32_t kappa = 12345678;
        if (tile32.thread_rank() == 0) {
            kappa = cols * randXOR(state + ant);
            v[tileId * alignedCols + kappa] = 0;
            tabu[ant * alignedCols] = kappa;
        }
        kappa = tile32.shfl(kappa, 0);

        for (uint32_t s = 1; s < cols; ++s) {

            tile32.sync();

            for (uint32_t pid = tile32.thread_rank(); pid < cols4; pid += warpSize) {
                const uint32_t idx = tileId * cols4 + pid;
                const float4 f4 = reinterpret_cast<const float4 *>(fitness)[kappa * cols4 + pid];
                const uchar4 v4 = reinterpret_cast<uchar4 *>(v)[idx];
                reinterpret_cast<float4 *>(p)[idx] = make_float4(f4.x * v4.x,
                                                                 f4.y * v4.y,
                                                                 f4.z * v4.z,
                                                                 f4.w * v4.w);
            }
            tile32.sync();

            float sum = 0.0f;
            for (uint32_t pid = tile32.thread_rank(); pid < cols4; pid += warpSize) {
                const uint32_t idx = tileId * cols4 + pid;
                const float4 p4 = reinterpret_cast<float4 *>(p)[idx];
                const float xP4 = p4.x + p4.y + p4.z + p4.w;
                const float y   = sum + scanTileFloat(tile32, xP4);
                reinterpret_cast<float4 *>(p)[idx] = make_float4( y - p4.y - p4.z - p4.w,
                                                                  y        - p4.z - p4.w,
                                                                  y               - p4.w,
                                                                  y );
                sum = tile32.shfl(y, 31);
            }
            tile32.sync();

            float randomFloat = -1.0;
            if (tile32.thread_rank() == 0) {
                randomFloat = randXOR(state + ant);
            }
            randomFloat = tile32.shfl(randomFloat, 0);

            const float probability = randomFloat * sum;
            uint32_t l = 0;
            uint32_t r = (cols + 31) / 32 - 1;

            while ( l < r ){
                const uint32_t m = (l + r) / 2;
                const uint32_t pid = (m * 32) + tile32.thread_rank();
                const float prob = p[tileId * alignedCols + pid];
                const uint32_t ballotMask = tile32.ballot(probability <= prob);
                const uint32_t ntid = __popc(ballotMask);

                if (ntid == 0) {
                    l = m + 1;
                } else {
                    r = m;
                } 
            }

            const uint32_t pid = (l * 32) + tile32.thread_rank();
            const float prob = p[tileId * alignedCols + pid];
            const uint32_t ballotMask = tile32.ballot(probability <= prob);
            const uint32_t winner = __ffs(ballotMask) - 1;
            if (tile32.thread_rank() == winner) {
                kappa = pid;
                tabu[ant * alignedCols + s] = pid;
                v[tileId * alignedCols + pid]= 0;
            }
            kappa = tile32.shfl(kappa, winner);
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

    uint32_t bestAnt = 1234567890;
    float minLength = FLT_MAX;

    for (uint32_t lid = tile32.thread_rank(); lid < cols; lid += 32) {
        const float x = tourLength[lid];
        if ( x < minLength ) {
            minLength = x;
            bestAnt = lid;
        }
    }

    const float y = minTileFloat(tile32, minLength);
    const uint32_t ballotMask = tile32.ballot( minLength == y );
    const uint32_t winner = __ffs(ballotMask) - 1;
    bestAnt = tile32.shfl(bestAnt, winner);

    for (uint32_t bid = tile32.thread_rank(); bid < cols; bid += 32) {
        bestTour[bid] = tabu[bestAnt * cols + bid];
    }

    if (tile32.thread_rank() == winner) {
        *bestTourLength = minLength;
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
    uint32_t maxEpoch = 1;
    uint32_t threadsPerBlock = 128;
    uint32_t nWarpsPerBlock = 1;
    uint32_t nMaxAntsPerWarp = 1;

    if ( argc < 7 ) {
        std::cout << "Usage:"
        << " ./acogpu"
        << " file.tsp"
        << " alpha"
        << " beta"
        << " q"
        << " rho"
        << " maxEpoch"
        << " [threadsPerBlock = " << threadsPerBlock << "]"
        << " [nWarpsPerBlock = "  << nWarpsPerBlock  << "]"
        << " [nMaxAntsPerWarp = " << nMaxAntsPerWarp << "]"
        << std::endl;
        exit(-1);
    }

    path            = argv[1];
    alpha           = parseArg<float>   (argv[2]);
    beta            = parseArg<float>   (argv[3]);
    q               = parseArg<float>   (argv[4]);
    rho             = parseArg<float>   (argv[5]);
    maxEpoch        = parseArg<uint32_t>(argv[6]);
    if ( argc > 7 ) threadsPerBlock = parseArg<uint32_t>(argv[7]);
    if ( argc > 8 ) nWarpsPerBlock  = parseArg<uint32_t>(argv[8]);
    if ( argc > 9 ) nMaxAntsPerWarp = parseArg<uint32_t>(argv[9]);

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

    const float gmemRequired = (randStateElems  * sizeof(float)    +
                                   edgesElems      * sizeof(float)    +
                                   etaElems        * sizeof(float)    +
                                   pheromoneElems  * sizeof(float)    +
                                   fitnessElems    * sizeof(float)    +
                                   deltaElems      * sizeof(float)    +
                                   tabuElems       * sizeof(uint32_t) +
                                   tourLengthElems * sizeof(float)    +
                                   bestTourElems   * sizeof(uint32_t) +
                                   1               * sizeof(float)
                                   ) / 1048576.0;

    const uint32_t smemRequired  = nWarpsPerBlock * alignedCities * 5;

    int deviceCount = 0;
    cudaCheck( cudaGetDeviceCount(&deviceCount) );
    if (deviceCount == 0) {
        std::cout << "There are no available device(s) that support CUDA" << std::endl;
        exit(-1);
    }
    cudaDeviceProp deviceProp;
    cudaCheck( cudaGetDeviceProperties(&deviceProp, 0) );

    const float globalMemory = deviceProp.totalGlobalMem / 1048576.0;
    const uint32_t sharedMemory = deviceProp.sharedMemPerBlock;

    std::cout << "       Device: " << deviceProp.name           << std::endl
              << "Global memory: " << std::setw(8) << std::setprecision(2) << std::fixed << globalMemory     << " MB" << std::endl
              << "     required: " << std::setw(8) << std::setprecision(2) << std::fixed << gmemRequired     << " MB" << std::endl
              << "Shared memory: " << std::setw(8) << std::setprecision(2) << std::fixed << sharedMemory     << "  B" << std::endl
              << "     required: " << std::setw(8) << std::setprecision(2) << std::fixed << smemRequired     << "  B" << std::endl;


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
    const dim3 initRandBlock( threadsPerBlock );
    const dim3 initRandGrid( numberOfBlocks(randStateElems, initRandBlock.x) );
    // Eta
    const dim3 initEtaBlock( threadsPerBlock );
    const dim3 initEtaGrid( numberOfBlocks(etaCols, initEtaBlock.x) );
    // Pheromone
    const dim3 initPheroBlock( threadsPerBlock );
    const dim3 initPheroGrid( numberOfBlocks(pheromoneCols, initPheroBlock.x) );
    // Delta
    const dim3 initDeltaBlock( threadsPerBlock );
    const dim3 initDeltaGrid( numberOfBlocks(deltaCols, initDeltaBlock.x) );
    // Fitness
    const dim3 fitBlock( threadsPerBlock );
    const dim3 fitGrid( numberOfBlocks(fitnessCols, fitBlock.x) );
    // Tour
    const dim3 tourGrid( divUp(nAnts, nWarpsPerBlock * nMaxAntsPerWarp) );
    const dim3 tourBlock(32 * nWarpsPerBlock);
    const uint32_t tourShared  = nWarpsPerBlock * (alignedCities  * sizeof(uint8_t) + alignedCities  * sizeof(float));
    // TourLength
    const dim3 lenGrid( nAnts );                // must be nAnts
    const dim3 lenBlock( threadsPerBlock );
    const uint32_t lenShared = lenBlock.x / 32 * sizeof(float);
    // Update best
    const dim3 bestGrid(1);                     // must be 1
    const dim3 bestBlock(32);                   // must be 32
     // Update Delta
    const dim3 deltaGrid( nAnts );              // must be nAnts
    const dim3 deltaBlock( threadsPerBlock );
    const uint32_t deltaShared = alignedCities * sizeof(uint32_t);
     // Update Pheromone
    const dim3 pheroBlock( threadsPerBlock );
    const dim3 pheroGrid( numberOfBlocks(pheromoneCols, pheroBlock.x) );

    //
    uint32_t threadsActive = 0;
    uint32_t realActiveBlocks = 0;
    uint32_t maxActiveBlocks = 0;
    cudaCheck( cudaOccupancyMaxActiveBlocksPerMultiprocessor((int *)&maxActiveBlocks, calcTour, tourBlock.x, tourShared) );

    realActiveBlocks = (tourGrid.x < maxActiveBlocks * deviceProp.multiProcessorCount) ?
                        tourGrid.x : maxActiveBlocks * deviceProp.multiProcessorCount;

    threadsActive = realActiveBlocks * tourBlock.x;

    if ( tourShared > deviceProp.sharedMemPerBlock ) {
        std::cout << "Shared memory is not enough. Please reduce nWarpsPerBlock." << std::endl;
        printResult(tsp.getName(),
                    0,
                    threadsActive,
                    maxEpoch,
                    0,
                    0,
                    nWarpsPerBlock,
                    nMaxAntsPerWarp,
                    false);
        exit(-1);
    }

    initCurand <<< initRandGrid, initRandBlock >>>(randState, seed, alignedAnts);
    cudaCheck( cudaGetLastError() );
    initEta <<<initEtaGrid, initEtaBlock >>>(eta, edges, etaRows, etaCols);
    cudaCheck( cudaGetLastError() );
    initPheromone <<<initPheroGrid, initPheroBlock>>> (pheromone, valPheromone, pheromoneRows, pheromoneCols);
    cudaCheck( cudaGetLastError() );

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaCheck( cudaEventCreate(&start) );
    cudaCheck( cudaEventCreate(&stop) );
    cudaCheck( cudaEventRecord(start, 0) );

    uint32_t epoch = 0;
    do {
        initDelta <<<initDeltaGrid, initDeltaBlock>>> (delta, deltaRows, deltaCols);
        cudaCheck( cudaGetLastError() );
        calcFitness <<<fitGrid, fitBlock >>> (fitness, pheromone, eta, alpha, beta, fitnessRows, fitnessCols);
        cudaCheck( cudaGetLastError() );
        calcTour <<<tourGrid, tourBlock, tourShared>>> (tabu, fitness, nAnts, tabuRows, tabuCols, randState);
        cudaCheck( cudaGetLastError() );
        calcTourLength <<<lenGrid, lenBlock, lenShared>>> (tourLength, edges, tabu, nAnts, alignedCities, nCities);
        cudaCheck( cudaGetLastError() );
        updateBestTour <<<bestGrid, bestBlock>>> (bestTour, bestTourLength, tabu, tourLength, nAnts, alignedCities);
        cudaCheck( cudaGetLastError() );
        updateDelta <<<deltaGrid, deltaBlock, deltaShared>>> (delta, tabu, tourLength, nAnts, alignedCities, nCities, q);
        cudaCheck( cudaGetLastError() );
        updatePheromone <<<pheroGrid, pheroBlock>>> (pheromone, delta, pheromoneRows, pheromoneCols, rho);
        cudaCheck( cudaGetLastError() );
    } while (++epoch < maxEpoch);

    cudaCheck( cudaEventRecord(stop, 0) );
    cudaCheck( cudaEventSynchronize(stop) );
    float msec;
    long usec;
    cudaCheck( cudaEventElapsedTime(&msec, start, stop) );
    usec = msec * 1000;
    std::cout << "Compute time: " << msec << " ms " << usec << " usec " << std::endl;
    printMatrix("bestTour", bestTour, 1, nCities);
    printResult(tsp.getName(),
                realActiveBlocks,
                threadsActive,
                maxEpoch,
                msec,
                usec,
                nWarpsPerBlock,
                nMaxAntsPerWarp,
                tsp.checkTour(bestTour));

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
