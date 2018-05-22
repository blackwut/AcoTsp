#include <iostream>
#include <fstream>
#include <cmath>
#include <climits>

#include <curand.h>
#include <curand_kernel.h>

#include "common.hpp"
#include "TSPReader.cpp"


__global__ 
void initCurand(curandStateXORWOW_t * state, unsigned long seed, int nAnts)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if ( idx >= nAnts ) return;

    curand_init(seed, idx, 0, &state[idx]);
}

__device__ 
float randXOR(curandState *state)
{
    return (float) curand_uniform(state);
}

__global__
void initialize(float * distance, float * eta, float * pheromone, float * delta, float initialPheromone, int rows, int cols)
{    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ( row >= rows || col >= cols ) return;

    int id = row * cols + col;
    float d = distance[id];
    if ( d == 0 ) {
        eta[id] = 0.0f;
    } else {
        eta[id] = 1.0f / d;
    }

    pheromone[id] = initialPheromone;
    delta[id] = 0.0f;
}

__global__
void calculateFitness(float * fitness, float * pheromone, float * eta, float alpha, float beta, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int id = row * cols + col;

    if ( row >= rows || col >= cols ) return;
    
    fitness[id] = __powf(pheromone[id], alpha) * __powf(eta[id], beta);
}

__global__
void claculateTour(int * tabu, float * fitness, int rows, int cols, curandStateXORWOW_t * state)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if ( idx >= rows ) return;

    float p[1024];
    int visited[1024];
    float r;
    float sum;
    int i;
    int j;
    int k;

    for (i = 0; i < cols; ++i) {
        visited[i] = 1;
    }

    k = cols * randXOR(state + idx);
    visited[k] = 0;
    tabu[idx * cols] = k;

    for (int s = 1; s < cols; ++s) {

        sum = 0.0f;
        i = k;
        for (j = 0; j < cols; ++j) {
            sum += fitness[i * cols + j] * visited[j];
            p[j] = sum;
        }

        r = randXOR(state + idx) * sum;
        k = -1;
        for (j = 0; j < cols; ++j) {
            if ( k == -1 && p[j] > r ) {
                k = j;
            }
            //k += (k == -1) * (p[j] > r) * j;
        }
        //k += 1;
        visited[k] = 0;
        tabu[idx * cols + s] = k;
    }
}

__global__
void calculateTourLen(int * tabu, float * distance, int * tourLen, int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if ( idx >= rows ) return;

    float length = 0.0f;
    int from;
    int to;
    for (int i = 0; i < cols - 1; ++i) {
        from = tabu[idx * cols + i];
        to = tabu[idx * cols + i + 1];
        length += distance[from * cols + to];
    }

    from = tabu[idx * cols + cols - 1];
    to = tabu[idx * cols];
    tourLen[idx] = length + distance[from * cols + to];
}

__global__
void updateBest(int * bestPath, int * tabu, int * tourLen, int rows, int cols, int * bestPathLen, int last)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if ( idx >= rows ) return;

    int len = tourLen[idx];
    atomicMin(bestPathLen, len);

    if (*bestPathLen == len) {
        for (int i = 0; i < cols; ++i) {
            bestPath[i] = tabu[idx * cols + i];
        }
    }
}

__global__
void updateDelta(float * delta, int * tabu, int * tourLen, int rows, int cols, float q)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if ( idx >= rows ) return;
    
    int from;
    int to;
    for (int i = 0; i < cols - 1; ++i) {
        from = tabu[idx * cols + i];
        to = tabu[idx * cols + i + 1];
        atomicAdd(delta + (from * cols + to), q / tourLen[idx]);
        atomicAdd(delta + (to * cols + from), q / tourLen[idx]);
    }

    from = tabu[idx * cols + cols - 1];
    to = tabu[idx * cols];
    atomicAdd(delta + (from * cols + to), q / tourLen[idx]);
    atomicAdd(delta + (to * cols + from), q / tourLen[idx]);
}

__global__
void updatePheromone(float * pheromone, float * delta, int rows, int cols, float rho)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ( row >= rows || col >= cols ) return;

    float p = pheromone[row * cols + col];
    p = p * (1.0f - rho) + delta[row * cols + col];
    pheromone[row * cols + col] = p;
    delta[row * cols + col] = 0.0f;
}


int numberOfBlocks(int numberOfElements, int blockSize) {
    return (numberOfElements + blockSize - 1) / blockSize;
}

int roundWithBlockSize(int numberOfElements, int blockSize)
{
    return numberOfBlocks(numberOfElements, blockSize) * blockSize; 
}

int main(int argc, char * argv[]) {

    char * path = (char *) malloc(MAX_LEN);
    float alpha = 4.0f;
    float beta = 2.0f;
    float q = 55.0f;
    float rho = 0.8f;
    int maxEpochs = 30;

    int seed = time(0);//123;
    
    argc--;
    argv++;
    int args = 0;
    stringArg(argc, argv, args++, path);
    floatArg(argc, argv, args++, &alpha);
    floatArg(argc, argv, args++, &beta);
    floatArg(argc, argv, args++, &q);
    floatArg(argc, argv, args++, &rho);
    intArg(argc, argv, args++, &maxEpochs);

    TSP * tsp = getTPSFromFile(path);

    int nAnts = tsp->numberOfCities;
    int nCities = tsp->numberOfCities;
    float valPheromone = 1.0f / nCities;

    curandStateXORWOW_t * state;
    float * distance;
    float * eta;
    float * pheromone;
    float * fitness;
    float * delta;
    int * tabu;
    int * tourLen;
    int * bestPath;
    int * bestPathLen;

    int elems = nCities * nCities;
    
    startTimer();

    cudaMallocManaged(&state, nAnts * sizeof(curandStateXORWOW_t));
    cudaMallocManaged(&distance, elems * sizeof(float));
    cudaMallocManaged(&eta, elems * sizeof(float));
    cudaMallocManaged(&pheromone, elems * sizeof(float));
    cudaMallocManaged(&fitness, elems * sizeof(float));
    cudaMallocManaged(&delta, elems * sizeof(float));
    cudaMallocManaged(&tabu, nAnts * nCities * sizeof(int));
    cudaMallocManaged(&tourLen, nAnts * sizeof(int));
    cudaMallocManaged(&bestPath, nCities * sizeof(int));
    cudaMallocManaged(&bestPathLen, sizeof(int));

    *bestPathLen = INT_MAX;

    for (int i = 0; i < nCities; ++i) {
        for (int j = 0; j < nCities; ++j) {
            distance[i * nCities + j] = tsp->distance[i * nCities +j];
        }
    }

    dim3 dimBlock1D(32);
    dim3 dimBlock2D(16, 16);

    dim3 gridAnt1D(numberOfBlocks(nAnts, dimBlock1D.x));
    // dim3 gridAnt2D(numberOfBlocks(nAnts, dimBlock2D.y), numberOfBlocks(nCities, dimBlock2D.x));
    dim3 gridMatrix2D(numberOfBlocks(nCities, dimBlock2D.y), numberOfBlocks(nCities, dimBlock2D.x));

    initCurand<<<gridAnt1D, dimBlock1D>>>(state, seed, nAnts);
    initialize<<<gridMatrix2D, dimBlock2D>>>(distance, eta, pheromone, delta, valPheromone, nCities, nCities);

    int epoch = 0;
    do {
        calculateFitness<<<gridMatrix2D, dimBlock2D>>>(fitness, pheromone, eta, alpha, beta, nCities, nCities);        
        claculateTour<<<gridAnt1D, dimBlock1D>>>(tabu, fitness, nAnts, nCities, state);
        calculateTourLen<<<gridAnt1D, dimBlock1D>>>(tabu, distance, tourLen, nAnts, nCities);
        
        updateBest<<<gridAnt1D, dimBlock1D>>>(bestPath, tabu, tourLen, nAnts, nCities, bestPathLen, ((epoch + 1)== maxEpochs));
        updateDelta<<<gridAnt1D, dimBlock1D>>>(delta, tabu, tourLen, nAnts, nCities, q);
        updatePheromone<<<gridMatrix2D, dimBlock2D>>>(pheromone, delta, nCities, nCities, rho);

    } while (++epoch < maxEpochs);

    cudaDeviceSynchronize();

    stopAndPrintTimer();

    cout << (checkPathPossible(tsp, bestPath) == 1 ? "Path OK!" : "Error in the path!") << endl;
    cout << "bestPathLen: " << *bestPathLen << endl;
    cout << "CPU Path distance: " << calculatePath(tsp, bestPath) << endl;
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
