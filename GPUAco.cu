#include <iostream>
#include <fstream>
#include <cmath>
#include <climits>

#include <curand.h>
#include <curand_kernel.h>

#include "common.hpp"
#include "TSPReader.cpp"


__global__ void initCurand(curandStateXORWOW_t * state, unsigned long seed, int nAnts)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if ( idx >= nAnts) return;

    curand_init(seed, idx, 0, &state[idx]);
}

__device__ 
float randXOR(curandState *state)
{
    return (float) curand_uniform(state);
}

__global__
void initEta(float * distance, float * eta, int rows, int cols)
{    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ( row >= rows || col >= cols ) return;

    //eta[row * cols + col] = __saturatef((1 - (row == col)) / distance[row * cols + col]);
    float d = distance[row * cols + col];
    if ( d == 0 ) {
        eta[row * cols + col] = 0.0f;
    } else {
        eta[row * cols + col] = 1.0f / d;
    }
}

__global__
void initPheromone(float * pheromone, float * delta, float val, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ( row >= rows || col >= cols ) return;

    pheromone[row * cols + col] = val;
    delta[row * cols + col] = 0.0f;
}

__global__
void clearAnts(int * visited, int * tabu, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ( row >= rows || col >= cols ) return;

    visited[row * rows + col] = 0;
    tabu[row * cols + col] = 0;
}

__global__
void placeAnts(int * visited, int * tabu, int nAnts, int nCities, curandStateXORWOW_t * state)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= nAnts) return;

    int j = nCities * randXOR(state + idx);
    visited[idx * nCities + j] = 1;
    tabu[idx * nCities] = j;
}

 __global__
void calculateFitness(float * fitness, float * pheromone, float * eta, float alpha, float beta, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int id = row * cols + col;

    if ( row >= rows || col >= cols ) return;
    
    fitness[id] = __powf(pheromone[id], alpha) + __powf(eta[id], beta);
}


__global__
void claculateTour(int * visited, int * tabu, float * fitness, const int rows, const int cols, curandStateXORWOW_t * state)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if ( idx >= rows ) return;

    //float * p = (float *) malloc( cols * sizeof(float) );
    float p[1024];
    int _visited[1024] = {0};
    float r;
    float sum;
    int i;
    int j;
    int k;

    _visited[tabu[idx * cols]] = 1;

    for (int s = 1; s < cols; ++s) {

        // r = randXOR(state + idx);
        sum = 0.0f;
        i = tabu[idx * cols + (s - 1)];
        for (j = 0; j < cols; ++j) {
            sum += fitness[i * cols + j] * (1 - _visited[j]);//visited[idx * cols + j]);
            p[j] = sum;
        }

        // for (j = 0; j < cols; ++j) {
        //     p[j] /= sum;
        // }

        r = randXOR(state + idx) * sum;
        k = -1;
        for (j = 0; j < cols; ++j) {
            if ( k == -1 && p[j] > r) {
                k = j;
            }
        }
        //visited[idx * cols + k] = 1;
        _visited[k] = 1;
        tabu[idx * cols + s] = k;
    }

    //free(p);
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
void updateBest(int * bestPath, int * tabu, int * tourLen, int rows, int cols, int * bestPathLen)
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

    int seed = 123;
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
    int * visited;
    int * tabu;
    int * tourLen;
    int * bestPath;
    int * bestPathLen;

    int blockSize = 32;
    int elems = nCities * nCities;

    startTimer();

    cudaMallocManaged(&state, nAnts * sizeof(curandStateXORWOW_t));
    cudaMallocManaged(&distance, elems * sizeof(float));
    cudaMallocManaged(&eta, elems * sizeof(float));
    cudaMallocManaged(&pheromone, elems * sizeof(float));
    cudaMallocManaged(&fitness, elems * sizeof(float));
    cudaMallocManaged(&delta, elems * sizeof(float));
    cudaMallocManaged(&visited, nAnts * nCities * sizeof(int));
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

    int grid = roundWithBlockSize(nAnts, blockSize);
    dim3 dimBlock(8, 8);
    dim3 dimGridAnt(roundWithBlockSize(nAnts, dimBlock.y), roundWithBlockSize(nCities, dimBlock.x));
    dim3 dimGridMat(roundWithBlockSize(nCities, dimBlock.y), roundWithBlockSize(nCities, dimBlock.x));

    initCurand<<<grid, blockSize>>>(state, seed, nAnts);
    initPheromone<<<dimGridMat, dimBlock>>>(pheromone, delta, valPheromone, nCities, nCities);
    initEta<<<dimGridMat, dimBlock>>>(distance, eta, nCities, nCities);

    int epoch = 0;
    do {
        clearAnts<<<dimGridAnt, dimBlock>>>(visited, tabu, nAnts, nCities);
        placeAnts<<<grid, blockSize>>>(visited, tabu, nAnts, nCities, state);
        calculateFitness<<<dimGridMat, dimBlock>>>(fitness, pheromone, eta, alpha, beta, nCities, nCities);
        claculateTour<<<grid, blockSize>>>(visited, tabu, fitness, nAnts, nCities, state);
        calculateTourLen<<<grid, blockSize>>>(tabu, distance, tourLen, nAnts, nCities);
        updateBest<<<grid, blockSize>>>(bestPath, tabu, tourLen, nAnts, nCities, bestPathLen);
        updateDelta<<<grid, blockSize>>>(delta, tabu, tourLen, nAnts, nCities, q);
        updatePheromone<<<dimGridMat, dimBlock>>>(pheromone, delta, nCities, nCities, rho);

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
    cudaFree(visited);
    cudaFree(tabu);
    cudaFree(tourLen);
    cudaFree(bestPath);
    cudaFree(bestPathLen);

    return 0;
}
