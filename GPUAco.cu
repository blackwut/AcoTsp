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


#define TOUR_GLOBAL 1
#define TOUR_PRIVATE 2
#define TOUR_PRIVATE_2 3
#define TOUR_REGISTER 4

#define TOUR_SELECTED TOUR_REGISTER

__global__ 
void initCurand(curandStateXORWOW_t * state,
                const unsigned long seed,
                const int nAnts)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
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
                const int rows,
                const int cols)
{    
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if ( row >= rows || col >= cols ) return;

    const int idx = row * cols + col;
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
                      const int rows,
                      const int cols)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if ( row >= rows || col >= cols ) return;

    const int idx = row * cols + col;
    fitness[idx] = __powf(pheromone[idx], alpha) * __powf(eta[idx], beta);
}

/*
__global__
void claculateTour(int * tabu, float * fitness, int rows, int cols, curandStateXORWOW_t * state)
{
	int antIdx = blockIdx.x;
	int tid = threadIdx.x;
	int idx = ((blockIdx.x * blockDim.x * (width >> 5)) + threadIdx.x);
	
	if ( antIdx >= rows ) return; //It is not required because the kernel will be launched with only "rows" blocks
	
	float p[1024];
	int visited[1024];
	float r;
	float sum;
	int i;
	int j;
	int k;
	
	for (i = 0; i < cols >> 5; ++i) {
		visited[i] = 1;
	}
	
	if (tid == 0) {
		k = cols * randXOR(state + idx);
		visited[k] = 0;
		tabu[idx * cols] = k;
	}
	
	for (int s = 1; s < cols; ++s) {
		
		//shuffle_sync of r
		//shuffle_sync of the last value of sum
		
		//reduce of 32 elements
		//#pragma unroll
		//for (int i = 1; i <= 32; i *= 2) {
		//	int n = __shfl_up_sync(0xffffffff, value, i, 32);
		//	if (lane_id >= i) value += n;
		//}
		
		//ballot?? to select the right thread that will write the next city
		
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
}*/

__global__
void claculateTourPrivate(int * tabu, float * fitness, int rows, int cols, curandStateXORWOW_t * state)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= rows ) return;

    float p[1024];
    int visited[1024];
    float r;
    float sum;
    int j;
    int k;

    for (int i = 0; i < cols; ++i) {
        visited[i] = 1;
    }

    k = cols * randXOR(state + idx);
    visited[k] = 0;
    tabu[idx * cols] = k;

    for (int s = 1; s < cols; ++s) {

        sum = 0.0f;
        const int i = k;
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
void claculateTourPrivate_2(int * tabu, float * fitness, int rows, int cols, curandStateXORWOW_t * state)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= rows ) return;

    int visited[128];
    int p[128];

    for (int i = 0; i < cols; ++i) {
        visited[i] = 1;
    }

    int k = cols * randXOR(state + idx);
    visited[k] = 0;
    tabu[idx * cols] = k;

    for (int s = 1; s < cols; ++s) {

        float sum = 0.0f;
        const int i = k;
        for (int j = 0; j < cols; ++j) {
            sum += fitness[i * cols + j] * visited[j];
            p[j] = sum;
        }

        const float r = randXOR(state + idx) * sum;
        k = -1;
        for (int j = 0; j < cols; ++j) {
            k += (k == -1) * (p[j] > r) * (j + 1);
        }
        visited[k] = 0;
        tabu[idx * cols + s] = k;
    }
}

__global__
void claculateTourGlobal(int * tabu, float * fitness, float * p, int * visited, int diff, int rows, int cols, curandStateXORWOW_t * state)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= rows ) return;

    for (int i = 0; i < cols; ++i) {
        visited[idx * diff + i] = 1;
    }

    int k = cols * randXOR(state + idx);
    visited[idx * diff + k] = 0;
    tabu[idx * cols] = k;

    for (int s = 1; s < cols; ++s) {

        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            sum += fitness[k * cols + j] * visited[idx * diff + j];
            p[idx * diff + j] = sum;
        }

        k = -1;
        const float r = randXOR(state + idx) * sum;
        for (int j = 0; j < cols; ++j) {
            // if ( k == -1 && p[j] >= r ) {
            //     k = j;
            // }
            k += (k == -1) * (p[idx * diff + j] >= r) * (j + 1);
        }
        visited[idx * diff + k] = 0;
        tabu[idx * cols + s] = k;
    }
}

__device__ __forceinline__ float atomicMaxFloat (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}


#define FULLMASK 0xFFFFFFFF

__inline__ __device__
float scanWarpFloat(float x) {
    const int laneid = threadIdx.x % warpSize;
    #pragma unroll
    for( int offset = 1 ; offset < 32 ; offset <<= 1 ) {
        float y = __shfl_up_sync(FULLMASK, x, offset);
        if(laneid >= offset) x += y;
    }
    return x;
}

__global__
void claculateTourRegister(int * tabu, float * fitness, float * dumpFloat, int * dumpInt, int rows, int cols, int warps, curandStateXORWOW_t * state)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // if (tid >= cols) return;
    
    const uint32_t numberOfBlocks = (cols + warpSize - 1) / warpSize;

    extern __shared__ int smem[];
    int   * visited = smem;
    float * p       = (float *) &visited[cols];
    int   * k       = (int *)   &p[cols];
    float * reduce  = (float *) &k[1];

    for (int i = tid; i < cols; i += blockDim.x) { 
        visited[i] = 1;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        *k = cols * 0.5;//randXOR(state + blockIdx.x);
        visited[*k] = 0;
        tabu[0] = *k; //TODO: generare indice corretto per la formica calcolata da blockIdx.x
    }
    __syncthreads();

    uint32_t kappa = *k;
    for (int s = 1; s < cols; ++s) {
        for (int i = tid; i < cols; i += blockDim.x) { 
            p[i] = fitness[kappa * cols + i] * visited[i];
            // dumpFloat[i] = p[i];
        }

        __syncthreads();

        const uint32_t warpId = threadIdx.x / 32;
        for (int blockIndex = warpId; blockIndex < numberOfBlocks; blockIndex += warps) {
            const uint32_t warpTid = (threadIdx.x & 31) + (blockIndex * warpSize);

            const float x = (warpTid < cols ? p[warpTid] : 0.f);
            const float y = scanWarpFloat(x);               // Scan and return the warpTid corresponding value
            const float z = __shfl_sync(FULLMASK, y, 31);   // Broadcast the last value of Scan

            __syncwarp();
            if (warpTid < cols) {p[warpTid] = y / z; reduce[blockIndex] = z;}
            if (warpTid < cols) dumpFloat[warpTid] = p[warpTid];
        }

        __syncthreads();


        if ( threadIdx.x < 32 ) {

            const uint32_t localID = threadIdx.x & 31;

            const uint32_t randomBlock = s / 32;


            float randomFloat = 0.f;
            if (localID == 0) {
                randomFloat = randXOR(state + blockIdx.x);
            }

            randomFloat = __shfl_sync(FULLMASK, randomFloat, 0);

//             float rng = 0.f;
//             uint32_t selectedBlock = 0;
//             if (tid == 0) {
//                 float max = reduce[0];
//                 for (uint32_t i = 1; i < numberOfBlocks; ++i) {
//                     const float r = reduce[i];
//                     printf("r - max %0.24f - %0.24f \n", r, max);
//                     max = fmaxf(r, max);
//                     printf("i - max %d - %0.24f\n", i, max);
// #define fcmp(a, b) fabs(a - b) < FLT_EPSILON
//                     if (fcmp(r, max)) {
//                         selectedBlock = i;
//                     }
//                     printf("selectedBlock: %d\n", selectedBlock);
//                 }
//                 rng = randXOR(state + blockIdx.x);
//             }

//             selectedBlock = __shfl_sync(FULLMASK, selectedBlock, 0);
//             rng = __shfl_sync(FULLMASK, rng, 0);

            const uint32_t pIndex = randomBlock * 32 + localID;
            const uint32_t bitmask = __ballot_sync(FULLMASK, randomFloat < p[pIndex]);
            const uint32_t selected = __ffs(bitmask) - 1;
            kappa = randomBlock * 32 + selected;

            if (localID == selected) {
                // printf("%d - %d - %d) rng: %f - p[%d] = %f\n", s, localID, selectedBlock, rng, pIndex, p[pIndex]);
                tabu[s] = pIndex; //TODO: generare indice corretto per la formica calcolata da blockIdx.x
                visited[pIndex] = 0;
            }
        }
    }
        // float sum = 0.0f;
        // const int i = k;
        // for (int j = 0; j < cols; ++j) {
        //     sum += fitness[i * cols + j] * visited[j];
        //     p[j] = sum;
        // }

        // for (int s = blockDim.x / 2; s > 0; s >>= 1) { 
        //     if (tid < s) {
        //         sdata[tid] += sdata[tid + s];
        //     }
        //     __syncthreads();
        // }

        // const float r = randXOR(state + idx) * sum;
        // k = -1;
        // for (int j = 0; j < cols; ++j) {
        //     k += (k == -1) * (p[j] > r) * (j + 1);
        // }
        // visited[k] = 0;
        // tabu[idx * cols + s] = k;
    // }
}

// __global__
// void claculateTour(int * tabu, float * fitness, int rows, int cols, curandStateXORWOW_t * state)
// {
//     const int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if ( idx >= rows ) return;

//     float p[32];
//     int visited[32];

//     for (int i = 0; i < cols; ++i) {
//         visited[i] = 1;
//     }

//     const int k = cols * randXOR(state + idx);
//     visited[k] = 0;
//     tabu[idx * cols] = k;

//     for (int s = 1; s < cols; ++s) {

//         float sum = 0.0f;
//         for (int j = 0; j < cols; ++j) {
//             sum += fitness[k * cols + j] * visited[j];
//             p[j] = sum;
//         }

//         const float r = randXOR(state + idx) * sum;
//         int to = -1;
//         for (int j = 0; j < cols; ++j) {
//             // if ( to == -1 && p[j] >= r ) {
//             //     to = j;
//             // }
//             to += (to == -1) * (p[j] >= r) * (j + 1);
//         }

//         //k += 1;
//         visited[to] = 0;
//         tabu[idx * cols + s] = to;
//     }
// }

__global__
void calculateTourLen(const int * tabu,
                      const float * distance,
                      int * tourLen,
                      const int rows,
                      const int cols)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
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
void updateBest(int * bestPath,
                const int * tabu,
                const int * tourLen,
                const int rows,
                const int cols,
                int * bestPathLen,
                const int last)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= rows ) return;

    const int len = tourLen[idx];
    atomicMin(bestPathLen, len);

    __syncthreads();

    if (*bestPathLen == len) {
        for (int i = 0; i < cols; ++i) {
            bestPath[i] = tabu[idx * cols + i];
        }
    }
}

__global__
void updateDelta(float * delta,
                 const int * tabu,
                 const int * tourLen,
                 const int rows,
                 const int cols,
                 const float q)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= rows ) return;

    for (int i = 0; i < cols - 1; ++i) {
        const int from = tabu[idx * cols + i];
        const int to = tabu[idx * cols + i + 1];
        atomicAdd(delta + (from * cols + to), q / tourLen[idx]);
        // atomicAdd(delta + (to * cols + from), q / tourLen[idx]);
    }

    const int from = tabu[idx * cols + cols - 1];
    const int to = tabu[idx * cols];
    atomicAdd(delta + (from * cols + to), q / tourLen[idx]);
    // atomicAdd(delta + (to * cols + from), q / tourLen[idx]);
}

__global__
void updatePheromone(float * pheromone,
                     float * delta,
                     const int rows,
                     const int cols,
                     const float rho)
{
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if ( row >= rows || col >= cols ) return;

    const int idx = row * cols + col;

    const float p = pheromone[idx];
    pheromone[idx] = p * (1.0f - rho) + delta[idx];
    delta[idx] = 0.0f;
}


int numberOfBlocks(int numberOfElements, int blockSize) {
    return (numberOfElements + blockSize - 1) / blockSize;
}

int roundWithBlockSize(int numberOfElements, int blockSize)
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
	int maxEpoch = 30;
	
	if (argc < 7) {
		cout << "Usage: ./acogpu file.tsp alpha beta q rho maxEpoch" << endl;
		exit(-1);
	}
	
    int seed = time(0);//123;
    
	argc--;
	argv++;
	int args = 0;
	strArg(argc, argv, args++, path);
	fltArg(argc, argv, args++, &alpha);
	fltArg(argc, argv, args++, &beta);
	fltArg(argc, argv, args++, &q);
	fltArg(argc, argv, args++, &rho);
	intArg(argc, argv, args++, &maxEpoch);

	TSP<D_TYPE> * tsp = new TSP<D_TYPE>(path);
	ACO<D_TYPE> * aco = new ACO<D_TYPE>(tsp->dimension, tsp->dimension, alpha, beta, q, rho, maxEpoch, 0);

    int nAnts = aco->nAnts;
    int nCities = tsp->dimension;
    float valPheromone = 1.0f / nCities;
    cout << "initialPheromone: " << valPheromone << endl;

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

    int * dumpInt;
    float * dumpFloat;

    int dumpSize = 512;

    float * p;
    int * visited;

    int elems = nCities * nCities;
    

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

    cudaMallocManaged(&dumpInt, dumpSize * sizeof(int));
    cudaMallocManaged(&dumpFloat, dumpSize * sizeof(float));

    int diff = nAnts * nCities;
    std::cout << "diff: " << diff << std::endl;
    cudaMallocManaged(&p      , diff * sizeof(float));
    cudaMallocManaged(&visited, diff * sizeof(int));

    for (int i = 0; i < dumpSize; ++i) {
        dumpInt[i] = -1;
        dumpFloat[i] = -1;
    }

    *bestPathLen = INT_MAX;

    for (int i = 0; i < nCities; ++i) {
        for (int j = 0; j < nCities; ++j) {
            distance[i * nCities + j] = tsp->edges[i * nCities +j];
        }
    }

    for (int i = 0; i < nCities; ++i) {
        tabu[i] = -127;
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

    int epoch = 0;
    do {
        calculateFitness<<<gridMatrix2D, dimBlock2D>>>(fitness, pheromone, eta, alpha, beta, nCities, nCities);

        dim3 gridTour(1);
        dim3 dimBlockTour(1);

        switch (TOUR_SELECTED) {
            case TOUR_GLOBAL:
                gridTour.x = 64;
                dimBlockTour.x = (nAnts + gridTour.x - 1) / gridTour.x;
                claculateTourGlobal<<<gridTour, dimBlockTour>>>(tabu, fitness, p, visited, diff, nAnts, nCities, state);
            break;

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

            case TOUR_REGISTER:
                gridTour.x = 1;            // number of blocks
                dimBlockTour.x = 64;         // number of threads in a block
                const uint32_t numberOfWarps = dimBlockTour.x / 32;

                claculateTourRegister<<<gridTour, dimBlockTour, (nAnts * 2 + 1 + numberOfWarps) * sizeof(int)>>>(tabu, fitness, dumpFloat, dumpInt, nAnts, nCities, numberOfWarps, state);
            break;
        }

        cudaDeviceSynchronize();


        // // for (uint32_t i = 0; i < nAnts; ++i) {
        // //     cout << "[" << i << "] = " << dumpInt[i] << endl;
        // // }
        for (uint32_t i = 0; i < nAnts; ++i) {
            cout << "[" << i << "] = " << dumpFloat[i] << endl;
        }

        
        // calculateTourLen<<<gridAnt1D, dimBlock1D>>>(tabu, distance, tourLen, nAnts, nCities);
        
        // updateBest<<<gridAnt1D, dimBlock1D>>>(bestPath, tabu, tourLen, nAnts, nCities, bestPathLen, ((epoch + 1) == maxEpoch));
        // updateDelta<<<gridAnt1D, dimBlock1D>>>(delta, tabu, tourLen, nAnts, nCities, q);
        // updatePheromone<<<gridMatrix2D, dimBlock2D>>>(pheromone, delta, nCities, nCities, rho);

    } while (++epoch < maxEpoch);

    cudaDeviceSynchronize();

    stopAndPrintTimer();

    printMatrix("tabu", tabu, 1, nCities);

    // cout << (tsp->checkPath(bestPath) == 1 ? "Path OK!" : "Error in the path!") << endl;
    // cout << "bestPathLen: " << *bestPathLen << endl;
    // cout << "CPU Path distance: " << tsp->calculatePathLen(bestPath) << endl;
    // printMatrix("bestPath", bestPath, 1, nCities);


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
