#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

const int numThreads = 16; // You can adjust the number of threads per block here

__device__ int min(int a, int b, int c) {
    int minVal = a;
    if (b < minVal) minVal = b;
    if (c < minVal) minVal = c;
    return minVal;
}

__global__ void dtwKernel(const int* seqA, const int* seqB, int seqLengthA, int seqLengthB, int* results) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; // Global thread ID

    // Compute DTW for the current pair of time series (seqA[tid] vs. seqB[tid])
    for (int i = 0; i < seqLengthA; i++) {
        for (int j = 0; j < seqLengthB; j++) {
            int cost = abs(seqA[tid * seqLengthA + i] - seqB[tid * seqLengthB + j]);
            int prevCost = (i == 0 || j == 0) ? 0 : results[tid * seqLengthA * seqLengthB + (i - 1) * seqLengthB + (j - 1)];
            int minCost = min(results[tid * seqLengthA * seqLengthB + i * seqLengthB + j], prevCost, results[tid * seqLengthA * seqLengthB + i * seqLengthB + j - 1]);
            results[tid * seqLengthA * seqLengthB + i * seqLengthB + j] = cost + minCost;
        }
    }
}

int main() {
    int numTimeSeries = 11; // You can adjust the number of time series data pairs to compute DTW
    int seqLengthA = 20;    // Length of the first time series data
    int seqLengthB = 10;    // Length of the second time series data

    // Initialize multiple sets of time series data (you can replace this with your actual input data)
    //double* hostSeqA = (double*)malloc(seqLengthA * sizeof(double));
    //double* hostSeqB = (double*)malloc(seqLengthB * sizeof(double));
    double hostSeqA[] = {1.0, 7.0, 3.0, 4.0, 1.0, 10.0, 5.0, 4.0, 7.0, 4.0, 3.0, 7.0, 5.0, 4.0, 5.0, 7.0, 5.0, 2.0, 3.0, 9.0};
        
    double hostSeqB[] = {1.0, 4.0, 5.0, 10.0, 9.0, 3.0, 2.0, 6.0, 8.0, 4.0};
    

    int* deviceSeqA;
    int* deviceSeqB;
    int* deviceResults;

    int resultsSize = numTimeSeries * seqLengthA * seqLengthB;
    int* hostResults = (int*)malloc(resultsSize * sizeof(int));

    cudaMalloc((void**)&deviceSeqA, numTimeSeries * seqLengthA * sizeof(int));
    cudaMalloc((void**)&deviceSeqB, numTimeSeries * seqLengthB * sizeof(int));
    cudaMalloc((void**)&deviceResults, resultsSize * sizeof(int));

    cudaMemcpy(deviceSeqA, hostSeqA, numTimeSeries * seqLengthA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceSeqB, hostSeqB, numTimeSeries * seqLengthB * sizeof(int), cudaMemcpyHostToDevice);

    int numBlocks = (numTimeSeries + numThreads - 1) / numThreads;
    dtwKernel<<<numBlocks, numThreads>>>(deviceSeqA, deviceSeqB, seqLengthA, seqLengthB, deviceResults);

    cudaMemcpy(hostResults, deviceResults, resultsSize * sizeof(int), cudaMemcpyDeviceToHost);

    // Display the DTW result matrices for each pair of time series
    for (int i = 0; i < numTimeSeries; i++) {
        printf("DTW Result Matrix for Time Series Pair %d:\n", i + 1);
        for (int j = 0; j < seqLengthA; j++) {
            for (int k = 0; k < seqLengthB; k++) {
                printf("%d ", hostResults[i * seqLengthA * seqLengthB + j * seqLengthB + k]);
            }
            printf("\n");
        }
        printf("\n");
    }

    free(hostSeqA);
    free(hostSeqB);
    free(hostResults);
    cudaFree(deviceSeqA);
    cudaFree(deviceSeqB);
    cudaFree(deviceResults);

    return 0;
}

