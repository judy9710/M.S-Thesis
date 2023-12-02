/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

/* Template project which demonstrates the basics on how to setup a project 
* example application.
* Host code.
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include <time.h>

// includes, project
//#include <cutil_inline.h>
#include <cuda.h>
//#include <cutil.h>

const long int testSize = 8000;
const long int trainSize = 1024;
const int blockSize = 512;
const int window_size = 1024;
__constant__ long int N = 1048576;


__global__ void DTW (float* data_in, float* query, float* data_out, const int window_size)
{
	int k, l , g;
	
	long long int i,j;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	float min_nb;
	float array[1024][2];
    float instance[1024];

	float sum = 0, sum_sqr = 0, mean = 0, mean_sqr = 0, variance = 0, std_dev = 0;
	
//	int s = 2*window_size*(idx/window_size);
	int t = idx;
    if(idx+window_size>N-1)
       return;

	for(i=t; i<t+window_size; i++)
	{
		sum += data_in[i];
		sum_sqr += data_in[i] * data_in[i];
	}


	mean = sum / window_size;
	mean_sqr = mean*mean;
			
	variance = (sum_sqr/window_size) - mean_sqr;
	std_dev = sqrt(variance);


	i = 0;
	for(; i<window_size; i++)
		instance[i] = (data_in[t+i]-mean) / std_dev; 





	k = 0;
	l = 1;
	

	for(i=0; i<window_size; i++)
	{
		array[i][k] = abs(instance[i] - query[0]); 
	}

    k = 1;
    l = 0;
	
	for(j=1; j<window_size; j++)
	{
 		i = 0;
        array[i][k] = abs(instance[i] - query[j]) + array[i][l]; 

    	for (i=1; i<window_size; i++)
		{

			if (array[i-1][l] < array[i][l])
				min_nb = array[i-1][l];
			else 
				min_nb = array[i][l];

			if (array[i-1][k] < min_nb)
				min_nb = array[i-1][k];

			array[i][k] = fabs(instance[i] - query[j]) + min_nb;
		}
        g = k;
        k = l;
        l = g;
	}
	data_out[idx] = array[1023][k];
}



int main( int argc, char** argv)
{
	
	
	int i,j;
	float tmp;
	double endtime;
	clock_t init, final;
	

	if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
		cutilDeviceInit(argc, argv);
	else
		cudaSetDevice( cutGetMaxGflopsDeviceId() );


	unsigned int timer = 0, timermem=0;	
	unsigned long long int trainBytes = trainSize * window_size * sizeof(float);
	unsigned long long int testBytes = testSize * window_size * sizeof(float);
	
	float* h_train = (float*) malloc (trainBytes);
	int * trainLabels = (int *) malloc(trainSize*sizeof(int));
	float* h_test = (float*) malloc (testBytes);
    int * testLabels = (int *) malloc(testSize*sizeof(int));

    std::ifstream file;
	file.open("train.txt");
    for(i=0; i<trainSize; i++)
	{
		file >> tmp;   // label of the object
		trainLabels[i] = (int)tmp;
		for (j=0; j<window_size; j++)
		{
			file >> tmp;
			h_train[i*window_size+j] = tmp;
			//h_train[(2*i+1)*window_size+j] = tmp;
		}
	}
	file.close();

	std::ifstream qFile;
	qFile.open("test.txt");
    for(i=0; i<testSize; i++)
	{
		qFile >> tmp;

		testLabels[i] = (int)tmp;
		for (j=0; j<window_size; j++)
		{
			qFile >> tmp;
			h_test[i*window_size+j] = tmp;
		}
	}
	qFile.close();

	printf("Data has been read\n");
	cutCreateTimer(&timermem);
	cutStartTimer(timermem);
	init = clock();


	//Normalize the training set

	float* d_A = 0;
	cudaMalloc((void**)&d_A, trainBytes);
	cudaMemcpy(d_A, h_train, trainBytes, cudaMemcpyHostToDevice);
	printf("1\n");

	float* d_Out = 0;
	float* d_query = 0;

	cudaMalloc((void**)&d_query,window_size*sizeof(float));
	cudaMemcpy(d_query, h_test, window_size*sizeof(float), cudaMemcpyHostToDevice);
	
	cutCreateTimer(&timer);
	cutStartTimer(timer);

	cudaMalloc((void**)&d_Out, trainSize*window_size*sizeof(float));
	dim3 grid(trainSize*window_size/blockSize,1);
	dim3 threads(blockSize,1);
	
	
	
	
	DTW <<<grid, threads>>> (d_A, d_query, d_Out, window_size);
	

	cudaGetLastError();
	cudaThreadSynchronize();
	//cudaThreadSynchronize();

	printf("4\n");
	cutStopTimer(timer);
	printf("%f ms ", cutGetTimerValue(timer));
	cutDeleteTimer(timer);
	final = clock() - init;


	float* h_Out = (float*) malloc (trainSize*window_size*sizeof(float));
	cudaMemcpy(h_Out, d_Out, trainSize*window_size*sizeof(float) , cudaMemcpyDeviceToHost);

    for( i = 0 ; i < trainSize ; i++ )
         printf("%lf\n",h_Out[i]);


	cudaFree(d_A);
	cudaFree(d_Out);

	
	final = clock() - init;
	endtime = (double)final / ((double)CLOCKS_PER_SEC);
	printf("%f", endtime);
	

	cutStopTimer(timermem);
	printf("\nWith Memory: %f ms\n", cutGetTimerValue(timermem));
	cutDeleteTimer(timermem);
	

	return 0;

}	
