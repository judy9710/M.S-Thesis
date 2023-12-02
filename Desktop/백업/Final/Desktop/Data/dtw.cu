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


long int testSize = 5;
long int trainSize = 1024;
const int blockSize = 380;
const int window_size = 1024;

//cuFileHandle_t handle;
//cuFileInit(0, NULL, &handle); //Init cuFile


__global__ void DTW (float* data_in, float* query, float* data_out, const int window_size)
{
	int k, l , g;
	
	long long int i,j;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	float min_nb;
	float array[1024][2];
    float instance[1024];

	float sum = 0, sum_sqr = 0, mean = 0, mean_sqr = 0, variance = 0, std_dev = 0;
	
	int s = 2*window_size*(idx/window_size);
	int t = s + idx%window_size;

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

			array[i][k] = abs(instance[i] - query[j]) + min_nb;
		}
        g = k;
        k = l;
        l = g;
	}
	data_out[idx] = array[1023][k];
}



__global__ void ED (float* data_in, float* query, float* data_out, const int window_size)
{

	
	long long int i;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float instance[1024];

	float sum = 0, sum_sqr = 0, mean = 0, mean_sqr = 0, variance = 0, std_dev = 0;
	
	int s = 2*window_size*(idx/window_size);
	int t = s + idx%window_size;

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

	
    float sumErr  = 0;
	for(i=0; i < window_size; i++)
	{
		sumErr += (instance[i] - query[i])*(instance[i] - query[i]); 
	}
	
	data_out[idx] = sqrt(sumErr);

}


int main( int argc, char** argv)
{
	
	
	int i,j,cur;
	float tmp;
	double endtime;
	clock_t init, final;
	int dist = 0;
	
    if( argc >= 4 )	
        testSize = atoi(argv[3]); 
	
	cudaSetDevice(2); //특정한 디바이스 지정하는 것임

    //printf("%d\n",cutGetMaxGflopsDeviceId());
    
	unsigned long long int trainBytes = trainSize * 2 * window_size * sizeof(float);
	unsigned long long int testBytes = testSize * window_size * sizeof(float);
	
	float* h_train = (float*) malloc (trainBytes);
	int * trainLabels = (int *) malloc(trainSize*sizeof(int));
	float* h_test = (float*) malloc (testBytes);
    int * testLabels = (int *) malloc(testSize*sizeof(int));

	std::ifstream file; //reading data from files: ifstream 사용함
    
	//printf("%s\n",argv[1]);
    	//cuFileDescr_t fileDescr;
	//cuFileOpen(argv[1], O_RDONLY, 0, &fileDescr);
	file.open(argv[1]);
        printf("opened the file\n");	//time series file
	if(!file){
	      printf("error occured while opening the file");
	      exit(1);
	}
    for(i=0; i<trainSize; i++)
	{
		file >> tmp;   // label of the object
        	trainLabels[i] = (int)tmp;
	//	printf("%d\n", trainLabels[i]);
		for (j=0; j<window_size; j++)
		{
			file >> tmp;
			h_train[(2*i)*window_size+j] = tmp;
			h_train[(2*i+1)*window_size+j] = tmp;
		}
	}
	file.close();
    	//cuFileClose(fileDescr);
	//cuFileShutdown(handle);


	std::ifstream qFile;
	//printf("%s\n",argv[2]);
	qFile.open(argv[2]); //query file
	if(!qFile) exit(1);
    for(i=0; i<testSize; i++)
	{
		qFile >> tmp;
		testLabels[i] = (int)tmp;
		//printf("%d\n", testLabels[i]);
		for (j=0; j<window_size; j++)
		{
			qFile >> tmp;
			h_test[i*window_size+j] = tmp;
		}
	}
	qFile.close(); 
    if( argc >= 5 )
        dist = atoi(argv[4]);
	
    
	float* h_Out = (float*) malloc (trainSize*window_size*sizeof(float));
	float* d_A = 0;
	cudaMalloc((void**)&d_A, trainBytes);
	cudaMemcpy(d_A, h_train, trainBytes, cudaMemcpyHostToDevice);


	float* d_Out = 0;
	float* d_query = 0;
	cudaMalloc((void**)&d_query,window_size*sizeof(float));
   	cudaMalloc((void**)&d_Out, trainSize*window_size*sizeof(float));

	dim3 grid(trainSize*window_size/blockSize,1);
	dim3 threads(blockSize,1);


    init = clock();

	for( j = 0 ; j < trainSize ; j++ )
    cur = 0;
    int err = 0 , errNR = 0 , minI = -1 , minINR  = -1;
    while( cur < testSize ) 
    {
        	cudaMemcpy(d_query, h_test + window_size*cur , window_size*sizeof(float), cudaMemcpyHostToDevice);
       
            if( dist == 1)
               	DTW <<<grid, threads>>> (d_A, d_query, d_Out, window_size);
        	else
        	    ED <<<grid, threads>>> (d_A, d_query, d_Out, window_size);
            
                   
        	cudaGetLastError();
        	cudaThreadSynchronize();
        
        	cudaMemcpy(h_Out, d_Out, trainSize*window_size*sizeof(float) , cudaMemcpyDeviceToHost);
        	
        	float  min = 9999999999.99;
            minI = -1;
            float minNR = 99999999999.99;
            minINR = -1; 
            i = 0;
        for( j = 0 ; j < trainSize ; j++ )
        	{
                  if ( minNR > h_Out[j*window_size] )
                  {
                     minNR = h_Out[j*window_size];
                     minINR = j;
                  }
        	     for( i = 0 ; i < window_size ; i++ )
                 {
                      int t = j*window_size+i;
                      if ( min > h_Out[t] )
                      {
                         min = h_Out[t];
                         minI = j;
                      }
                 }
            }
            if( trainLabels[minI] != testLabels[cur] )
                err++;
          
            if( trainLabels[minINR] != testLabels[cur] )
                errNR++;

            printf("%d\t%d\tRI : %d\t%d\t%3.6f \t\t NRI : %d\t%d\t%3.6f\n",cur , testLabels[cur] , trainLabels[minI] ,  minI, min, trainLabels[minINR], minINR , minNR );
        cur++;
    }
	cudaFree(d_A);
	cudaFree(d_Out);
	cudaFree(d_query);
	
    free(trainLabels);
    free(testLabels);
	
	final = clock() - init;
	endtime = (double)final / ((double)CLOCKS_PER_SEC);
	printf("Total Time %f\n", endtime);
	printf("Rotation Invariant Accuracy is %f\n",(float)(testSize-err)*(100.0/testSize));
	printf("Regular Accuracy is %f\n",(float)(testSize-errNR)*(100.0/testSize));	
 
	return 0;

}	

