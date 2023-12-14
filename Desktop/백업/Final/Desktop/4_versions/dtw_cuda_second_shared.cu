#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>

#define NUM_MATRICES 51
#define MATRIX_SIZE 10
#define THREADS_PER_BLOCK 256

__global__ void generateMatrix(double* d_matrices, int num_matrices, double *y, int ysize, double *temp) {
    __shared__ double smem[NUM_MATRICES * MATRIX_SIZE * MATRIX_SIZE];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
 
    if (tid< num_matrices){
	    double first = fabs(temp[tid*10]-y[0]);
	    smem[tid*100]=first;
	    // fill in the first row
	    for (int i=1; i<10; i++){
		    double diff = fabs(temp[tid*10+i]-y[0]);
		    smem[i+100*tid]=diff+smem[i+100*tid-1];
		    //printf("%lf ", d_matrices[i]);
	    }
	    //printf("\n");
	    //fill in the first column
	    for (int i=10; i<100; i+=10){
		    double diff = fabs(temp[tid*10]-y[i/10]);
		    smem[i+100*tid]=diff+smem[i+100*tid-10];
		    //printf("%lf ", d_matrices[i]);
	    }
	    //printf("\n");

	    //now, compute the rest of the values
	    for (int i=1; i<10; i++){
		    for (int j=1; j<10; j++){
			    double min_cost;
			    min_cost=fmin(smem[i*10+j-1+tid*100], smem[i*10+j-10+tid*100]);
			    min_cost=fmin(smem[i*10+j-11+tid*100], min_cost);
			    smem[i*10+j+tid*100]=fabs(temp[tid*10+j]-y[i])+min_cost;
                d_matrices[i*10+j+tid*100]=smem[i*10+j+tid*100];
			    //printf("%lf ", d_matrices[i*10+j+tid*100]);
		    }
		    //printf("\n");
	    }
    }

}

int main() {
    double* d_matrices;
    double *d_x, *d_y;
    double *temp;

    double endtime;
    clock_t init, final;

    //init=clock();

    int matrices_size = NUM_MATRICES * MATRIX_SIZE * MATRIX_SIZE * sizeof(double);

    FILE *file;
    char filename[]="ECG_Data";  
    double o_x[60];
    int i=0;
    
    //read the o_x file

    file=fopen(filename, "r");

    if (file==NULL){
        perror("Error opening file");
        return 1;
    }

    while(fscanf(file, "%lf", &o_x[i])==1) {
        i++;
    }

    fclose(file);
    int numValues=i;

    printf("Read %d double values from the file:\n", numValues);
    for (i=0; i<numValues; i++){
        //printf("%lf ", o_x[i]);
    }

    int o_xsize = sizeof(o_x) / sizeof(double);

    // query set

    double y[] = {10.648, 30.333, 53.143, 72.429, 83.952, 86.571, 80.095, 65, 44.905, 21.762};
    int ysize = sizeof(y) / sizeof(double);


    // Allocate memory on the device (GPU) for the matrices
    cudaMalloc((void**)&d_matrices, matrices_size);

    cudaMalloc((void**)&d_x, o_xsize*sizeof(double));
    cudaMalloc((void**)&d_y, ysize*sizeof(double));

    cudaMalloc((void**)&temp, NUM_MATRICES*MATRIX_SIZE*sizeof(double));
    int count=0;
    for (int i=0; i<(o_xsize-10+1); i++){
	    	    count++;
                    int l=i;
		    double x[10]={0};
                    for(int k=0; k<10; k++){
                            double x_value = o_x[l];
                            x[k]=x_value;
                            l++;
		    }
		    for (int j=0; j<NUM_MATRICES*MATRIX_SIZE; ){
			    if (count>=0){
				    j+=((count-1)*10);
				    //printf("the index is, %d\n", j);
			    }
			    for (int l=0; l<10; l++){
				    cudaMemcpy(&temp[j], &x[l], sizeof(double), cudaMemcpyHostToDevice);
				    j++;
			    }
		    }
		    continue;
		    
        }
    

    cudaMemcpy(d_x, o_x, o_xsize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, ysize * sizeof(double), cudaMemcpyHostToDevice);

    init=clock();

    // Launch the kernel to generate matrices concurrently on the GPU
    generateMatrix<<<(NUM_MATRICES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_matrices, NUM_MATRICES, d_y, ysize, temp);

    final=clock()-init;
	endtime = (double)final / ((double)CLOCKS_PER_SEC);

    // Wait for the GPU to finish
    cudaDeviceSynchronize();
	    
    //printf("DTW distance : %lf", distance);
	//printf("\n");

    // Copy the matrices back to the host (CPU)
    double* h_matrices = (double*)malloc(matrices_size);
    cudaMemcpy(h_matrices, d_matrices, matrices_size, cudaMemcpyDeviceToHost);

    // Print the matrices on the CPU (host)
    for (int matrix_num = 0; matrix_num < NUM_MATRICES; ++matrix_num) {
        //printf("Matrix %d:\n", matrix_num + 1);
        //for (int i = 0; i < MATRIX_SIZE; ++i) {
          //  for (int j = 0; j < MATRIX_SIZE; ++j) {
                //printf("%lf ", h_matrices[matrix_num * MATRIX_SIZE * MATRIX_SIZE + i * MATRIX_SIZE + j]);
            //}
            //printf("\n");
       //}
        //printf("\n");
        printf("DTW distance: %lf\n", h_matrices[(matrix_num+1)* MATRIX_SIZE * MATRIX_SIZE-1]);
    }

    //final=clock()-init;

    //endtime = (double)final / ((double)CLOCKS_PER_SEC);
    printf("Total Time %f\n", endtime);

    // Free the memory on the device
    cudaFree(d_matrices);

    return 0;
}

