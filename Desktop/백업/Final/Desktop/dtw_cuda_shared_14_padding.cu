#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BLOCK_SIZE 256

// CUDA kernel to calculate the pairwise distances and costs
__global__ void dtw_kernel(double *x, int xsize, double *y, int ysize, double *distance) {
    //Set some values to INFINITY
    __shared__ double smem[240];
    __shared__ double x_s[14];
    __shared__ double y_s[14];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i<xsize)
    {
        x_s[i]=x[i];
        //printf("%d thread copying %f value", i, x_s[i]);
    }
    __syncthreads();

    if (i<ysize){

         y_s[i]=y[i];
         //printf("%d thread copying %f value", i, y_s[i]);
    }
    
    __syncthreads();

    if (i <= xsize) {
        //distance[i] = INFINITY;
        smem[i]=INFINITY;
	//printf("%d the value of 1st rows are: %f \n",i, smem[i]);
    }

    int new_ysize = ysize+2; // ysize now becomes 17

    if (i <= ysize) {
        //distance[i * (ysize + 1)] = INFINITY;
        smem[i * (new_ysize)] = INFINITY;
	//printf("%d the value of 1st columns are: %f \n",i, smem[i * (ysize +1)]);
    }

    for (int k=15; k<240; k+=16) {
        smem[k] = 0;
        //printf("%f", smem[k]);
        //printf("%f", smem[k+1]);

    } //did it manually, need to formularize...

    distance[0]= 0;
    smem[0]=0;

//    if (i<xsize) {
// 	    for (int k=1; k<=new_ysize; k++){
//        	smem[(i+1) * (new_ysize + 1) +k] = fabs(x_s[i] - y_s[k-1]);
// 		printf("%d, %f\n ", i, smem[(i+1) * (new_ysize + 1) + k]); 
//    }
// }

    //printf("%d, %d ", xsize, new_ysize);
    

    for (int j=1; j<= xsize+ysize-1; j++) {//j는 iteration
    //printf("starting");
	    if (i+1<=j){//i는 threadIdx.x
        //printf("starting");
		if (i>=xsize) { //if 최대 thread 개수를 넘어가면, continue하기
			   continue;
		    }
       // if (j>=32 && i<(j-31)){
       //     continue;
       // }
            smem[(i+1)*(new_ysize)+j-i] = fabs(x_s[j-i-1]-y_s[i]);
            //printf("starting");
		    double cost = smem[i*(new_ysize)+j-i-1];
		    cost = fmin(cost, smem[(i+1)*(new_ysize)+j-i-1]);
		    cost = fmin(cost, smem[i*(new_ysize)+j-i]);
		    smem[(i+1)*(new_ysize)+j-i]+=cost;
            distance[(i+1)*(new_ysize)+j-i]=smem[(i+1)*(new_ysize)+j-i];
		   // printf("The thread id is: %d, The value that it computed is: %f \n", i, smem[(i+1)*(new_ysize)+j-i]);
         
        }
		   
	    }	    
	}

double dtw(double *x, int xsize, double *y, int ysize) {
    double *distance, *d_x, *d_y;
    //double endtime;
    //clock_t init, final;

    cudaMalloc((void **)&distance, (xsize + 1) * (ysize + 2) * sizeof(double));
    cudaMalloc((void **)&d_x, xsize * sizeof(double));
    cudaMalloc((void **)&d_y, ysize * sizeof(double));

    cudaMemcpy(d_x, x, xsize * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, ysize * sizeof(double), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((xsize * ysize + BLOCK_SIZE - 1) / BLOCK_SIZE);

    //init=clock();

    dtw_kernel<<<dimGrid, dimBlock>>>(d_x, xsize, d_y, ysize, distance);

    //final=clock()-init;
    
    //endtime=(double)final/((double)CLOCKS_PER_SEC);
    //printf("Total Time: %lf\n", endtime);

    double *result = (double *)malloc(sizeof(double));
    cudaMemcpy(result, &distance[xsize * (ysize+2) + ysize], sizeof(double), cudaMemcpyDeviceToHost);
    //cudaMemcpy(result, &distance[131], sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(distance);
    cudaFree(d_x);
    cudaFree(d_y);

    return *result;
}

int main() {

    FILE *file;
    char filename[]="ECG_Data.txt";  
    double o_x[420];
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

    //printf("Read %d double values from the file:\n", numValues);
    for (i=0; i<numValues; i++){
        //printf("%lf ", o_x[i]);
    }

    int o_xsize = sizeof(o_x) / sizeof(double);

    // query set

    //double y[] = {10.648, 30.333, 53.143, 72.429, 83.952, 86.571, 80.095, 65, 44.905, 21.762};
    double y[] = {10.648, 30.333, 53.143, 72.429, 83.952, 86.571, 80.095, 65, 44.905, 21.762, 2.1129, -10.226, -14.627, -13.883};

    int ysize = sizeof(y) / sizeof(double);

    for (int i=0; i < (o_xsize-14+1); i++) {
	    double x[14]={0};
	    int l = i;
	    for (int k=0; k<14; k++) {
		    //printf("%f ",  o_x[l]);
		    double temp = o_x[l];
		    x[k] = temp;
		    //printf("The value for k and i is %d, %d\n",k,l);
		    l++;
		    //printf("The value for l++ is %d\n", l);
	    }
	    for (int z=0; z<14; z++){
		//printf("%f ", x[z]);
	    }
	    //printf("\n");
	    int xsize = sizeof(x) / sizeof(double);
	    //init=clock();
	    double distance = dtw(x, xsize, y, ysize);
	    //final = clock() - init;
    	//endtime = (double)final / ((double)CLOCKS_PER_SEC);
    	//("Total Time %f\n", endtime);
	    printf("DTW distance : %lf", distance);
	    printf("\n");

    }
    //final=clock()-init;
    //endtime=(double)final/((double)CLOCKS_PER_SEC);
    //printf("Total Time: %lf\n", endtime);


    return 0;
}

