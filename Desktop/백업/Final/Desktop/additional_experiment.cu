#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BLOCK_SIZE 256

// CUDA kernel to calculate the pairwise distances and costs
__global__ void dtw_kernel(double *x, int xsize, double *y, int ysize, double *distance) {
    //Set some values to INFINITY
    //__shared__ double smem[1089];
    __shared__ double x_s[32];
    __shared__ double y_s[32];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i<xsize)
    {
        x_s[i]=x[i];
        printf("%d thread copying %f value\n", i, x_s[i]);
    }
    __syncthreads();

    //if (i<ysize){

    //     y_s[i]=y[i];
         //printf("%d thread copying %f value", i, y_s[i]);
    //}
    
    //__syncthreads();

    //if (i <= xsize) {
    //    //distance[i * (ysize + 1)] = INFINITY;
    //    smem[i * (ysize + 1)] = INFINITY;
	//printf("%d the value of 1st columns are: %f \n",i, distance[i * (ysize +1)]);
    //}
    //if (i <= ysize) {
        //distance[i] = INFINITY;
    //    smem[i]=INFINITY;
	//printf("%d the value of 1st rows are: %f \n",i, distance[i]);
    //}

    //distance[0]= 0;
    //smem[0]=0;

//    if (i+1 <= xsize) {
//	    for (int k=1; k<=ysize; k++){
//        	distance[(i+1) * (ysize + 1) +k] = fabs(x[i] - y[k-1]);
//		printf("%d, %f\n ", i, distance[(i+1) * (ysize + 1) + k]); 
//    }
//}

/*    for (int j=1; j<= xsize+ysize-1; j++) {//j는 iteration
	    if (i+1<=j){//i는 threadIdx.x
		 if (i>=xsize) { //if 최대 thread 개수를 넘어가면, continue하기
			   continue;
		    }

        // if (j>=11 && i<(j-10)){
        //     continue;
        // }
		   //iteration 1일때는 스레드 1만 돌고, iteration2일땐 1,2 iteration3일땐 1,2,3만 돌고있음 나머지는 놀고 있는 스레드들임
		    //distance[(i+1)*(ysize+1)+j-i] = fabs(x[j-i-1]-y[i]);
            smem[(i+1)*(ysize+1)+j-i] = fabs(x_s[j-i-1]-y_s[i]);
		    double cost = smem[i*(ysize+1)+j-i-1];
		    cost = fmin(cost, smem[(i+1)*(ysize+1)+j-i-1]);
		    cost = fmin(cost, smem[i*(ysize+1)+j-i]);
		    smem[(i+1)*(ysize+1)+j-i]+=cost;
            //smem[(i+1)*(ysize+1)+j-i]+=cost;
            distance[(i+1)*(ysize+1)+j-i]=smem[(i+1)*(ysize+1)+j-i];
		   // printf("The thread id is: %d, The value that it computed is: %f \n", i, smem[(i+1)*(ysize+1)+j-i]);
	    }	    
	} */
}

double dtw(double *x, int xsize, double *y, int ysize) {
    double *distance, *d_x, *d_y;
    //double endtime;
    //clock_t init, final;

    cudaMalloc((void **)&distance, (xsize + 1) * (ysize + 1) * sizeof(double));
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
    cudaMemcpy(result, &distance[xsize * (ysize + 1) + ysize], sizeof(double), cudaMemcpyDeviceToHost);

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
    double y[]= {10.648, 30.333, 53.143, 72.429, 83.952, 86.571, 80.095, 65, 44.905, 21.762, 2.1129, -10.226, -14.627, -13.883, -11.345, -10.661, -10.318, -10.424, -10.43, -10.377,-10.498,-10.687,-10.767,-10.571,-9.7884,-8.9894,-8.1481,-7.3651,-6.6984,-6.1587,-5.7196, -5.3492};
    int ysize = sizeof(y) / sizeof(double);

    for (int i=0; i < (o_xsize-32+1); i++) {
	    double x[32]={0};
	    int l = i;
	    for (int k=0; k<32; k++) {
		    //printf("%f ",  o_x[l]);
		    double temp = o_x[l];
		    x[k] = temp;
		    //printf("The value for k and i is %d, %d\n",k,l);
		    l++;
		    //printf("The value for l++ is %d\n", l);
	    }
	    for (int z=0; z<32; z++){
		//printf("%f ", x[z]);
	    }
	    //printf("\n");
	    int xsize = sizeof(x) / sizeof(double);
	    //init=clock();
	    double distance = dtw(x, xsize, y, ysize);
	    //final = clock() - init;
    	//endtime = (double)final / ((double)CLOCKS_PER_SEC);
    	//("Total Time %f\n", endtime);
	    //printf("DTW distance : %lf", distance);
	    //printf("\n");

    }
    //final=clock()-init;
    //endtime=(double)final/((double)CLOCKS_PER_SEC);
    //printf("Total Time: %lf\n", endtime);


    return 0;
}

