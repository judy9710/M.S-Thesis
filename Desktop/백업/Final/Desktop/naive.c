#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

double min(double a, double b, double c) {
    if (a < b && a < c)
        return a;
    else if (b < a && b < c)
        return b;
    else
        return c;
}

double dtw(double *x, int xsize, double *y, int ysize) {
    double **distance;
    double **dtw_matrix;
    int i, j;

    double endtime;
    clock_t init, final;	

    // Allocate memory for the distance
    distance = (double **)malloc((xsize + 1) * sizeof(double *));
    
    for (i = 0; i <= xsize; ++i) {
        distance[i] = (double *)malloc((ysize + 1) * sizeof(double));
    }

    init=clock();
    
    // Initialize the distance matrix
    for (i = 1; i <= xsize; ++i)
        distance[i][0] = INFINITY;
    for (j = 1; j <= ysize; ++j)
        distance[0][j] = INFINITY;
    distance[0][0] = 0.0;

    // Calculate the pairwise distances between elements of x and y
    for (i = 1; i <= xsize; ++i) {
        for (j = 1; j <= ysize; ++j) {
            double cost = fabs(x[i-1] - y[j-1]);
            distance[i][j] = cost; //store the pairwise distance
	    double min_cost;
	    min_cost = fmin(distance[i-1][j], distance[i][j-1]);
	    min_cost = fmin(min_cost, distance[i-1][j-1]);
	    distance[i][j] += min_cost;

            //printf("The naive implementation of cost matrix is: %f \n", distance[i][j]);
        }
        
    }

    final= clock() - init;

    endtime = (double)final / ((double)CLOCKS_PER_SEC);
    printf("Total Time %f\n", endtime);

    double dtw_distance = distance[xsize][ysize];

    // Free allocated memory
    for (i = 0; i <= xsize; ++i) {
        free(distance[i]);
    }
    free(distance);

    return dtw_distance;
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
	    printf("DTW distance : %lf", distance);
	    printf("\n");

    }

    return 0;
}
