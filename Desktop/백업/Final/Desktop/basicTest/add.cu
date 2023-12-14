__global__ void add(int *a, int *b, int *c) {
	*c= *a + *b;
}

#include <stdio.h>
#include <math.h>

int main(void) {
	int a, b, c;
	int *dev_a, *dev_b, *dev_c;
	int size = sizeof(int);

	//allocate device copies of a, b, c
	cudaMalloc((void **)&dev_a, size);
	cudaMalloc((void **)&dev_b, size);
	cudaMalloc((void **)&dev_c, size);

	a=2;
	b=7;

	//copy inputs to device
	cudaMemcpy(dev_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, &b, size, cudaMemcpyHostToDevice);

	add<<<1,1>>>(dev_a, dev_b, dev_c);

	//copy device result back to host copy of c
	cudaMemcpy(&c, dev_c, size, cudaMemcpyDeviceToHost);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	printf("the value of c is %d\n", c);
	return 0;
}
