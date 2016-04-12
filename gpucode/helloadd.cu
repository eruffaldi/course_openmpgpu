	#include <stdio.h>
	__global__ void add( int *a, int *b, int *c ) {
		*c = *a + *b;
	}
	int main( void ) {
		int a=2, b=7, c; // host copies of a, b, c
		int *dev_a, *dev_b, *dev_c; // device copies of a, b, c
		int size = sizeof( int ); // we need space for an integer
		cudaMalloc( (void**)&dev_a, size );
		cudaMalloc( (void**)&dev_b, size );
		cudaMalloc( (void**)&dev_c, size );
		cudaMemcpy( dev_a, &a, size, cudaMemcpyHostToDevice );
		cudaMemcpy( dev_b, &b, size, cudaMemcpyHostToDevice );
		add<<< 1, 1 >>>( dev_a, dev_b, dev_c );
		cudaMemcpy( &c, dev_c, size, cudaMemcpyDeviceToHost );
		cudaFree( dev_a );
		cudaFree( dev_b );
		cudaFree( dev_c );
		printf("Result %d\n",c);
		return 0;
	}