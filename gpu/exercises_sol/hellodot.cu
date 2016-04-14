#include <stdio.h>
#include <stdlib.h>

#define N 512

void random_ints(int * a, int q)
{
	for(int i = 0; i < q; i++)
		a[i] = 	rand() % 100;
}

__global__ void dot( int *a, int *b, int *c ) {
	__shared__ int temp[N];
	temp[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
	__syncthreads();
	if( 0 == threadIdx.x ) {
		int sum = 0;
		for( int i = 0; i < N; i++ )
		sum += temp[i];
		*c = sum;
	}
}

int cpu(int*a,int*b,int q)
{
	int r = 0,i;
	for(i = 0; i < q; i++)
		r +=a[i]*b[i];
	return r;
}

int main( void ) {
	int *a, *b, *c; // copies of a, b, c
	int *dev_a, *dev_b, *dev_c; // device copies of a, b, c
	int size = N * sizeof( int ); // we need space for 512 integers
	int i;
	// allocate device copies of a, b, c
	cudaMalloc( (void**)&dev_a, size );
	cudaMalloc( (void**)&dev_b, size );
	cudaMalloc( (void**)&dev_c, sizeof( int ) );
	a = (int *)malloc( size );
	b = (int *)malloc( size );
	c = (int *)malloc( sizeof( int ) );
	random_ints( a, N );
	random_ints( b, N );
	// copy inputs to device
	cudaMemcpy( dev_a, a, size, cudaMemcpyHostToDevice );
	cudaMemcpy( dev_b, b, size, cudaMemcpyHostToDevice );
	// launch dot() kernel with 1 block and N threads
	dot<<< 1, N >>>( dev_a, dev_b, dev_c );
	// copy device result back to host copy of c
	cudaMemcpy( c, dev_c, sizeof( int ) , cudaMemcpyDeviceToHost );
	printf("GPU:%d CPU:%d\n",c[0],cpu(a,b,N));
	free( a ); free( b ); free( c );
	cudaFree( dev_a );
	cudaFree( dev_b );
	cudaFree( dev_c );
	return 0;
}