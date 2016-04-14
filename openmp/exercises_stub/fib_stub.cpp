#include <omp.h>
#include <stdio.h>

// omp_get_wtime()

int fib(int n)
{
	// usi openmp
}

int pfib(int q)
{
	// ... something ...
}

 int main(int argc, char const *argv[])
{
	int n = pfib(atoi(argv[1]));
	printf("hello %d\n",n);
	return 0;
}