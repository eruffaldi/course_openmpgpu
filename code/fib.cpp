#include <iostream>
#include <chrono>
#include "stdlib.h"

int fib(int n)
{
	if(n < 2)
		return n;
	int x = 0,y = 0;
	// no paralell construct here so default is firstprivate
	#pragma omp task shared(x) if(n > 5)
	x = fib(n-1);
	#pragma omp task shared(y) if(n > 5)
	y = fib(n-2);
	#pragma omp taskwait
	return x + y;
}

int main(int argc, char const *argv[])
{
	int n = argc == 1 ? 10 : atoi(argv[1]);

	auto before = std::chrono::high_resolution_clock::now();
	std::cout << "doing " << n << " " << fib(n) << std::endl;
	auto after = std::chrono::high_resolution_clock::now();
	std::cout << "elapsed us: " << (after-before).count() << std::endl;
	return 0;
}