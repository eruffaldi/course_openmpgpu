#include <cmath>
#include <chrono>
#include <iostream>
#include "omp.h"
int main()
{
	const int size = 32768;
	double sinTable[size];

	auto before = std::chrono::high_resolution_clock::now();
	auto dt = [] (decltype(before) now) { return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()-now).count(); };
	auto before1 = omp_get_wtime();
	#pragma omp parallel for
	for(int n=0; n<size; ++n)
	  sinTable[n] = std::sin(2 * M_PI * n / size);

	auto after1 = omp_get_wtime();
	std::cout << "elapsed us: " << dt(before) << std::endl;
	std::cout << "elapsed omp(s): " << (after1-before1) << std::endl;
	return 0;
}