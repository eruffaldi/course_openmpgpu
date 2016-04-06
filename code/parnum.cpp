#include <cmath>
#include <chrono>
#include <iostream>
#include <sstream>
#include <omp.h>

int main()
{

	#pragma omp parallel
	{
		std::ostringstream ons;
		ons << omp_get_thread_num() << " over " << omp_get_num_threads() << std::endl;
		std::cout << ons.str();
	}
	return 0;
}