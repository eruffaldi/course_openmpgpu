#include <cmath>
#include <chrono>
#include <memory.h>
#include <iostream>
#include "omp.h"
int main()
{
		const int size = 10;
		double sinTable[size];
		memset(sinTable,0,sizeof(sinTable));
		auto before1 = omp_get_wtime();
		
		#pragma omp parallel
		{
			#pragma omp single
			{
				std::cout << "single " << omp_get_thread_num() << std::endl;
				for(int n=0; n<size; ++n)
				{
					#pragma omp task default(none) shared(sinTable) firstprivate(n) shared(std::cout)
					{
						#pragma omp critical
						{
							std::cout << "thread " << omp_get_thread_num() << " iter " << n << " ptr " << sinTable << std::endl;
						}
						sinTable[n] = std::sin(2 * M_PI * n / size);
					}
				}
				#pragma omp taskwait
			}

		}
		auto after1 = omp_get_wtime();

		std::cout << "elapsed omp(s): " << (after1-before1) << std::endl;
		for(int n=0; n<size; ++n)
		{
			std::cout << "result " << n << " " << sinTable[n] << std::endl;
		}
		return 0;

}