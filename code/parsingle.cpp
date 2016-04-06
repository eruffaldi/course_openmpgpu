#include <iostream>

int main()
{
	#pragma omp parallel 
	{
		std::cout << "multiple \n";
		#pragma omp single
		{
			std::cout << "only one\n";
		}
	}
	return 0;
}