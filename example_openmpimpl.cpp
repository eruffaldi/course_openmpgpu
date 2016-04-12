#include <cmath>
#include <chrono>
#include <iostream>

void fx(int n, ..., double *sinTable, int size)
{
	  sinTable[n] = std::sin(2 * M_PI * n / size);	
}

void pfx(void * a)
{
	struct qq * q = (struct qq*)a;
	fx(q->n,q->sinTable,...)
}

int main()
{
	const int size = 32768;
	double sinTable[size];


	//#pragma omp parallel for
	for(int n=0; n<size; ++n)
	{
		#pragma omp task 
		struct qq q;
		q.n = n;
		q.sinTable = sinTable;
		q.size = size;
		alloc_thread(pfx,q)
	}
	#pragma omp barrier

	return 0;
}