/*
 * Emanuele Ruffaldi 2016
 
 * As several algorithms also quick sort can be paralellized
 *
 * Take the following quicksort serial and paralellize it. Which other sorting algorithms can be parallelized
 */
#include <iostream>
#include <iterator>
#include <vector>
#include <list>
#include <random>
#include <memory.h>
#include <algorithm>
#include <omp.h>

bool mergesortstat_debug(int what)
{
	static bool dodebug = false;
	switch(what)
	{
		case 0: dodebug = false; break;
		case 1: dodebug = true; break;
		default: break;
	}
	return dodebug;
}

template <class I>
void mergesortstat(I b, I e, I bb)
{
	if(!mergesortstat_debug(-1))
		return;
	char buf[128];
	sprintf(buf,"Start-End %08d-%08d Team %02d Thread/Max %02d/%02d Level %02d Parent %02d\n",
		(int)(b-bb),(int)(e-bb),
		omp_get_team_num(),
		omp_get_thread_num(),
		omp_get_num_threads(),
		omp_get_level(),
		omp_get_ancestor_thread_num (omp_get_level())
		);
	#pragma omp critical (io)
	{
		std::cout << buf << std::flush;
	}
}

// check lexicographic ordering
template <class I>
bool check(I b, I e)
{
	if(b == e) return true;
	auto p = *b++;	
	while(b != e)
	{
		if(p > *b)
			return false;
		p = *b++;
	}
	return true;
}


template <class T>
void	merge(T *X, int n, T *tmp)
{
int	i = 0;
int	j = n/2;
int	ti = 0;

	while (i<n/2 && j<n) {
		if (X[i] < X[j]) {
			tmp[ti] = X[i];
			ti++; i++;
		}
		else {
			tmp[ti] = X[j];
			ti++; j++;
		}
	}

	while (i<n/2) { /* finish up lower half */
		tmp[ti] = X[i];
		ti++; i++;
	}
	while (j<n) { /* finish up upper half */
		tmp[ti] = X[j];
		ti++; j++;
	}
	memcpy(X, tmp, n*sizeof(T));
}

template <class T>
void	mergesort(T *X, int n, T *tmp)
{
	if (n < 2) return;
	#pragma omp task if(n > 10)
	{
		mergesort(X, n/2, tmp);
	}
	#pragma omp task if(n > 10)
	{
		mergesort(X+(n/2), n-(n/2), tmp+n/2);
	}
	#pragma omp taskwait
	merge(X, n, tmp);		
}

int main(int argc, char * argv[])
{
	std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);
	std::vector<int> q(10000000),qt(q.size());

	// QUESTION: can we parallelize this?
    for(int i = 0; i < q.size(); i++)
    	q[i] = dis(gen);

    omp_set_dynamic(1);
	double t0,t1,t00;
 	t00 = omp_get_wtime();
	#pragma omp parallel
	{
		#pragma omp master
		{
			t0 = omp_get_wtime();
			mergesort(&q[0],q.size(),&qt[0]);
		}
	}
	t1 = omp_get_wtime();
	std::cout << "parallel sort gives " << is_sorted(q.begin(),q.end()) << " total " << t1-t00 << " = net " << t1-t0 << " + setup " << t0-t00 << std::endl;
}