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
#include "tbb/task_group.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/tick_count.h"


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
void	mergesingle(T *X, T*Y, int nx, int ny, T *tmp)
{
	int	i = 0;
	int	j = 0;
	int	ti = 0;

	while (i<nx && j<ny) {
		if (X[i] < Y[j]) {
			tmp[ti] = X[i];
			ti++; i++;
		}
		else {
			tmp[ti] = Y[j];
			ti++; j++;
		}
	}

	while (i<nx) { /* finish up lower half */
		tmp[ti] = X[i];
		ti++; i++;
	}
	while (j<ny) { /* finish up upper half */
		tmp[ti] = Y[j];
		ti++; j++;
	}
	memcpy(X, tmp, ti*sizeof(T));
}


template <class T>
void	mergesort(T *X, int n, T *tmp)
{
	if (n < 2) return;

	if(n < 100 )
	{
		mergesort(X, n/2, tmp);
		mergesort(X+(n/2), n-(n/2), tmp+(n/2));
	}
	else
	{
		tbb::task_group g;
		g.run([&] { mergesort(X, n/2, tmp);});
		g.run([&] { mergesort(X+(n/2), n-(n/2), tmp+(n/2)); });
		g.wait();		
	}

	mergesingle(X, X+n/2, n/2,n-n/2, tmp);		
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

    std::cout << "start\n";
 	auto t00 = tbb::tick_count::now();
	int n = argc > 2 ? atoi(argv[2]) : tbb::task_scheduler_init::default_num_threads();
	tbb::task_scheduler_init init(n);
 	auto t0 = tbb::tick_count::now();
	mergesort(&q[0],q.size(),&qt[0]);
 	auto t1 = tbb::tick_count::now();
	std::cout << "parallel sort check " << check(q.begin(),q.end()) << " total " << (t1-t00).seconds() << " = net " << (t1-t0).seconds() << " + setup " << (t0-t00).seconds() << std::endl;
	return 0;
}