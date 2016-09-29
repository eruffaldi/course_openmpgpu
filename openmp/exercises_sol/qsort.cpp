/*
 * Emanuele Ruffaldi 2016
 
 * As several algorithms also quick sort can be paralellized
 *
 * Take the following quicksort serial and paralellize it. Which other sorting algorithms can be parallelized
 */
#include <omp.h>
#include <iostream>
#include <iterator>
#include <vector>
#include <list>
#include <random>
#include <algorithm>

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

std::vector<int>::iterator bb; // hack for diagnostics

// This is a non-stable quick sort
// can we use OpenMP tasks for this?
template <class I>
void qsort1(I b, I e)
{
	if(b == e)
		return;
	//std::cout << "omp_get_thread_num " << omp_get_thread_num() << " " << (b-bb) << "/" << (e-bb) << std::endl;
	auto d = std::distance(b,e);
	auto p = *std::next(b, d/2);
	using W = decltype(p);
	// NOTE: see http://en.cppreference.com/w/cpp/algorithm/partition for alternative/better
	//C++14 auto mid1 = std::partition(b,e,[p] (const auto & em) { return em < p; });
	//C++14 auto mid2 = std::partition(mid1,e,[p] (const auto & em) { return !(p < em); });
	auto mid1 = std::partition(b,e,[p] (const W & em) { return em < p; });
	auto mid2 = std::partition(mid1,e,[p] (const W & em) { return !(p < em); });
	#pragma omp task if (d > 20) untied
	{ 	
		qsort1(b,mid1);
	}
	// continue in this (tail parallelism)
	qsort1(mid2,e);
}


int main(int argc, char * argv[])
{
	std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);
	std::vector<int> q(10000000);

    for(int i = 0; i < q.size(); i++)
    	q[i] = dis(gen);

	//std::vector<int> q0 = q;
	//std::sort(q0.begin(),q0.end());
	//std::cout << "regular sort gives " << check(q0.begin(),q0.end()) << std::endl;
	double t0,t1;
	#pragma omp parallel
	{
		#pragma omp single
		{
			t0 = omp_get_wtime()
			bb = q.begin();
			std::cout << "starting with " << omp_get_num_threads() << std::endl;
			qsort1(q.begin(),q.end());
		}
		#pragma omp barrier
	}
	t1 = omp_get_wtime();
	std::cout << "parallel sort gives " << check(q.begin(),q.end()) << " duration " << t1-t0 << std::endl;
	return 0;
}