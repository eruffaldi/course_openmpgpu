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
	decltype(mid1) ps[2] = { b,mid2};
	decltype(mid1) pe[2] = { mid1,e};
	#pragma omp parallelif (d > 20)
	{
		#pragma omp  for nowait 
		for(int i = 0; i < 2; i++)
			qsort1(ps[i],pe[i]);
	}
}


int main(int argc, char * argv[])
{
	std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);
	std::vector<int> q(argc == 1 ? 10000000 : atoi(argv[1]));

	std::cout << "problem size is " << q.size() << std::endl;
    for(int i = 0; i < q.size(); i++)
    	q[i] = dis(gen);

	//std::vector<int> q0 = q;
	//std::sort(q0.begin(),q0.end());
	//std::cout << "regular sort gives " << check(q0.begin(),q0.end()) << std::endl;
	omp_set_dynamic(1);
	double t0,t1,t00;
 	t00 = omp_get_wtime();
	std::cout << "starting with " << omp_get_num_threads() << std::endl;
	bb = q.begin();
 	t0 = omp_get_wtime();
	qsort1(q.begin(),q.end());
 	t1 = omp_get_wtime();
	std::cout << "parallel sort gives " << check(q.begin(),q.end()) << " total " << t1-t00 << " = net " << t1-t0 << " + setup " << t0-t00 << std::endl;
	return 0;
}