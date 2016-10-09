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

std::vector<int>::iterator bb; // hack for diagnostics

// This is a non-stable quick sort
// can we use OpenMP tasks for this?
template <class I>
void qsort1(I b, I e,bool top = false)
{
	if(b == e)
		return;
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
	if(d > 20)
	{
		tbb::task_group g;
		g.run([&] { qsort1(ps[0],pe[0]); });
		g.run([&] { qsort1(ps[1],pe[1]); });
		g.wait();
	}
	else
	{
		qsort1(ps[0],pe[0]);
		qsort1(ps[1],pe[1]);
	}
}


int main(int argc, char * argv[])
{

	std::random_device rd();
	std::seed_seq seed1({1,2,3,4,5});
    std::mt19937 gen(seed1); // rd());
    std::uniform_int_distribution<> dis(1, 100);
	std::vector<int> q(argc == 1 ? 10000000 : atoi(argv[1]));

	std::cout << "problem size is " << q.size() << std::endl;
    for(int i = 0; i < q.size(); i++)
    	q[i] = dis(gen);


	//std::vector<int> q0 = q;
	//std::sort(q0.begin(),q0.end());
	//std::cout << "regular sort gives " << check(q0.begin(),q0.end()) << std::endl;
 	auto t00 = tbb::tick_count::now();
	int n = argc > 1 ? atoi(argv[1]) : tbb::task_scheduler_init::default_num_threads();
	tbb::task_scheduler_init init(n);
	bb = q.begin();
 	auto t0 = tbb::tick_count::now();
	qsort1(q.begin(),q.end(),true);

 	auto t1 = tbb::tick_count::now();
	std::cout << "parallel sort check " << check(q.begin(),q.end()) << " total " << (t1-t00).seconds() << " = net " << (t1-t0).seconds() << " + setup " << (t0-t00).seconds() << std::endl;
	return 0;
}