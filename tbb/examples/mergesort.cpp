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

int nitems = 10000000;
int reclimit = 5000;
int mergelimit = 10000;

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
}

/*
 X[0:n/2] and X[n/2:n] are sorted, can we merge them provided that we are supporting stable sort
 Let's suppose we can divide an conquer as  ABCD with  A<=B and C<=D then we do

 AC in parallel to BD, and then merge them: 
 */
template <class T>
void	merge(T *X, T * Y, int nx, int ny, T *tmp)
{
	if(nx+ny < mergelimit)
	{
		mergesingle(X,Y,nx,ny,tmp);		
		return;
	}

	// X always bigger
	if(nx < ny)
	{
		std::swap(X,Y);
		std::swap(nx,ny);
	}

	int r = nx/2;
	auto s = std::lower_bound(Y,Y+ny,X[r])-Y; // relative index

	// special cases s==ny s==0
	// merge X[0..r] with Y[0..s]
	// add X[r] to out[t]
	// merge X[r+1..] with Y[s..]

	int t = r + s;
	tmp[t] = X[r];

	tbb::task_group g;
	g.run([&] { merge(X,Y, r,s, tmp);});
	g.run([&] { merge(X+r+1,Y+s,nx-r-1,ny-s,tmp+t+1);});
	g.wait();		
}




template <class T>
void	mergesort(T *X, int n, T *tmp)
{
	if (n < 2) return;

	if(n < reclimit )
	{
		mergesort(X, n/2, tmp);
		mergesort(X+(n/2), n-(n/2), tmp+(n/2));
		mergesingle(X, X+n/2, n/2,n-n/2, tmp);
		memcpy(X,tmp,sizeof(T)*n);
	}
	else
	{
		tbb::task_group g;
		g.run([&] { mergesort(X, n/2, tmp);});
		g.run([&] { mergesort(X+(n/2), n-(n/2), tmp+(n/2)); });
		g.wait();		
		merge(X, X+n/2, n/2,n-n/2, tmp);
		memcpy(X,tmp,sizeof(T)*n);
	}


}

int main(int argc, char * argv[])
{
    if((argc == 2 && strcmp(argv[1],"--help") == 0) || argc > 5)
	{
		std::cerr << "test [cpus [nitems [reclimit [mergelimit]]]]";
		return -1;
	}
	int ncpus = tbb::task_scheduler_init::default_num_threads();
	if(argc > 1)
		ncpus = std::atoi(argv[1]);
	if(argc > 2)
		nitems = std::atoi(argv[2]);
	if(argc > 3)
		reclimit = std::atoi(argv[3]);
	if(argc > 4)
		mergelimit = std::atoi(argv[4]);
	std::cout << "running with ncpus:" << ncpus << " nitems:" << nitems << " reclimit:" << reclimit << " mergelimit:" << mergelimit << std::endl;


	std::random_device rd;
	std::seed_seq seed1({1,2,3,4,5});
    std::mt19937 gen(seed1); // rd());
//    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);
	std::vector<int> q(nitems),qt(q.size());

	// QUESTION: can we parallelize this?
    for(int i = 0; i < q.size(); i++)
    	q[i] = dis(gen);


    std::cout << "start\n";
 	auto t00 = tbb::tick_count::now();
	tbb::task_scheduler_init init(ncpus);
 	auto t0 = tbb::tick_count::now();
	mergesort(&q[0],q.size(),&qt[0]);
 	auto t1 = tbb::tick_count::now();
	std::cout << "parallel sort check " << check(q.begin(),q.end()) << " total " << (t1-t00).seconds() << " = net " << (t1-t0).seconds() << " + setup " << (t0-t00).seconds() << std::endl;
	return 0;
}