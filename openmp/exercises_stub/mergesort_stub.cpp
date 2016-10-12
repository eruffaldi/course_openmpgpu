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
#include <chrono>
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
	mergesort(X, n/2, tmp);
	mergesort(X+(n/2), n-(n/2), tmp+n/2);
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

	auto start = std::chrono::high_resolution_clock::now();
	mergesort(&q[0],q.size(),&qt[0]);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> diff = end-start;
	std::cout << "parallel sort gives " << check(q.begin(),q.end()) << " in " << diff.count() << std::endl;
}