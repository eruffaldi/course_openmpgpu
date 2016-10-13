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

int nitems = 10000000;
int reclimit = 5000;
int mergelimit = 10000;


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

	#pragma omp task 
	{
		merge(X,Y, r,s, tmp);
	}
	#pragma omp task 
	{
		merge(X+r+1,Y+s,nx-r-1,ny-s,tmp+t+1);
	}
	#pragma omp taskwait
}



template <class T>
void	mergesort(T *X, int n, T *tmp)
{
	if (n < 2) return;
	if(n > reclimit)
	{
		// we move above if(n > reclimit)
		#pragma omp task 
		{
			mergesort(X, n/2, tmp);
		}
		#pragma omp task
		{
			mergesort(X+(n/2), n-(n/2), tmp+n/2);
		}
		#pragma omp taskwait
		merge(X, X+n/2, n/2,n-n/2, tmp);
		memcpy(X,tmp,sizeof(T)*n);
	}
	else
	{
		mergesort(X, n/2, tmp);
		mergesort(X+(n/2), n-(n/2), tmp+(n/2));
		mergesingle(X, X+n/2, n/2,n-n/2, tmp);
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
	int ncpus = -1;
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
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);
	std::vector<int> q(10000000),qt(q.size());

	// QUESTION: can we parallelize this?
    for(int i = 0; i < q.size(); i++)
    	q[i] = dis(gen);

    if(ncpus != -1)
    	omp_set_num_threads(ncpus);
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