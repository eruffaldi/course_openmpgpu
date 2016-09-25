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

template <class I>
I iternext(I q)
{
	return ++q;
}

template <class I>
I iterprev(I q)
{
	return --q;
}

template <class I>
void qsort1(I b, I e)
{
	return; // REMOVE ME
	if(b == e)
		return;
	auto p = *b; // PIVOT is FIRST POINT
	// TODO implement split around pivot and return sp
	auto sp = b;
	qsort1(b,iternext(sp));
	qsort1(iterprev(sp),e);
}


int main(int argc, char * argv[])
{
	std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);
	std::vector<int> q(1000);

    for(int i = 0; i < q.size(); i++)
    	q[i] = dis(gen);

	std::vector<int> q0 = q;
	std::sort(q0.begin(),q0.end());
	std::cout << "regular sort gives " << check(q0.begin(),q0.end()) << std::endl;
	qsort1(q.begin(),q.end());
	std::cout << "regular sort gives " << check(q.begin(),q.end()) << std::endl;
	return 0;
}