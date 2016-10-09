#pragma once
#include <omp.h>

bool qsortstat_debug(int what)
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
void qsortstat(I b, I e, I bb)
{
	if(!qsortstat_debug(-1))
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