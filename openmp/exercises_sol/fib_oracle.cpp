#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

/*
https://docs.oracle.com/cd/E19205-01/820-7883/6nj43o69j/index.html
The data-sharing attributes of variables that are not listed in data attribute clauses of a task construct, and are not predetermined according to the OpenMP rules, are implicitly determined as follows:
(a) In a task construct, if no default clause is present, a variable that is determined to be shared in all enclosing constructs, up to and including the innermost enclosing parallel construct, is shared.
(b) In a task construct, if no default clause is present, a variable whose data-sharing attribute is not determined by rule (a) is firstprivate.

It follows that:
(a) If a task construct is lexically enclosed in a parallel construct, then variables that are shared in all scopes enclosing the task construct remain shared in the generated task. Otherwise, variables are implicitly determined firstprivate.
(b) If a task construct is orphaned, then variables are implicitly determined firstprivate.
 */

int fib(int n)
{
  int i, j;
  if (n<2)
    return n;
  else
    {
       #pragma omp task shared(i) 
       i=fib(n-1);

       #pragma omp task shared(j) 
       j=fib(n-2);

       #pragma omp taskwait
       return i+j;
    }
}

int main(int argc, char * argv[])
{
  int n = argc == 1 ? 10 : atoi(argv[1]);

  omp_set_dynamic(0);
  omp_set_num_threads(4);

  #pragma omp parallel shared(n)
  {
    #pragma omp single
    printf ("fib(%d) = %d\n", n, fib(n));
  }
}