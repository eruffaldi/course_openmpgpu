/*
 * Emanuele Ruffaldi 2016
 *
 * It is known that the following integral has result pi
 *
 * integral[0,1, 4/(1+x*x)] = 4(atan(1)-atan(0)) = pi
 *
 * Objective:
 * - make it as OpenMP
 */
#include <math.h>
#include <iostream>
#include <limits>

/// returns a functional that changes the interval from (a,b) to [-1,1]
template <class F>
class change_interval_one_
{
public:
	change_interval_one_(double a, double b, F f): f_(f),s_((b-a)/2),q_((a+b)/2) {}

	double operator() (double x) const
	{
		return s_*f_(s_*x+q_);
	}

	double s_,q_;
	F f_;
};

template <class F>
auto change_interval_one(double a, double b, F f) -> change_interval_one_<F>
{
	return change_interval_one_<F>(a,b,f);
}

// rectangular
template <class F>
double rect_integrate(double a, double b, F f)
{
	return (b-a)*f((a+b)/2);
}

// trapezoidal
template <class F>
double trap_integrate(double a, double b, F f)
{
	return (b-a)*(f(a)+f(b))/2;
}

// composite integration rule
template <class F>
double comp_integrate(double a, double b, F f, int n)
{
	double s = 0;
	for(int k = 1; k < n; k++)
		s += f(a + k*(b-a)/n);
	return (b-a)/n*(f(a)/2+s+f(b)/2);
}

double pi_integral()
{
//	return comp_integrate(0,1,[] (double x) { return 4.0/(1+x*x); },2000);
	return comp_integrate(-1,1,change_interval_one(0,1,[] (double x) { return 4.0/(1+x*x); }),2000);
}


int main(int argc, char ** argv)
{
	double r = pi_integral();
	std::cout.precision(std::numeric_limits< double >::max_digits10);
	double e = fabs(r-M_PI);
	std::cout << "result " << r << " with e " << e << std::endl;
	return 0;
}