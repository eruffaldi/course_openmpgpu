#include "stb_image_write.h"
#include "stdio.h"
#include <vector>


int mandel(double p, double q, double maxv, int maxk)
{
	double x0 = p;
	double y0 = q;
	double r;
	int k = 0;
	do
        {
            double x = x0 * x0 - y0 * y0 + p;
            double y = 2 * x0 * y0 + q;
            x0 = x;
            y0 = y;
           	r = x * x + y * y;
            k++;
        }
        while (r <= maxv && k < maxk);
     return k;
}


void __global__ kernel(...)
{
}

int main(int argc, char const *argv[])
{
	double pmin = -2.25, pmax = 0.75, qmin = -1.5, qmax = 1.5;
	double maxv = 100;
	int maxk = 128;
	int w = 640;
	int h = 480;
// 			int k = mandel(i*((pmax-pmin)/h)+pmin,j*((qmax-qmin)/w)+qmin, maxv, maxk);


 	//printf ("Time for the kernel: %f ms\n", cutGetTimerValue(kernelTime));
 	stbi_write_png("outGPU.png",w,h,1,&image[0],w);
///	double t1 = omp_get_wtime();
	//printf("Output is %f and warmup %f\n",t1-t0,t0-tp);
	return 0;
}
