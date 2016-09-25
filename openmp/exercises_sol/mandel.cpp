// see details here: https://software.intel.com/en-us/articles/openmp-loop-scheduling
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "stdio.h"
#include "omp.h"
#include <vector>

/*	
	#pragma omp parallel for shared(image) firstprivate(h,w) schedule(runtime) shared(ww)
	Equivalent to: note default(shared) means &
		[&,h,w] (int i, int j) 
		{
			int k = mandel(i*((pmax-pmin)/h)+pmin,j*((qmax-qmin)/w)+qmin, maxv, maxk);
			image[i*w+j] = k >= maxk ? 0 : k;
			image2[i*w+j] = omp_get_thread_num()*256/n;
		}

	int qq = 20;
	#pragma omp parallel for shared(image) firstprivate(h,w) schedule(runtime) shared(ww) private(qq)
		[&,h,w] (int i, int j) 
		{
			int qq; // ADDED here
			int k = mandel(i*((pmax-pmin)/h)+pmin,j*((qmax-qmin)/w)+qmin, maxv, maxk);
			image[i*w+j] = k >= maxk ? 0 : k;
			image2[i*w+j] = omp_get_thread_num()*256/n;
		}
 */
struct W
{
	W(int k) { k_ = k; }
	int k_;
	W(const W & c) = delete;
};

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


int main(int argc, char const *argv[])
{
	int n = argc < 2 ? 2 : atoi(argv[1]);
	omp_set_num_threads(n);
	omp_set_dynamic(0);
	printf("set number of threads to %d\n",n);
	printf("disabled dynamic\n");
	double pmin = -2.25, pmax = 0.75, qmin = -1.5, qmax = 1.5;
	double maxv = 100;
	int maxk = 256;
	int w = 640;
	int h = 480;
	double tp = omp_get_wtime();
	std::vector<unsigned char> image(w*h);
	std::vector<unsigned char> image2(w*h);
	#pragma omp parallel
	{
		#pragma omp single
		{
			printf("warmup\n");
		}
	}
	printf("all grid testing using runtime (OMP_SCHEDULE) environment variable\n");
	double t0 = omp_get_wtime();
	#pragma omp parallel for shared(image) firstprivate(h,w) collapse(2) schedule(runtime)
	for(int i = 0; i < h; i++)
		for(int j = 0; j < w; j++)
		{
			int k = mandel(i*((pmax-pmin)/h)+pmin,j*((qmax-qmin)/w)+qmin, maxv, maxk);
			image[i*w+j] = k >= maxk ? 0 : k;
			image2[i*w+j] = omp_get_thread_num()*256/n;
		}

	stbi_write_png("outgrid_result.png",w,h,1,&image[0],w);
	stbi_write_png("outgrid_assign.png",w,h,1,&image2[0],w);
	double t1 = omp_get_wtime();
	printf("Output is %f and warmup %f\n",t1-t0,t0-tp);

	printf("only row testing using runtime (OMP_SCHEDULE) environment variable\n");
	{
	double t0 = omp_get_wtime();
	W ww(10);
	// NOTE: cannot use nor private or firstprivate for ww
	#pragma omp parallel for shared(image) firstprivate(h,w) schedule(runtime) shared(ww)
	for(int i = 0; i < h; i++)
		for(int j = 0; j < w; j++)
		{
			int k = mandel(i*((pmax-pmin)/h)+pmin,j*((qmax-qmin)/w)+qmin, maxv, maxk);
			image[i*w+j] = k >= maxk ? 0 : k;
			image2[i*w+j] = omp_get_thread_num()*256/n;
		}

	stbi_write_png("outrows_result.png",w,h,1,&image[0],w);
	stbi_write_png("outrows_assign.png",w,h,1,&image2[0],w);
	double t1 = omp_get_wtime();
	printf("Output is %f and warmup %f\n",t1-t0,t0-tp);
	}
	return 0;
}