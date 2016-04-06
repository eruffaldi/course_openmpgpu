//http://nothings.org/stb/stb_image_write.h
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "stdio.h"
#include "omp.h"
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


int main(int argc, char const *argv[])
{
	int n = atoi(argv[1]);
	omp_set_num_threads(n);
	omp_set_dynamic(0);
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
	double t0 = omp_get_wtime();
	#pragma omp parallel for shared(image) firstprivate(h,w) collapse(2)
	for(int i = 0; i < h; i++)
		for(int j = 0; j < w; j++)
		{
			int k = mandel(i*((pmax-pmin)/h)+pmin,j*((qmax-qmin)/w)+qmin, maxv, maxk);
			image[i*w+j] = k >= maxk ? 0 : k;
			image2[i*w+j] = omp_get_thread_num()*256/n;
		}

	stbi_write_png("out.png",w,h,1,&image[0],w);
	stbi_write_png("out2.png",w,h,1,&image2[0],w);
	double t1 = omp_get_wtime();
	printf("Output is %f and warmup %f\n",t1-t0,t0-tp);
	return 0;
}