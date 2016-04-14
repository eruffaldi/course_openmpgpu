//http://nothings.org/stb/stb_image_write.h
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "stdio.h"
#include <vector>

int __host__ __device__ mandel(double p, double q, double maxv, int maxk)
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

int iceil(int a, int b)
{
	return (a+b-1)/b;
}


void __global__ kernel(unsigned char * out, int w,int p,int h, double pmin, double pmax, double qmin, double qmax, double maxv, int maxk)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int index = j + i * p;

	if(i < h && j < w)
	{
		int k = mandel(i*((pmax-pmin)/h)+pmin,j*((qmax-qmin)/w)+qmin, maxv, maxk);
		//i*w+j
		out[index] = k >= maxk ? 0 : k;
	}
}

int main(int argc, char const *argv[])
{
	double pmin = -2.25, pmax = 0.75, qmin = -1.5, qmax = 1.5;
	double maxv = 100;
	int maxk = 256;
	int w = 640;
	int h = 480;
	//double tp = omp_get_wtime();
	std::vector<unsigned char> image(w*h);
	unsigned char * dev_i;
		//std::vector<unsigned char> image2(w*h);

	//double t0 = omp_get_wtime();
	for(int i = 0; i < h; i++)
		for(int j = 0; j < w; j++)
		{
			int k = mandel(i*((pmax-pmin)/h)+pmin,j*((qmax-qmin)/w)+qmin, maxv, maxk);
			image[i*w+j] = k >= maxk ? 0 : k;
			//image2[i*w+j] = omp_get_thread_num()*256/n;
		}
//	stbi_write_png("out.png",w,h,1,&image[0],w);

	// GPU
	// transfer
	size_t  pitch;
	cudaMallocPitch(&dev_i,&pitch,w,h);
	std::vector<unsigned char> imagei(pitch*h);
	// malloc
		unsigned int kernelTime;
 //cutCreateTimer(&kernelTime);
 //cutResetTimer(kernelTime);
	//cutStartTim,1);
	dim3 dimBlock(32, 32);
	dim3 dimGrid(iceil(w,dimBlock.x),iceil(h,dimBlock.y));
	printf("Blocks %d %d %d Threads %d %d %d Pitch %d\n",dimGrid.x,dimGrid.y,dimGrid.z,dimBlock.x,dimBlock.y,dimBlock.z,(int)pitch);
	kernel<<<dimGrid,dimBlock>>>(dev_i,w,pitch,h,pmin,pmax,qmin,qmax,maxv,maxk);
	cudaMemcpy(&imagei[0],dev_i,pitch*h,cudaMemcpyDeviceToHost);


	// transfer
//	cudaThreadSynchronize();
	//cutStopTimer(kernelTime);
	cudaFree(dev_i);
	stbi_write_png("outGPU.png",w,h,1,&imagei[0],pitch);
///	double t1 = omp_get_wtime();
	//printf("Output is %f and warmup %f\n",t1-t0,t0-tp);
	return 0;
}