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


void __global__ kernel(unsigned char * out, int w,int h, double pmin, double pmax, double qmin, double qmax, double maxv, int maxk)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int index = j + i * w;

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
	int maxk = 128;
	int w = 640;
	int h = 480;

        cudaSetDevice(0);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);

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
	stbi_write_png("out.png",w,h,1,&image[0],w);

	// GPU
	// transfer

	cudaMalloc(&dev_i,w*h);
	cudaMemset(dev_i,128,w*h);
	// malloc
		unsigned int kernelTime;
 //cutCreateTimer(&kernelTime);
 //cutResetTimer(kernelTime);
	//cutStartTim,1);
	dim3 dimBlock(32, 32,1);
	dim3 dimGrid(iceil(w,dimBlock.x),iceil(h,dimBlock.y));
	printf("Blocks %d %d %d Threads %d %d %d\n",dimGrid.x,dimGrid.y,dimGrid.z,dimBlock.x,dimBlock.y,dimBlock.z);
	kernel<<<dimGrid,dimBlock>>>(dev_i,w,h,pmin,pmax,qmin,qmax,maxv,maxk);
	{
    cudaError_t cudaerr = cudaPeekAtLastError();
    if (cudaerr != 0)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
	}
	cudaMemcpy(&image[0],dev_i,w*h,cudaMemcpyDeviceToHost);
	//cutStopTimer(kernelTime);
	cudaFree(dev_i);
 	//printf ("Time for the kernel: %f ms\n", cutGetTimerValue(kernelTime));
 	stbi_write_png("outGPU.png",w,h,1,&image[0],w);
///	double t1 = omp_get_wtime();
	//printf("Output is %f and warmup %f\n",t1-t0,t0-tp);
	return 0;
}