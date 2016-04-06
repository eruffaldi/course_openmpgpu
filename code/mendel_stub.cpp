//http://nothings.org/stb/stb_image_write.h
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "stdio.h"
#include <vector>

int mandel(double p, double q, double maxv, int maxk)
{
}


int main(int argc, char const *argv[])
{
	double pmin = -2.25, pmax = 0.75, qmin = -1.5, qmax = 1.5;
	double maxv = 100;
	int maxk = 256;
	int w = 640;
	int h = 480;
	std::vector<unsigned char> image(w*h);
	for(int i = 0; i < h; i++)
		for(int j = 0; j < w; j++)
		{
			int k = mandel(i*((pmax-pmin)/h)+pmin,j*((qmax-qmin)/w)+qmin, maxv, maxk);
			image[i*w+j] = k >= maxk ? 0 : k;
		}

	stbi_write_png("out.png",w,h,1,&image[0],w);
	return 0;
}