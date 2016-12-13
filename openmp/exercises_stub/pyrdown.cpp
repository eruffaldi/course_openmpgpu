/**
 * pyrdown is a OpenCV function that performs a given 5x5 convolution filter over an image and then downsamples it at 1/2
 * of the resolution
 *
 * buildPyramid consecutively performs this operation
 *
 * Required:
 * - opencv matrix as general tensor 
 */
#include "imagex.hpp"

template <class T, class UT>
template <class R>
RT Image<T,UT>::conv(const Image & other, T scaling)
{
	if(other.numel() != numel() || other.width() != width())
		throw std::exception("size mismatch");
	R out = 0; // TODO assignment for non scalar

	auto Ait = byrow_iterator_pair();
	auto Bit = other.byrow_rev_iterator_pair();
	for(; Ait.first != Ait.second;)
		out += *Ait.first++ * *Bit.second++;
	return out/scaling;
}

#if 0



#endif
template <class IM>
void gaussianConvolve(IM src, IM & dst)
{
	int dw = dst.width();
	int dh = dst.height();
	IM mask(5,5);
	mask = {1,4,6,4,1,  4,16,24,16,4,   6,24,36,24,6,   4,16,24,16,4,  1,4,6,4,1};
	
	// zero padding for policy around

	#pragma omp parallel for collapse(2) shared(dst)
	for(int i = 1; i < dh/2-1; i++)
	{
		for(int j = 1; j < dw/2-1; j++)
		{
			// (i,j) is the output computed over the full image
			// min is: i=1      ==>   1*2-2 = 0 OK
			// max is: i=dh/2-1 ==>   (dh/2*2-2-2)+2 = (dh-2-2+2) = dh-2 when EVEN
			dst(i,j) = mask.corr(src.region(i*2-2,j*2-2,5,5)),256);
		}
	}
}
#endif


int main(int argc, char * argv[])
{
	/*
	decltype(src) dst(src.width/2,src.height/2);
	gaussianConvolve(src,dst);
	imwrite(dst,"output.png");
	*/
	return 0;
}