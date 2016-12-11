/**
 * pyrdown is a OpenCV function that performs a given 5x5 convolution filter over an image and then downsamples it at 1/2
 * of the resolution
 *
 * buildPyramid consecutively performs this operation
 *
 * For borders OpenCV serveral methodologies
 *
 */
#include  <initializer_list>
#include  <algorithm>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/**
 * Simple image class with ROI and sharing
 *
 * TODO: support 2D stride 
 * TODO: support channels
 * TODO: support iterator by row
 */
template <class T, class UT = T>
struct Image
{
	using IT = std::vector<T>::iterator;
	using CIT = std::vector<T>::const_iterator;
	mutable std::shared_ptr<T> data_;
	int width_;
	int height_;
	int offset_;
	int stride_;

	Image(int w, int h);

	Image(int w, int h, const Image & other, int off);

	Image(int w, int h, Image & other, int off);

	int width() const { return width_; }

	int height() const { return height_; }

	int stride() const { r eturn stride_; }

	int numel() const { return width_*height_; }

	bool isContiguous() const { return stride_ == width_; }

	bool isRowContiguous() const { return true; }

	T * data() { return data_.get()+offset_; }

	const T * data() const { return data_.get()+offset_; }

	const Image region(int x, int y, int w, int h) const
	{
		return Image(w,h,*this,x+y*stride);
	}

	Image region(int x, int y, int w, int h)
	{
		return Image(w,h,*this,x+y*stride);
	}

	T & operator()(int x, int y)
	{
		return data()[offset_+x+y*stride_];
	}

	T & operator()(int d)
	{
		return data()[offset_+d];
	}

	const T & operator()(int x, int y) const
	{
		return data()[offset_+x+y*stride_];
	}

	const T & operator()(int d) const
	{
		return data()[offset_+d];
	}

	template <class R = UT>
	T conv(const Image & other, T scaling);

	template <class R = UT>
	T corr(const Image & other, T scaling);

	template <class Q>
	void operator= (std::initializer_list<Q> it);

	struct byrow_iterator
	{

	};

	struct bycol_iterator
	{
		
	};

	// used for convolution
	struct byrow_rev_iterator
	{

	};

	// used for convolution
	struct bycol_rev_iterator
	{
		
	};

	std::pair<IT,IT> byrow_iterator_pair();

	std::pair<IT,IT> bycol_iterator();
};


template <class T, class UT>
auto Image<T,UT>::byrow_iterator_pair() -> std::pair<IT,IT> 
{

}

template <class T, class UT>
auto  Image<T,UT>::bycol_iterator() -> std::pair<IT,IT> 
{

}

template <class T, class UT>
Image<T,UT>::Image(int w, int h) : width_(w),height_(h),data_(new std::vector<T>(w*h)),offset_(0) {}

template <class T, class UT>
Image<T,UT>::Image(int w, int h, const Image & other, int off): width_(w),height_(h),data_(other.data_),offset_(off) {}

template <class T, class UT>
Image<T,UT>::Image(int w, int h, Image & other, int off): width_(w),height_(h),data_(other.data_),offset_(off) {}

template <class T, class UT>
template <class R>
RT Image<T,UT>::conv(const Image & other, T scaling)
{
	if(other.numel() != numel() || other.width() != width())
		throw std::exception("size mismatch");
	R out = 0; // TODO assignment for non scalar

	// TODO: use zip iterator as in boost
	// TODO: some STL?
	auto Ait = byrow_iterator_pair();
	auto Bit = other.byrow_iterator_pair();
	for(; Ait.first != Ait.second;)
		out += *Ait.first++ * *Bit.second++;
	return out/scaling;
}

template <class T, class UT>
template <class Q>
void Image<T,UT>::operator= (std::initializer_list<Q> il)
{
	if(il.size() != numel())
	{
		throw std::exception("bad init");
	}
	if(isContiguous())
	{
		std::copy(il.begin(),il.end(),data());
	}
	else
	{
		// is this faster than a double loop 
		auto dit = byrow_iterator_pair().first;
		auto sit = il.begin();
		for(; sit != il.end(); )
			*dit++ = *it++;
	}
}

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

void imwrite(IM src, const char * name)
{
	// 1 is components
	stbi_write_png(name,src.width(),src.height(),1,src.data(),src.stride());
}

IM imread(const char * name)
{

}

int main(int argc, char * argv[])
{
	auto src = imread();
	decltype(src) dst(src.width/2,src.height/2);
	gaussianConvolve(src,dst);
	imwrite(dst,"output.png");

	return 0;
}