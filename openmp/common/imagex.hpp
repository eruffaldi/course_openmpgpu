/**
 * Simple imaging library for the exam
 */
#pragma once

#include  <initializer_list>
#include  <algorithm>
#include  <iostream>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


/**
 * Simple image class with ROI and sharing
 *
 * TODO: support generalized stride
 * TODO: support channels
 */
template <class T, class UT = T>
struct Image
{
	mutable std::shared_ptr<T> data_;
	
	int width_;
	int height_;
	int offset_;
	int stride_;

	Image(int w, int h);

	Image(int w, int h, std::shared_ptr<T> data);

	Image(int w, int h, const Image & other, int off);

	Image(int w, int h, Image & other, int off);

	int width() const { return width_; }

	int height() const { return height_; }

	int stride() const { return stride_; }

	int numel() const { return width_*height_; }

	bool isContiguous() const { return stride_ == width_; }

	bool isRowContiguous() const { return true; }

	T * data() { return data_.get()+offset_; }

	const T * data() const { return data_.get()+offset_; }

	const Image region(int x, int y, int w, int h) const
	{
		return Image(w,h,*this,x+y*stride());
	}

	Image region(int x, int y, int w, int h)
	{
		return Image(w,h,*this,x+y*stride());
	}

	T & operator()(int x, int y)
	{
		return *get(x,y);
	}

	T & operator()(int d)
	{
		return *get(d);
	}

	T * get(int d)
	{
		return data()+d;
	}

	const T * get(int x, int y) const
	{
		return data()+(x+y*stride_);
	}

	T * get(int x, int y)
	{
		return data()+(x+y*stride_);
	}

	const T & operator()(int x, int y) const
	{
		return *get(x,y);
	}

	const T & operator()(int d) const
	{
		return *get(d);
	}

	//template <class R = UT>
	//T conv(const Image & other, T scaling);

	//template <class R = UT>
	//T corr(const Image & other, T scaling);

	template <class Q>
	void operator= (std::initializer_list<Q> it);

	/**
	 * Any iterator over 2D matrix. 
	   Given a matrix: width,height with offset_ wrt data_ and stride_ for Y axis
		
	   The approach works with matrices and it is based on the idea of:
	   - start from (x0,y0)
	   - increment offset by inc1 for mod1 steps
	   - increment offset by inc2, then repeat inc1
		
	   syms stride real;
	   syms x y X Y W H real;
	   addressof_full = @(x,y,stride1,stride2) sum([x,y].*[stride1,stride2]);
	   addressof = @(x,y) addressof_full(x,y,1,stride);

	   When we reach mod1 steps we need to jump from a location (xi,yi) to a location (xj,yj) by means of inc2

		row then col ROI (W,H at X,Y): (data()+X+Y*stride_,1,W,stride_-W+X,W*H)

			run loop: start (X,y) and we end up at (X+W,y) then we need to jump to (X,y+1)
			inc1 = 1
			mod1 = W
			inc2 = addressof(X,y+1)-addressof(X+W,y)

		col then row ROW (W,H at X,Y): (data()+X+Y*stride_,stride_,1,W*H)

			run loop: start (x,Y) and we end up at (x,Y+H-1) then we need to jump to (x+1,Y)
			inc1 = stride
			mod1 = H
			inc2 = addressof(x+1,Y)-addressof(x,Y+H-1)

		rev row then col ROI (W,H at X,Y): (data()+(X+W-1)+(Y+H-1)*stride_,-1,W,stride_,W*H)

			run loop: start (X+W-1,y) and we end up at (X-1,y) then we need to jump to (X+W-1,y-1)
			inc1 = -1
			mod1 = W
			inc2 = addressof(X+W-1,y-1)-addressof(X-1,y)

		rev col then row ROI (W,H at X,Y): (data()+(X+W-1)+(Y+H-1)*stride_,stride_,-1,stride_,W*H)

			run loop: start (x,Y+H-1) and we end up at (x,Y-1) then we need to jump to (x-1,Y+H-1)
			inc1 = -stride_
			mod1 = H
			inc2 = addressof(x-1,Y+H-1)-addressof(x,Y+H-1)
	
		Generalization: 
		- tensor requires addressof(indexvector)
		- multiple (inc_i,mod_i) and one final incN

	 */
	struct any_iterator
	{
		any_iterator(int tot) : data_(0),step_(tot) {}

		any_iterator(T * data, int inc1, int mod1, int inc2) : inc1_(inc1),inc2_(inc2),step_(0),data_(data),mod1_(mod1) {}

		T & operator * () { return *data_; }

		T & at(int j) 
		{
			return data_[j*inc1_ + (j/mod1_)*(inc2_-inc1_)];
		}

		any_iterator& operator ++() 
		{
			if(++step_ % mod1_ == 0)
			{
				data_ += inc2_;
			}
			else
			{
				data_ += inc1_;
			}
			return *this;
		}

		bool operator == (const any_iterator & o) const 
		{
			return step_ == o.step_;
		}

		bool operator != (const any_iterator & o) const 
		{
			return step_ != o.step_;
		}

		int step_; // current step (or end step)
		int inc1_,inc2_; // increment for first and second dimension
		int mod1_; // trigger for the first dimension
		T * data_; // pointer (can co fwd or bwd)
	};

	using IT = any_iterator;

	std::pair<IT,IT> byrow_iterator_pair();

	std::pair<IT,IT> bycol_iterator_pair();

	std::pair<IT,IT> byrow_rev_iterator_pair();

	std::pair<IT,IT> bycol_rev_iterator_pair();
};

template <class T, class UT>
auto Image<T,UT>::byrow_rev_iterator_pair() -> std::pair<IT,IT> 
{
	int n = numel();
	return { any_iterator(get(width()-1,height()-1),-1,width(),width()-stride_-1), any_iterator(n) };
}

template <class T, class UT>
auto  Image<T,UT>::bycol_rev_iterator_pair() -> std::pair<IT,IT> 
{
	int n = numel();
	return { any_iterator(get(width()-1,height()-1),-stride(),height(),stride()*(height() - 1) - 1), any_iterator(n) };
}

template <class T, class UT>
auto Image<T,UT>::byrow_iterator_pair() -> std::pair<IT,IT> 
{
	int n = numel();
	return { any_iterator(data(),1,width(),stride()-width()+1), any_iterator(n) };
}

template <class T, class UT>
auto  Image<T,UT>::bycol_iterator_pair() -> std::pair<IT,IT> 
{
	int n = numel();
	return { any_iterator(data(),stride(),height(),1 - stride()*(height() - 1)), any_iterator(n) };
}

/// assignment
template <class T, class UT>
Image<T,UT>::Image(int w, int h, std::shared_ptr<T> data): width_(w),height_(h),stride_(w),data_(data),offset_(0)
{

}

/// allocation
template <class T, class UT>
Image<T,UT>::Image(int w, int h) : width_(w),height_(h),stride_(w),data_(new T[w*h]),offset_(0) {}

/// sub image const with offset 
template <class T, class UT>
Image<T,UT>::Image(int w, int h, const Image & other, int off): width_(w),height_(h),stride_(other.stride_+(other.width_-w)),data_(other.data_),offset_(off) {}

/// sub image
template <class T, class UT>
Image<T,UT>::Image(int w, int h, Image & other, int off ): width_(w),height_(h),stride_(other.stride_+(other.width_-w)),data_(other.data_),offset_(off) {}


template <class T, class UT>
template <class Q>
void Image<T,UT>::operator= (std::initializer_list<Q> il)
{
	if(il.size() != numel())
	{
		throw std::exception(); //"bad init");
	}
	if(isContiguous())
	{
		std::copy(il.begin(),il.end(),data());
	}
	else
	{
		auto pit = byrow_iterator_pair();
		auto sit = il.begin();
		for(; sit != il.end(); ++pit.first,++sit)
			*pit.first = *sit;
	}
}


template <class T>
void imwrite(Image<T,T> src, const char * name, int c)
{
	// 1 is components
	stbi_write_png(name,src.width()/c,src.height(),c,src.data(),src.stride());
}

template <class T>
Image<T,T> imread(const char * name)
{	
	int x = 0,y = 0,comp = 0;
	uint8_t * d = (uint8_t*)stbi_load(name,&x,&y,&comp,3);
	if(!d)
		return Image<T,T>(0,0);
	else
		return  Image<T,T>(x*(sizeof(T) == 1 ? 3 : 1),y,std::shared_ptr<uint8_t>(d, [] (uint8_t*p) { stbi_image_free(p); }) );
}