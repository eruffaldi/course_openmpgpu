#include <iostream>
#include <vector>


struct Timer
{

		cudaEvent_t xstart, xstop;

		Timer()
		{
		cudaEventCreate(&xstart);
		cudaEventCreate(&xstop);
		}

		void start()
		{
		cudaEventRecord(xstart, 0);

		}

		void stop()
		{
		cudaEventRecord(xstop, 0);

		}

		void sync()
		{
		cudaEventSynchronize(xstop);

		}

		float elapsed() const
		{
			float time = 0;
		cudaEventElapsedTime(&time, xstart, xstop);
			return time;
		}
};

/*
A = Ar x Ac
B = Br x Bc
E faccio che Ac==Br

matrixmul<<dim3(Ar,Bc),Ac>>(a,b,c)
Rowmajor senza pitch
Assuming Ac < 1024
*/

__global__ void mmulk(float * a, float * b, float * c, int Br, int Bc)
{

}

template<class T,int base=0>
class MatrixWrap
{
public:
	MatrixWrap(T * p, int a,int b): pp(p),r(a),c(b) {}

	T operator() (int i,int j) const { return pp[(i-base)*c+(j-base)]; }

	T &operator() (int i,int j)  { return pp[(i-base)*c+(j-base)]; }

	int r,c;
	T * pp;
};

template <class T>
void mmul(MatrixWrap<T> a,MatrixWrap<T> b,MatrixWrap<T> c)
{
	// TODO assert
	for(int i = 0; i < c.r; i++)
		for(int j = 0; j < c.c; j++)
		{
			T t = 0;
			for(int k = 0; k < a.c; k++)
				t += a(i,k)*b(k,j);
			c(i,j) = t;
		}
}

template <class T>
void minit(MatrixWrap<T,0> a,T rv, T cv, T dv)
{
	for(int i = 0; i < a.r; i++)
		for(int j = 0; j < a.c; j++)
			a(i,j) = i*rv+j*cv+dv;

}

template <class T>
std::ostream & operator << (std::ostream & ons, MatrixWrap<T> & w)
{
	ons << "(" << w.r << "," << w.c << ")\n";
	for(int i = 0; i < w.r; i++)
	{
		for(int j = 0; j < w.c; j++)
			ons << w(i,j) << " ";
		ons << std::endl;
	}
	return ons;
}

int main(int argc, char const *argv[])
{
	int Ar = 1, Ac = 512, Bc = 1;
	int & Br = Ac;
	int sizeA = sizeof(float)*Ar*Ac;
	int sizeB = sizeof(float)*Ac*Bc;
	int sizeC = sizeof(float)*Ar*Bc;
	float *dev_a,*dev_b,*dev_c;
	std::vector<float> a(sizeA/sizeof(float));
	std::vector<float> b(sizeB/sizeof(float));
	std::vector<float> c(sizeC/sizeof(float));
	std::vector<float> cr(sizeC/sizeof(float));
	MatrixWrap<float,0> mwA(&a[0],Ar,Ac);
	MatrixWrap<float,0> mwB(&b[0],Ac,Bc);
	MatrixWrap<float,0> mwC(&c[0],Ar,Bc);
	MatrixWrap<float,0> mwCr(&cr[0],Ar,Bc);

	cudaMalloc( (void**)&dev_a, sizeA );
	cudaMalloc( (void**)&dev_b, sizeB );
	cudaMalloc( (void**)&dev_c, sizeC);

	minit<float>(mwA,2,1,0);
	minit<float>(mwB,-8,-4,0);
	minit<float>(mwC,0,0,1);
	minit<float>(mwCr,0,0,2);

	mmul(mwA,mwB,mwC);

	std::cout << mwA << std::endl;
	std::cout << mwB << std::endl;
	std::cout << mwC << std::endl;
	Timer t;
	t.start();
	cudaMemcpy(dev_a,mwA.pp,sizeA,cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b,mwB.pp,sizeB,cudaMemcpyHostToDevice);
	if(Ac < 1024)
	{
		// invoke mmulk
	}
	cudaMemcpy(mwCr.pp,dev_c,sizeC,cudaMemcpyDeviceToHost);
	t.stop();
	t.sync();
	std::cout << mwCr << std::endl;
	std::cout << t.elapsed() << std::endl;
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	return 0;
}