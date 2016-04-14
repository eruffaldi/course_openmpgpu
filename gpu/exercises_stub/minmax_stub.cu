#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/random.h>


// compute minimum and maximum values in a single reduction

// minmax_pair stores the minimum and maximum 
// values that have been encountered so far
XXX
// minmax_unary_op is a functor that takes in a value x and
// returns a minmax_pair whose minimum and maximum values
// are initialized to x.
template <typename T>
struct minmax_unary_op
  : public thrust::unary_function< T, minmax_pair<T> >
{
  __host__ __device__
  minmax_pair<T> operator()(const T& x) const
  {
    return result;
  }
};

template <typename T>
struct XXXX
  : public thrust::binary_function< YYY<T>, YYY<T>, YYY<T> >
{
  __host__ __device__
  XX operator()(const XXX& x, const XXX& y) const
  {
    return result;
  }
};


int main(void)
{
  // input size
  size_t N = 10;

  // initialize random number generator
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist(10, 99);

  // initialize data on host
  thrust::device_vector<int> data(N);
  for (size_t i = 0; i < data.size(); i++)
      data[i] = dist(rng);

  // setup arguments
 XXX unary_op;
YYY binary_op;

  // initialize reduction with the first value
 ZZZ init = unary_op(data[0]);

  // compute minimum and maximum values
  ZZZ result = thrust::transform_reduce(data.begin(), data.end(), unary_op, init, binary_op);

  // print results
  std::cout << "[ ";
  for(size_t i = 0; i < N; i++)
    std::cout << data[i] << " ";
  std::cout << "]" << std::endl;
 
  std::cout << "minimum = " << result << std::endl;
  std::cout << "maximum = " << result << std::endl;

  return 0;
}