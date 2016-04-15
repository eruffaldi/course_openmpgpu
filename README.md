# course_openmpgpu
Learning material for the module about OpenMP and GPU in the Component-Based System Design

##Building Notes
Under OSX OpenMP requires to specify GCC (installed via brew/MacPort) to CMake BUT this configuration cannot be used for CUDA.

e.g.
cmake -DCMAKE_C_COMPILER=/opt/local/bin/gcc-mp-4.9 -DCMAKE_CXX_COMPILER=/opt/local/bin/g++-mp-4.9  ../openmp

For CUDA verify requirements on NVidia CUDA Toolkit. Tested with CUDA 7.5

Remember to use CMAKE_BUILD_TYPE=Release to make optimized builds
