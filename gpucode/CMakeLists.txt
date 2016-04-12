find_package(CUDA QUIET REQUIRED)

cuda_add_executable(helloadd helloadd.cu)
cuda_add_executable(hellodot hellodot.cu)
cuda_add_executable(mandel mandel.cu)
cuda_add_executable(norm norm.cu)