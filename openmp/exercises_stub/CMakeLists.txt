
add_definitions(--std=c++11 -fopenmp)
include_directories(.)
include_directories(../common)
link_libraries(gomp)

add_executable(qsort_stub qsort_stub.cpp)

add_executable(mergesort_stub mergesort_stub.cpp)

add_executable(pyrdown_stub pyrdown_stub.cpp)
