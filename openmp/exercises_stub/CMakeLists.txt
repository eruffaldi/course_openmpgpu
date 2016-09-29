
add_definitions(--std=c++11 -fopenmp)
include_directories(.)
link_libraries(gomp)

add_executable(qsort_stub qsort_stub.cpp)
