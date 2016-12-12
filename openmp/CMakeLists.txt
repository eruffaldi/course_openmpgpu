
#cmake -DCMAKE_CXX_COMPILER=/opt/local/bin/g++-mp-4.9 -DCMAKE_C_COMPILER=/opt/local/bin/gcc-mp-4.9 ..

#find_package(OpenMP)
#if (OPENMP_FOUND)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#   set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#endif()

add_definitions(--std=c++11 -fopenmp)
include_directories(.)
include_directories(common)
link_libraries(gomp)
add_subdirectory(samples)
add_subdirectory(exercises_stub)
add_subdirectory(exercises_sol)
