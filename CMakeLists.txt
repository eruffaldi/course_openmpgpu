
#cmake -DCMAKE_CXX_COMPILER=/opt/local/bin/g++-mp-4.9 -DCMAKE_C_COMPILER=/opt/local/bin/gcc-mp-4.9 ..

#find_package(OpenMP)
#if (OPENMP_FOUND)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#   set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#endif()

add_definitions(--std=c++11 -fopenmp)
link_libraries(gomp)

add_executable(parsin parsin.cpp)

add_executable(task1 task1.cpp)

add_executable(parnum parnum.cpp)

add_executable(parsingle parsingle.cpp)

add_executable(fib fib.cpp)

add_executable(mandel mandel_solved.cpp)

add_executable(fib_oracle fib_oracle.cpp)
