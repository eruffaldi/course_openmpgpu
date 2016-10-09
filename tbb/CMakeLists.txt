
#cmake -DCMAKE_CXX_COMPILER=/opt/local/bin/g++-mp-4.9 -DCMAKE_C_COMPILER=/opt/local/bin/gcc-mp-4.9 ..

#find_package(OpenMP)
#if (OPENMP_FOUND)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#   set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#endif()
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR})
find_package(TBB REQUIRED)
include_directories(${TBB_INCLUDE_DIR})
link_libraries(${TBB_tbb_LIBRARY_RELEASE})
add_definitions(--std=c++11)
if(APPLE)
add_definitions(-stdlib=libc++)
endif()
include_directories(.)
add_subdirectory(examples)
