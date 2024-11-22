cmake_minimum_required(VERSION 3.22)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_VERBOSE_MAKEFILE ON)

project("CuBigIntTests"
    VERSION 0.0.1
    DESCRIPTION "CuBigIntTests: Test suite for CuBigInt library"
    LANGUAGES CUDA CXX
)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CUDA_COMPUTE_CAPABILITY "50" CACHE STRING "CUDA Compute Capability")
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
# get the library
add_subdirectory(CuBigInt)


find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

file(GLOB GPU_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/tests/*gpu.cu" "${CMAKE_CURRENT_SOURCE_DIR}/tests/*main.cu")
file(GLOB CPU_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/tests/*cpu.cu" "${CMAKE_CURRENT_SOURCE_DIR}/tests/*main.cu")

add_executable(${PROJECT_NAME}_cpu ${CPU_TEST_SOURCES} $<TARGET_OBJECTS:CuBigInt>)
add_executable(${PROJECT_NAME}_gpu ${GPU_TEST_SOURCES} $<TARGET_OBJECTS:CuBigInt>)
set_target_properties(${PROJECT_NAME}_gpu
   PROPERTIES
       CUDA_SEPARABLE_COMPILATION ON
       CUDA_RESOLVE_DEVICE_SYMBOLS ON
       POSITION_INDEPENDENT_CODE ON
)

set_target_properties(${PROJECT_NAME}_cpu
   PROPERTIES
       CUDA_SEPARABLE_COMPILATION ON
       CUDA_RESOLVE_DEVICE_SYMBOLS ON
       POSITION_INDEPENDENT_CODE ON
)


target_link_libraries(${PROJECT_NAME}_cpu PRIVATE CuBigInt)
target_link_libraries(${PROJECT_NAME}_gpu PRIVATE CuBigInt)
#

target_link_libraries(${PROJECT_NAME}_cpu PRIVATE ${CUDART_LIBRARY})
target_link_libraries(${PROJECT_NAME}_cpu PUBLIC ${GTEST_LIBRARIES} ${CUDA_LIBRARIES})

target_link_libraries(${PROJECT_NAME}_gpu PRIVATE ${CUDART_LIBRARY})
target_link_libraries(${PROJECT_NAME}_gpu PUBLIC ${GTEST_LIBRARIES} ${CUDA_LIBRARIES})



# Add the test executable to CTest
enable_testing()
add_test(NAME ${PROJECT_NAME}_gpu COMMAND ${PROJECT_NAME}_gpu)
add_test(NAME ${PROJECT_NAME}_cpu COMMAND ${PROJECT_NAME}_cpu)


## Benchmarking
find_package(benchmark REQUIRED)
file(GLOB BENCHMARK_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/*main.cu")
message(STATUS "Benchmark sources: ${BENCHMARK_SOURCES}")

add_executable(${PROJECT_NAME}_benchmark ${BENCHMARK_SOURCES} $<TARGET_OBJECTS:CuBigInt>)

set_target_properties(${PROJECT_NAME}_benchmark
   PROPERTIES
       CUDA_SEPARABLE_COMPILATION ON
       CUDA_RESOLVE_DEVICE_SYMBOLS ON
       POSITION_INDEPENDENT_CODE ON
     )

target_link_libraries(${PROJECT_NAME}_benchmark PRIVATE CuBigInt)
target_link_libraries(${PROJECT_NAME}_benchmark PRIVATE ${CUDART_LIBRARY})
target_link_libraries(${PROJECT_NAME}_benchmark PRIVATE benchmark::benchmark)
