cmake_minimum_required(VERSION 4.4)
project(Gurabo LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

option(GURABO_ENABLE_CPU "Build the C11 CPU backend" ON)
option(GURABO_ENABLE_HIP "Build the HIP and rocBLAS backend" OFF)
option(GURABO_ENABLE_CUDA "Build the CUDA backend" OFF)
option(GURABO_BUILD_TESTS "Build Gurabo tests" OFF)
option(GURABO_BUILD_BENCHMARKS "Build Gurabo benchmarks" OFF)

if(GURABO_ENABLE_CUDA)
    message(FATAL_ERROR "Gurabo's CUDA backend is not implemented")
endif()
if(NOT GURABO_ENABLE_CPU AND NOT GURABO_ENABLE_HIP)
    message(FATAL_ERROR "Enable at least one Gurabo backend")
endif()

add_library(Gurabo STATIC
    src/gurabo.c
    src/core/device.c
    src/core/error.c
    src/core/memory.c
    src/core/stream.c
    src/core/tensor.c
)
add_library(Gurabo::Gurabo ALIAS Gurabo)
target_compile_features(Gurabo PUBLIC c_std_11)
target_include_directories(Gurabo PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(Gurabo PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/core")

if(NOT WIN32)
    find_package(Threads REQUIRED)
    target_link_libraries(Gurabo PRIVATE Threads::Threads)
endif()

if(GURABO_ENABLE_CPU)
    target_sources(Gurabo PRIVATE src/backends/cpu/cpu.c)
    target_compile_definitions(Gurabo PRIVATE GURABO_WITH_CPU=1)
endif()

if(GURABO_ENABLE_HIP)
    enable_language(CXX)
    find_package(hip CONFIG REQUIRED HINTS "$ENV{HIP_PATH}" "$ENV{ROCM_PATH}")
    find_package(rocblas CONFIG REQUIRED HINTS "$ENV{HIP_PATH}" "$ENV{ROCM_PATH}")
    target_sources(Gurabo PRIVATE src/backends/hip/hip.cpp)
    target_compile_features(Gurabo PRIVATE cxx_std_17)
    target_compile_definitions(Gurabo PRIVATE GURABO_WITH_HIP=1)
    target_link_libraries(Gurabo PRIVATE hip::host roc::rocblas)
endif()

if(GURABO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(GURABO_BUILD_BENCHMARKS)
    add_executable(gurabo_benchmark benchmarks/matmul.c)
    target_link_libraries(gurabo_benchmark PRIVATE Gurabo::Gurabo)
endif()
