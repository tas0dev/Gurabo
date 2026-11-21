cmake_minimum_required(VERSION 4.4)
project(Gurabo LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# CLion CMake profile options
option(GURABO_ENABLE_CPU "Enable the CPU backend" ON)
option(GURABO_ENABLE_HIP "Enable the AMD HIP backend (requires HIP SDK and rocBLAS)" OFF)
option(GURABO_ENABLE_CUDA "Enable the CUDA backend (reserved)" OFF)
option(GURABO_BUILD_TESTS "Show and build the Gurabo test targets" OFF)
option(GURABO_BUILD_BENCHMARKS "Show and build the Gurabo benchmark target" OFF)

if(GURABO_ENABLE_CUDA)
    message(FATAL_ERROR "Gurabo's CUDA backend is not implemented")
endif()
if(NOT GURABO_ENABLE_CPU AND NOT GURABO_ENABLE_HIP)
    message(FATAL_ERROR "Enable at least one Gurabo backend")
endif()

# HIP calls prebuilt ROCm libraries from C++. This also works with SDKs
# that do not support CMake's HIP language on Windows.
if(GURABO_ENABLE_HIP)
    enable_language(CXX)
    if(WIN32 AND NOT MSVC)
        message(FATAL_ERROR
            "Gurabo HIP on Windows requires an MSVC-compatible toolchain. "
            "Select the Visual Studio toolchain in the active CLion CMake profile.")
    endif()

    set(GURABO_HIP_SDK_ROOT "" CACHE PATH "Optional installation prefix for HIP and rocBLAS")
    if(GURABO_HIP_SDK_ROOT)
        list(PREPEND CMAKE_PREFIX_PATH "${GURABO_HIP_SDK_ROOT}")
    endif()
    foreach(sdk_root IN ITEMS "$ENV{HIP_PATH}" "$ENV{ROCM_PATH}")
        if(sdk_root)
            list(APPEND CMAKE_PREFIX_PATH "${sdk_root}")
        endif()
    endforeach()
    if(WIN32 AND DEFINED ENV{ProgramFiles})
        list(APPEND CMAKE_PREFIX_PATH "$ENV{ProgramFiles}/AMD/ROCm/7.2")
    endif()

    find_package(hip CONFIG QUIET)
    find_package(rocblas CONFIG QUIET)
    if(NOT hip_FOUND OR NOT rocblas_FOUND OR NOT TARGET hip::host OR NOT TARGET roc::rocblas)
        message(FATAL_ERROR
            "Gurabo could not find HIP and rocBLAS development packages. "
            "Install the HIP SDK and HIP Libraries (Runtime and Development). "
            "For a custom installation, set -DGURABO_HIP_SDK_ROOT=<SDK path> "
            "in the CLion profile's CMake options.\n"
            "Detected package directories: hip='${hip_DIR}', rocblas='${rocblas_DIR}'.")
    endif()
    message(STATUS "Gurabo HIP: HIP ${hip_VERSION}, rocBLAS ${rocblas_VERSION}")
endif()

# Library and headers exposed to consumers and the IDE.
add_library(Gurabo STATIC
    src/gurabo.c
    src/core/device.c
    src/core/error.c
    src/core/float16.c
    src/core/memory.c
    src/core/stream.c
    src/core/tensor.c
    src/core/backend.h
    src/core/core.h
)
add_library(Gurabo::Gurabo ALIAS Gurabo)
target_compile_features(Gurabo PUBLIC c_std_11)
target_include_directories(Gurabo PRIVATE src/core)
target_sources(Gurabo PUBLIC
    FILE_SET HEADERS
    BASE_DIRS include
    FILES
        include/error.h
        include/gurabo.h
        include/gurabo/device.h
        include/gurabo/error.h
        include/gurabo/gurabo.h
        include/gurabo/memory.h
        include/gurabo/ops.h
        include/gurabo/stream.h
        include/gurabo/tensor.h
)

if(NOT WIN32)
    find_package(Threads REQUIRED)
    target_link_libraries(Gurabo PRIVATE Threads::Threads)
endif()

# Backend sources and dependencies follow the active CMake profile.
if(GURABO_ENABLE_CPU)
    target_sources(Gurabo PRIVATE src/backends/cpu/cpu.c)
    target_compile_definitions(Gurabo PRIVATE GURABO_WITH_CPU=1)
endif()

if(GURABO_ENABLE_HIP)
    target_sources(Gurabo PRIVATE src/backends/hip/hip.cpp)
    target_compile_features(Gurabo PRIVATE cxx_std_17)
    target_compile_definitions(Gurabo PRIVATE GURABO_WITH_HIP=1)
    target_link_libraries(Gurabo PRIVATE hip::host roc::rocblas)
endif()

# Optional CTest and benchmark targets.
if(GURABO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(GURABO_BUILD_BENCHMARKS)
    add_executable(gurabo_benchmark benchmarks/matmul.c)
    target_link_libraries(gurabo_benchmark PRIVATE Gurabo::Gurabo)
endif()
