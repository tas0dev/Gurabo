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
option(GURABO_BUILD_EXAMPLES "Build the Gurabo inference example" OFF)

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
    find_package(hiprtc CONFIG QUIET)
    if(NOT TARGET hip::host OR NOT TARGET roc::rocblas OR NOT TARGET hiprtc::hiprtc)
        message(FATAL_ERROR
            "Gurabo requires the HIP, rocBLAS and HIPRTC development packages. "
            "Install the HIP SDK, HIP Libraries and HIP Runtime Compiler. "
            "For a custom installation, set -DGURABO_HIP_SDK_ROOT=<SDK path> "
            "in the CLion profile's CMake options.\n"
            "Detected package directories: hip='${hip_DIR}', rocblas='${rocblas_DIR}', "
            "hiprtc='${hiprtc_DIR}'.")
    endif()
    message(STATUS "Gurabo HIP: HIP ${hip_VERSION}, rocBLAS ${rocblas_VERSION}")
endif()

# Library and headers exposed to consumers and the IDE.
add_library(Gurabo STATIC
    src/gurabo.c
    src/core/device.c
    src/core/elementwise.c
    src/core/error.c
    src/core/float16.c
    src/core/memory.c
    src/core/stream.c
    src/core/tensor.c
    src/core/backend.h
    src/core/core.h
    src/core/float16_bits.h
    src/core/pointwise.h
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
    if(NOT WIN32)
        target_link_libraries(Gurabo PRIVATE m)
    endif()
endif()

if(GURABO_ENABLE_HIP)
    # Embed the sources for HIPRTC so running a program needs no source files.
    set(hip_kernel_source src/backends/hip/kernels.hip)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${hip_kernel_source}" src/core/pointwise.h src/core/float16_bits.h)
    file(READ "${hip_kernel_source}" GURABO_HIP_SOURCE)
    file(READ src/core/pointwise.h GURABO_POINTWISE_HEADER)
    file(READ src/core/float16_bits.h GURABO_FLOAT16_HEADER)
    configure_file(src/backends/hip/kernel_sources.h.in gurabo_hip_kernels.h @ONLY)
    set_source_files_properties("${hip_kernel_source}" PROPERTIES HEADER_FILE_ONLY ON)
    target_include_directories(Gurabo PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
    target_sources(Gurabo PRIVATE src/backends/hip/hip.cpp "${hip_kernel_source}")
    target_compile_features(Gurabo PRIVATE cxx_std_17)
    target_compile_definitions(Gurabo PRIVATE GURABO_WITH_HIP=1)
    target_link_libraries(Gurabo PRIVATE hip::host roc::rocblas hiprtc::hiprtc)
endif()

# Optional inference example and verification targets.
if(GURABO_BUILD_EXAMPLES OR GURABO_BUILD_TESTS)
    add_executable(gurabo_inference examples/inference.c)
    target_link_libraries(gurabo_inference PRIVATE Gurabo::Gurabo)
    if(NOT MSVC)
        target_link_libraries(gurabo_inference PRIVATE m)
    endif()
endif()

if(GURABO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set(gurabo_cmake_install_dir "${CMAKE_INSTALL_LIBDIR}/cmake/Gurabo")
configure_package_config_file(
    cmake/GuraboConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/GuraboConfig.cmake"
    INSTALL_DESTINATION "${gurabo_cmake_install_dir}"
)
install(TARGETS Gurabo EXPORT GuraboTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    FILE_SET HEADERS DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
install(EXPORT GuraboTargets NAMESPACE Gurabo:: DESTINATION "${gurabo_cmake_install_dir}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/GuraboConfig.cmake"
    DESTINATION "${gurabo_cmake_install_dir}"
)

if(GURABO_BUILD_BENCHMARKS)
    add_executable(gurabo_benchmark benchmarks/matmul.c)
    target_link_libraries(gurabo_benchmark PRIVATE Gurabo::Gurabo)
    if(NOT MSVC)
        target_link_libraries(gurabo_benchmark PRIVATE m)
    endif()
endif()
