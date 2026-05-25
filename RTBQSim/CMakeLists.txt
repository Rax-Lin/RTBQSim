# set required cmake version
cmake_minimum_required(VERSION 3.19)

# This avoids googletest complaining that this (IPO) policy is not set
cmake_policy(SET CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

project(
  flatdd
  LANGUAGES CXX CUDA
  DESCRIPTION "RTBQSim - GPU-accelerated Batch Quantum Circuit Simulation")

if (NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
    SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")
elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
    SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
    SET(CUDA_NVCC_FLAGS "-g -G")
    set(CMAKE_BUILD_TYPE Debug)
    # SET(CUDA_NVCC_FLAGS "-lineinfo -Xptxas -v --gpu-architecture=sm_80")
endif()
message("Build type: " ${CMAKE_BUILD_TYPE})
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mavx>)
else()
  message(STATUS "Non-x86 host detected (${CMAKE_SYSTEM_PROCESSOR}); skip -mavx.")
endif()
# check whether `modulename` is correctly cloned in the `extern` directory.
macro(CHECK_SUBMODULE_PRESENT modulename)
  if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/${modulename}/CMakeLists.txt")
    message(
      FATAL_ERROR
        "${modulename} submodule not cloned properly. \
        Please run `git submodule update --init --recursive` \
        from the main project directory")
  endif()
endmacro()

set(CUSTATEVEC_LIBRARY $ENV{CUSTATEVEC_LIBRARY})
set(CUQUANTUM_ROOT $ENV{CUQUANTUM_ROOT})
message("-- Found custatevec library: ${CUSTATEVEC_LIBRARY}")
message("-- Found custatevec include: ${CUQUANTUM_ROOT}/include")
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

set(OptiX_INSTALL_DIR "" CACHE PATH "Path to NVIDIA OptiX SDK")
add_compile_definitions(BQSIM_RT_FP64)

include_directories(${CUDA_INCLUDE_DIRS})
message(STATUS "CUDA_INCLUDE_DIRS: " ${CUDA_INCLUDE_DIRS})

if(NOT OptiX_INSTALL_DIR)
  message(FATAL_ERROR "RTSpMSpM is hard-enabled and requires -DOptiX_INSTALL_DIR=/path/to/OptiX")
endif()
include_directories(${OptiX_INSTALL_DIR}/include)
check_submodule_present(taskflow)

add_subdirectory(src)
add_subdirectory(apps)
add_subdirectory(cuquantum_test)
