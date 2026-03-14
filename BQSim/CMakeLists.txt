# set required cmake version
cmake_minimum_required(VERSION 3.19)

# This avoids googletest complaining that this (IPO) policy is not set
cmake_policy(SET CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

project(
  flatdd
  LANGUAGES CXX CUDA
  DESCRIPTION "BQSim - GPU-accelerated Batch Quantum Circuit Simulation using Decision Diagram")

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
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-mavx>)
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

option(BQSIM_USE_RTSPMSPM "Enable RTSpMSpM OptiX integration" OFF)
set(OptiX_INSTALL_DIR "" CACHE PATH "Path to NVIDIA OptiX SDK")
set(BQSIM_RT_NUMERIC_PRECISION "fp32" CACHE STRING "RT numeric precision (fp32 or fp64)")
set_property(CACHE BQSIM_RT_NUMERIC_PRECISION PROPERTY STRINGS fp32 fp64)
if(NOT BQSIM_RT_NUMERIC_PRECISION STREQUAL "fp32" AND
   NOT BQSIM_RT_NUMERIC_PRECISION STREQUAL "fp64")
  message(FATAL_ERROR "BQSIM_RT_NUMERIC_PRECISION must be fp32 or fp64")
endif()
if(BQSIM_RT_NUMERIC_PRECISION STREQUAL "fp64")
  add_compile_definitions(BQSIM_RT_FP64)
endif()

include_directories(${CUDA_INCLUDE_DIRS})
message(STATUS "CUDA_INCLUDE_DIRS: " ${CUDA_INCLUDE_DIRS})

if(BQSIM_USE_RTSPMSPM)
  if(NOT OptiX_INSTALL_DIR)
    message(FATAL_ERROR "BQSIM_USE_RTSPMSPM=ON requires -DOptiX_INSTALL_DIR=/path/to/OptiX")
  endif()
  include_directories(${OptiX_INSTALL_DIR}/include)
  add_definitions(-DBQSIM_USE_RTSPMSPM)
endif()
# check_submodule_present(cxxopts)
check_submodule_present(taskflow)

add_subdirectory(src)
add_subdirectory(apps)
add_subdirectory(cuquantum_test)
# add_subdirectory(extern/MemoryAllocator.KanameShiki)
# add_subdirectory(extern/cxxopts)
# add_subdirectory(extern/taskflow)
if(NOT BUILD_DDSIM_TESTS)
  set(BUILD_MQT_CORE_TESTS
      OFF
      CACHE BOOL "")
endif()

# if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
#   check_submodule_present(cxxopts)
#   add_subdirectory(apps)
# endif()
