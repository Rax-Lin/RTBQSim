#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err__ = (call);                              \
    if (err__ != cudaSuccess) {                              \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n",      \
                   __FILE__, __LINE__, cudaGetErrorString(err__)); \
      std::exit(EXIT_FAILURE);                               \
    }                                                         \
  } while (0)

// Historical qubit threshold kept for existing runtime guard behavior.
#define MAX_LEV 40
