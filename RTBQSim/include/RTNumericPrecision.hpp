#pragma once

#include <cuComplex.h>
#include <cuda_runtime.h>

namespace bqsim_rt {

using Real = double;
using Complex = cuDoubleComplex;
using MatrixElem = double2;

__host__ __device__ inline Complex make_complex(Real re, Real im) {
  return make_cuDoubleComplex(re, im);
}

__host__ __device__ inline Complex cadd(const Complex& a, const Complex& b) {
  return cuCadd(a, b);
}

__host__ __device__ inline Complex cmul(const Complex& a, const Complex& b) {
  return cuCmul(a, b);
}

__host__ __device__ inline MatrixElem make_matrix_elem(Real re, Real im) {
  return make_double2(re, im);
}

}  // namespace bqsim_rt
