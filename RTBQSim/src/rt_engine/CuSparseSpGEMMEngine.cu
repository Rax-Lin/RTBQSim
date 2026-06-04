#include "CuSparseSpGEMMEngine.hpp"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

inline std::string cudaMemInfoSuffix() {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  const cudaError_t info_rc = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (info_rc != cudaSuccess) {
    cudaGetLastError();
    return std::string(" (cudaMemGetInfo failed: ") + cudaGetErrorString(info_rc) + ")";
  }
  return " (free=" + std::to_string(free_bytes) + " bytes, total=" +
         std::to_string(total_bytes) + " bytes)";
}

inline void checkCuda(cudaError_t rc, const char* msg) {
  if (rc != cudaSuccess) {
    std::string err = std::string(msg) + ": " + cudaGetErrorString(rc);
    if (rc == cudaErrorMemoryAllocation) {
      err += cudaMemInfoSuffix();
    }
    throw std::runtime_error(err);
  }
}

inline void checkCusparse(cusparseStatus_t rc, const char* msg) {
  if (rc != CUSPARSE_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(msg) + " (cuSPARSE error " + std::to_string(static_cast<int>(rc)) + ")");
  }
}

#define CUDA_CHECK(call) checkCuda((call), #call)
#define CUSPARSE_CHECK(call) checkCusparse((call), #call)

uint64_t envUInt64(const char* name, uint64_t fallback) {
  const char* value = std::getenv(name);
  if (!value) return fallback;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end == value) return fallback;
  return static_cast<uint64_t>(parsed);
}

bool envFlag(const char* name) {
  const char* value = std::getenv(name);
  if (!value) return false;
  return std::strcmp(value, "1") == 0 ||
         std::strcmp(value, "true") == 0 ||
         std::strcmp(value, "TRUE") == 0 ||
         std::strcmp(value, "on") == 0 ||
         std::strcmp(value, "ON") == 0;
}

bool envFlagDefaultTrue(const char* name) {
  const char* value = std::getenv(name);
  if (!value) return true;
  return envFlag(name);
}

__host__ __device__ inline bool isZeroMatrixEntry(const bqsim_rt::MatrixElem& value) {
  return value.x == 0.0 && value.y == 0.0;
}

int gateRowNNZUpperBound(const qc::GatePrimitive& gate) {
  if (gate.target_count <= 0 || gate.matrix_dim <= 0) {
    return 1;
  }
  int max_row = 0;
  for (int r = 0; r < gate.matrix_dim; ++r) {
    int nnz = 0;
    for (int c = 0; c < gate.matrix_dim; ++c) {
      if (!isZeroMatrixEntry(gate.matrix[r * gate.matrix_dim + c])) {
        ++nnz;
      }
    }
    max_row = std::max(max_row, nnz);
  }
  return std::max(1, max_row);
}

bool isDiagonalGate(const qc::GatePrimitive& gate) {
  const int dim = gate.matrix_dim;
  if (dim <= 0 || dim > 4) {
    return false;
  }
  for (int r = 0; r < dim; ++r) {
    for (int c = 0; c < dim; ++c) {
      if (r != c && !isZeroMatrixEntry(gate.matrix[r * dim + c])) {
        return false;
      }
    }
  }
  return true;
}

bool gateHasOneRayPerRow(const qc::GatePrimitive& gate) {
  if (gate.target_count != 1 || gate.matrix_dim != 2) {
    return false;
  }
  for (int r = 0; r < 2; ++r) {
    int row_nnz = 0;
    for (int c = 0; c < 2; ++c) {
      if (!isZeroMatrixEntry(gate.matrix[r * 2 + c])) {
        ++row_nnz;
      }
    }
    if (row_nnz != 1) {
      return false;
    }
  }
  return true;
}

__global__ void csr_to_ell_kernel(const int* csr_row_ptr,
                                  const int* csr_col,
                                  const bqsim_rt::Complex* csr_val,
                                  int rows,
                                  int ell_width,
                                  bqsim_rt::Complex* ell_val,
                                  int* ell_idx) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const int start = csr_row_ptr[row];
  const int end = csr_row_ptr[row + 1];
  const int nnz = end - start;
  const int count = nnz < ell_width ? nnz : ell_width;
  const int base = row * ell_width;
  for (int i = 0; i < count; ++i) {
    ell_idx[base + i] = csr_col[start + i];
    ell_val[base + i] = csr_val[start + i];
  }
}

__global__ void gate_row_nnz_kernel(const qc::GatePrimitive gate,
                                    int nDim,
                                    int* row_nnz) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= nDim) return;

  bool controls_on = true;
  if (gate.is_controlled && gate.control_count > 0) {
    for (int i = 0; i < gate.control_count; ++i) {
      const int c = gate.controls[i];
      if (((static_cast<unsigned long long>(row) >> c) & 1ULL) == 0ULL) {
        controls_on = false;
        break;
      }
    }
  }
  if (!controls_on) {
    row_nnz[row] = 1;
    return;
  }

  int local_row = 0;
  for (int t = 0; t < gate.target_count; ++t) {
    local_row |= static_cast<int>(((static_cast<unsigned long long>(row) >> gate.targets[t]) & 1ULL) << t);
  }
  int cnt = 0;
  const int dim = gate.matrix_dim;
  for (int local_col = 0; local_col < dim; ++local_col) {
    const auto m = gate.matrix[local_row * dim + local_col];
    if (m.x != 0.0 || m.y != 0.0) {
      ++cnt;
    }
  }
  row_nnz[row] = cnt;
}

__global__ void gate_finalize_row_ptr_kernel(int* row_ptr, const int* row_nnz, int nDim) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    row_ptr[nDim] = row_ptr[nDim - 1] + row_nnz[nDim - 1];
  }
}

__global__ void gate_fill_csr_kernel(const qc::GatePrimitive gate,
                                     int nDim,
                                     const int* row_ptr,
                                     int* col_ind,
                                     bqsim_rt::Complex* vals) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= nDim) return;
  int write = row_ptr[row];

  bool controls_on = true;
  if (gate.is_controlled && gate.control_count > 0) {
    for (int i = 0; i < gate.control_count; ++i) {
      const int c = gate.controls[i];
      if (((static_cast<unsigned long long>(row) >> c) & 1ULL) == 0ULL) {
        controls_on = false;
        break;
      }
    }
  }

  if (!controls_on) {
    col_ind[write] = row;
    vals[write] = bqsim_rt::make_complex(static_cast<bqsim_rt::Real>(1), static_cast<bqsim_rt::Real>(0));
    return;
  }

  int local_row = 0;
  for (int t = 0; t < gate.target_count; ++t) {
    local_row |= static_cast<int>(((static_cast<unsigned long long>(row) >> gate.targets[t]) & 1ULL) << t);
  }
  const int dim = gate.matrix_dim;
  for (int local_col = 0; local_col < dim; ++local_col) {
    const auto m = gate.matrix[local_row * dim + local_col];
    if (m.x == 0.0 && m.y == 0.0) {
      continue;
    }
    unsigned long long col = static_cast<unsigned long long>(row);
    for (int t = 0; t < gate.target_count; ++t) {
      const unsigned long long mask = 1ULL << gate.targets[t];
      col = (col & ~mask) |
            (((static_cast<unsigned long long>(local_col) >> t) & 1ULL) << gate.targets[t]);
    }
    col_ind[write] = static_cast<int>(col);
    vals[write] = bqsim_rt::make_complex(static_cast<bqsim_rt::Real>(m.x),
                                         static_cast<bqsim_rt::Real>(m.y));
    ++write;
  }
}

__global__ void apply_left_diagonal_gate_csr_kernel(const int* row_ptr,
                                                    const bqsim_rt::Complex* in_vals,
                                                    bqsim_rt::Complex* out_vals,
                                                    int nDim,
                                                    qc::GatePrimitive gate) {
  const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (row >= nDim) {
    return;
  }

  bool controls_ok = true;
  for (int c = 0; c < gate.control_count; ++c) {
    const int qb = gate.controls[c];
    if (((row >> qb) & 1) == 0) {
      controls_ok = false;
      break;
    }
  }

  bqsim_rt::Complex factor = bqsim_rt::make_complex(1.0, 0.0);
  if (controls_ok && gate.target_count == 1 && gate.matrix_dim == 2) {
    const int target = gate.targets[0];
    const int bit = (row >> target) & 1;
    const int diag_idx = bit * 2 + bit;
    const auto d = gate.matrix[diag_idx];
    factor = bqsim_rt::make_complex(d.x, d.y);
  }

  const int begin = row_ptr[row];
  const int end = row_ptr[row + 1];
  for (int idx = begin; idx < end; ++idx) {
    out_vals[idx] = bqsim_rt::cmul(factor, in_vals[idx]);
  }
}

__global__ void build_inverse_rowmap_for_nnz1_gate_kernel(const qc::GatePrimitive gate,
                                                           int nDim,
                                                           int* inv_rows,
                                                           bqsim_rt::Complex* inv_scales) {
  const int r = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (r >= nDim) {
    return;
  }

  int src = r;
  bqsim_rt::Complex a = bqsim_rt::make_complex(1.0, 0.0);
  bool controls_ok = true;
  for (int c = 0; c < gate.control_count; ++c) {
    const int qb = gate.controls[c];
    if (((r >> qb) & 1) == 0) {
      controls_ok = false;
      break;
    }
  }

  if (controls_ok && gate.target_count == 1 && gate.matrix_dim == 2) {
    const int target = gate.targets[0];
    const int out_bit = (r >> target) & 1;
    const int row_base = out_bit * 2;
    const bool first_nz = !isZeroMatrixEntry(gate.matrix[row_base]);
    const bool second_nz = !isZeroMatrixEntry(gate.matrix[row_base + 1]);
    if (first_nz) {
      src = r & ~(1 << target);
      const auto v = gate.matrix[row_base];
      a = bqsim_rt::make_complex(v.x, v.y);
    } else if (second_nz) {
      src = r | (1 << target);
      const auto v = gate.matrix[row_base + 1];
      a = bqsim_rt::make_complex(v.x, v.y);
    }
  }

  inv_rows[src] = r;
  inv_scales[src] = a;
}

__global__ void build_row_nnz_from_inverse_map_kernel(const int* in_row_ptr,
                                                      const int* inv_rows,
                                                      int nDim,
                                                      int* out_row_nnz) {
  const int src = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (src >= nDim) {
    return;
  }
  const int dst = inv_rows[src];
  if (dst < 0 || dst >= nDim) {
    return;
  }
  out_row_nnz[dst] = in_row_ptr[src + 1] - in_row_ptr[src];
}

__global__ void finalize_row_ptr_kernel(int* row_ptr, int nDim) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    row_ptr[nDim] = row_ptr[nDim - 1] + (row_ptr[nDim] - row_ptr[nDim - 1]);
  }
}

__global__ void apply_nnz1_gate_via_rowmap_csr_kernel(const int* in_row_ptr,
                                                       const int* in_col,
                                                       const bqsim_rt::Complex* in_val,
                                                       int nDim,
                                                       const int* inv_rows,
                                                       const bqsim_rt::Complex* inv_scales,
                                                       const int* out_row_ptr,
                                                       int* out_col,
                                                       bqsim_rt::Complex* out_val) {
  const int src = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (src >= nDim) {
    return;
  }
  const int dst = inv_rows[src];
  if (dst < 0 || dst >= nDim) {
    return;
  }
  const int in_begin = in_row_ptr[src];
  const int in_end = in_row_ptr[src + 1];
  const int out_begin = out_row_ptr[dst];
  const bqsim_rt::Complex scale = inv_scales[src];
  for (int i = 0; i < in_end - in_begin; ++i) {
    out_col[out_begin + i] = in_col[in_begin + i];
    out_val[out_begin + i] = bqsim_rt::cmul(scale, in_val[in_begin + i]);
  }
}

}  // namespace

struct CuSparseSpGEMMEngine::Impl {
  cusparseHandle_t handle = nullptr;
  cudaStream_t stream = nullptr;

  int* d_csr_row_ptr = nullptr;
  int* d_csr_col = nullptr;
  bqsim_rt::Complex* d_csr_val = nullptr;
  std::size_t csr_row_capacity = 0;
  std::size_t csr_nnz_capacity = 0;

  int* d_gate_row_ptr = nullptr;
  int* d_gate_col = nullptr;
  bqsim_rt::Complex* d_gate_val = nullptr;
  std::size_t gate_row_capacity = 0;
  std::size_t gate_nnz_capacity = 0;
  int* d_gate_row_nnz = nullptr;
  std::size_t gate_row_nnz_capacity = 0;
  void* d_scan_temp = nullptr;
  std::size_t scan_temp_capacity = 0;

  int* d_next_row_ptr = nullptr;
  int* d_next_col = nullptr;
  bqsim_rt::Complex* d_next_val = nullptr;
  std::size_t next_row_capacity = 0;
  std::size_t next_nnz_capacity = 0;

  int* h_row_ptr_pinned = nullptr;
  std::size_t h_row_ptr_capacity = 0;

  void* spgemm_work_estimation = nullptr;
  std::size_t spgemm_work_estimation_capacity = 0;
  void* spgemm_compute = nullptr;
  std::size_t spgemm_compute_capacity = 0;
  int* d_inv_rows = nullptr;
  bqsim_rt::Complex* d_inv_scales = nullptr;
  std::size_t inv_map_capacity = 0;

  std::size_t nnz = 0;
  std::size_t nDim = 0;
  int max_row_nnz = 0;
  std::size_t last_fused_gates = 0;
  bool precomputed_result = false;
  bool diag_value_only = false;

  std::string debug_circuit_name;
  std::size_t debug_block_start_gate = 0;

  ~Impl() {
    releaseBuffers();
    if (handle) {
      cusparseDestroy(handle);
      handle = nullptr;
    }
    if (stream) {
      cudaStreamDestroy(stream);
      stream = nullptr;
    }
  }

  void ensureRuntime() {
    if (!stream) {
      CUDA_CHECK(cudaStreamCreate(&stream));
    }
    if (!handle) {
      CUSPARSE_CHECK(cusparseCreate(&handle));
      CUSPARSE_CHECK(cusparseSetStream(handle, stream));
    }
  }

  void releaseBuffers() {
    if (d_csr_row_ptr) cudaFree(d_csr_row_ptr);
    if (d_csr_col) cudaFree(d_csr_col);
    if (d_csr_val) cudaFree(d_csr_val);
    if (d_gate_row_ptr) cudaFree(d_gate_row_ptr);
    if (d_gate_col) cudaFree(d_gate_col);
    if (d_gate_val) cudaFree(d_gate_val);
    if (d_next_row_ptr) cudaFree(d_next_row_ptr);
    if (d_next_col) cudaFree(d_next_col);
    if (d_next_val) cudaFree(d_next_val);
    if (spgemm_work_estimation) cudaFree(spgemm_work_estimation);
    if (spgemm_compute) cudaFree(spgemm_compute);
    if (h_row_ptr_pinned) cudaFreeHost(h_row_ptr_pinned);
    if (d_gate_row_nnz) cudaFree(d_gate_row_nnz);
    if (d_scan_temp) cudaFree(d_scan_temp);
    if (d_inv_rows) cudaFree(d_inv_rows);
    if (d_inv_scales) cudaFree(d_inv_scales);
    d_csr_row_ptr = nullptr;
    d_csr_col = nullptr;
    d_csr_val = nullptr;
    d_gate_row_ptr = nullptr;
    d_gate_col = nullptr;
    d_gate_val = nullptr;
    d_next_row_ptr = nullptr;
    d_next_col = nullptr;
    d_next_val = nullptr;
    spgemm_work_estimation = nullptr;
    spgemm_compute = nullptr;
    h_row_ptr_pinned = nullptr;
    d_gate_row_nnz = nullptr;
    d_scan_temp = nullptr;
    d_inv_rows = nullptr;
    d_inv_scales = nullptr;
    csr_row_capacity = 0;
    csr_nnz_capacity = 0;
    gate_row_capacity = 0;
    gate_nnz_capacity = 0;
    gate_row_nnz_capacity = 0;
    next_row_capacity = 0;
    next_nnz_capacity = 0;
    h_row_ptr_capacity = 0;
    scan_temp_capacity = 0;
    spgemm_work_estimation_capacity = 0;
    spgemm_compute_capacity = 0;
    inv_map_capacity = 0;
  }

  void resetState() {
    nnz = 0;
    nDim = 0;
    max_row_nnz = 0;
    last_fused_gates = 0;
    precomputed_result = false;
    diag_value_only = envFlag("RT_DIAG_VALUE_ONLY");
    if (!std::getenv("RT_DIAG_VALUE_ONLY")) {
      diag_value_only = false;
    }
  }

  void ensureFinalCapacity(std::size_t required_nnz, std::size_t rows) {
    if (!d_csr_row_ptr || (rows + 1) > csr_row_capacity) {
      if (d_csr_row_ptr) cudaFree(d_csr_row_ptr);
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_csr_row_ptr), (rows + 1) * sizeof(int)));
      csr_row_capacity = rows + 1;
    }
    if (required_nnz > csr_nnz_capacity) {
      if (d_csr_col) cudaFree(d_csr_col);
      if (d_csr_val) cudaFree(d_csr_val);
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_csr_col), required_nnz * sizeof(int)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_csr_val), required_nnz * sizeof(bqsim_rt::Complex)));
      csr_nnz_capacity = required_nnz;
    }
  }

  void ensureGateCapacity(std::size_t required_nnz, std::size_t rows) {
    if (!d_gate_row_ptr || (rows + 1) > gate_row_capacity) {
      if (d_gate_row_ptr) cudaFree(d_gate_row_ptr);
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gate_row_ptr), (rows + 1) * sizeof(int)));
      gate_row_capacity = rows + 1;
    }
    if (required_nnz > gate_nnz_capacity) {
      if (d_gate_col) cudaFree(d_gate_col);
      if (d_gate_val) cudaFree(d_gate_val);
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gate_col), required_nnz * sizeof(int)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gate_val), required_nnz * sizeof(bqsim_rt::Complex)));
      gate_nnz_capacity = required_nnz;
    }
  }

  void ensureNextCapacity(std::size_t required_nnz, std::size_t rows) {
    if (!d_next_row_ptr || (rows + 1) > next_row_capacity) {
      if (d_next_row_ptr) cudaFree(d_next_row_ptr);
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_next_row_ptr), (rows + 1) * sizeof(int)));
      next_row_capacity = rows + 1;
    }
    if (required_nnz > next_nnz_capacity) {
      if (d_next_col) cudaFree(d_next_col);
      if (d_next_val) cudaFree(d_next_val);
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_next_col), required_nnz * sizeof(int)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_next_val), required_nnz * sizeof(bqsim_rt::Complex)));
      next_nnz_capacity = required_nnz;
    }
  }

  void ensureSpgemmWorkEstimationCapacity(std::size_t required_bytes) {
    if (required_bytes <= spgemm_work_estimation_capacity) {
      return;
    }
    if (spgemm_work_estimation) {
      CUDA_CHECK(cudaFree(spgemm_work_estimation));
      spgemm_work_estimation = nullptr;
    }
    CUDA_CHECK(cudaMalloc(&spgemm_work_estimation, required_bytes));
    spgemm_work_estimation_capacity = required_bytes;
  }

  void ensureSpgemmComputeCapacity(std::size_t required_bytes) {
    if (required_bytes <= spgemm_compute_capacity) {
      return;
    }
    if (spgemm_compute) {
      CUDA_CHECK(cudaFree(spgemm_compute));
      spgemm_compute = nullptr;
    }
    CUDA_CHECK(cudaMalloc(&spgemm_compute, required_bytes));
    spgemm_compute_capacity = required_bytes;
  }

  void ensurePinnedRowCapacity(std::size_t rows_plus_one) {
    if (rows_plus_one <= h_row_ptr_capacity) {
      return;
    }
    if (h_row_ptr_pinned) {
      CUDA_CHECK(cudaFreeHost(h_row_ptr_pinned));
      h_row_ptr_pinned = nullptr;
    }
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_row_ptr_pinned), rows_plus_one * sizeof(int)));
    h_row_ptr_capacity = rows_plus_one;
  }

  void ensureGateRowNnzAndScanTemp(std::size_t rows) {
    if (!d_gate_row_nnz || rows > gate_row_nnz_capacity) {
      if (d_gate_row_nnz) {
        CUDA_CHECK(cudaFree(d_gate_row_nnz));
      }
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gate_row_nnz), rows * sizeof(int)));
      gate_row_nnz_capacity = rows;
    }
    std::size_t bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, bytes, d_gate_row_nnz, d_gate_row_ptr, rows, stream));
    if (bytes > scan_temp_capacity) {
      if (d_scan_temp) {
        CUDA_CHECK(cudaFree(d_scan_temp));
        d_scan_temp = nullptr;
      }
      CUDA_CHECK(cudaMalloc(&d_scan_temp, bytes));
      scan_temp_capacity = bytes;
    }
  }

  void ensureInvMapCapacity(std::size_t rows) {
    if (rows == 0) {
      return;
    }
    if (d_inv_rows && d_inv_scales && inv_map_capacity >= rows) {
      return;
    }
    if (d_inv_rows) {
      CUDA_CHECK(cudaFree(d_inv_rows));
      d_inv_rows = nullptr;
    }
    if (d_inv_scales) {
      CUDA_CHECK(cudaFree(d_inv_scales));
      d_inv_scales = nullptr;
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_inv_rows), rows * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_inv_scales), rows * sizeof(bqsim_rt::Complex)));
    inv_map_capacity = rows;
  }
};

CuSparseSpGEMMEngine::CuSparseSpGEMMEngine() : impl(std::make_unique<Impl>()) {}
CuSparseSpGEMMEngine::~CuSparseSpGEMMEngine() = default;

bool CuSparseSpGEMMEngine::isAvailable() const { return available; }
void CuSparseSpGEMMEngine::setAvailable(bool value) { available = value; }

bool CuSparseSpGEMMEngine::prepareGeometryFromGates(const qc::GatePrimitive* gates,
                                                    std::size_t gate_count,
                                                    int,
                                                    std::size_t nDim,
                                                    bool force_full) {
  last_stats = {};
  if (!available || !impl || !gates || gate_count == 0 || nDim == 0) {
    return false;
  }

  try {
    const bool collect_breakdown = envFlagDefaultTrue("BQSIM_ENABLE_BREAKDOWN");
    const auto stage_start = std::chrono::high_resolution_clock::now();
    impl->ensureRuntime();
    impl->resetState();
    impl->nDim = nDim;
    double total_csr_build_ms = 0.0;
    double total_h2d_ms = 0.0;
    double total_spgemm_ms = 0.0;
    double total_row_scan_ms = 0.0;
    double total_nnz1_ms = 0.0;
    std::size_t total_skip_count = 0;

    const uint64_t max_gates_env = envUInt64("BQSIM_RT_SPM_MAX_GATES", static_cast<uint64_t>(gate_count));
    const std::size_t max_gates = std::min<std::size_t>(gate_count, static_cast<std::size_t>(max_gates_env));
    const int row_nnz_limit = static_cast<int>(envUInt64("BQSIM_RT_SPM_ROW_NNZ_LIMIT", 4ULL));

    impl->ensurePinnedRowCapacity(nDim + 1);
    impl->ensureGateRowNnzAndScanTemp(nDim);
    const int threads = 256;
    const int blocks = static_cast<int>((nDim + threads - 1) / threads);

    auto buildGateCSRDevice = [&](const qc::GatePrimitive& gate,
                                  int* d_row_ptr,
                                  int* d_col,
                                  bqsim_rt::Complex* d_val,
                                  std::size_t& out_nnz) {
      gate_row_nnz_kernel<<<blocks, threads, 0, impl->stream>>>(gate, static_cast<int>(nDim), impl->d_gate_row_nnz);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cub::DeviceScan::ExclusiveSum(impl->d_scan_temp,
                                               impl->scan_temp_capacity,
                                               impl->d_gate_row_nnz,
                                               d_row_ptr,
                                               nDim,
                                               impl->stream));
      gate_finalize_row_ptr_kernel<<<1, 1, 0, impl->stream>>>(d_row_ptr, impl->d_gate_row_nnz, static_cast<int>(nDim));
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpyAsync(impl->h_row_ptr_pinned,
                                 d_row_ptr + nDim,
                                 sizeof(int),
                                 cudaMemcpyDeviceToHost,
                                 impl->stream));
      CUDA_CHECK(cudaStreamSynchronize(impl->stream));
      out_nnz = static_cast<std::size_t>(impl->h_row_ptr_pinned[0]);
      if (out_nnz == 0) {
        out_nnz = 1;
      }
      gate_fill_csr_kernel<<<blocks, threads, 0, impl->stream>>>(
          gate, static_cast<int>(nDim), d_row_ptr, d_col, d_val);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaStreamSynchronize(impl->stream));
    };

    auto t0 = std::chrono::high_resolution_clock::now();
    const std::size_t first_cap = static_cast<std::size_t>(nDim) * 4;
    impl->ensureFinalCapacity(first_cap, nDim);
    buildGateCSRDevice(gates[0], impl->d_csr_row_ptr, impl->d_csr_col, impl->d_csr_val, impl->nnz);
    auto t1 = std::chrono::high_resolution_clock::now();
    total_csr_build_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

    int running_max_row_nnz = 1;
    CUDA_CHECK(cudaMemcpy(impl->h_row_ptr_pinned, impl->d_csr_row_ptr, (nDim + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    for (std::size_t r = 0; r < nDim; ++r) {
      running_max_row_nnz = std::max(running_max_row_nnz, impl->h_row_ptr_pinned[r + 1] - impl->h_row_ptr_pinned[r]);
    }

    std::size_t fused = 1;
    ++total_skip_count;
    for (std::size_t g = 1; g < max_gates; ++g) {
      const bool is_diag = impl->diag_value_only && isDiagonalGate(gates[g]);
      const bool is_nnz1_gate = gateHasOneRayPerRow(gates[g]);
      const int gate_row_nnz_ub = gateRowNNZUpperBound(gates[g]);
      if (!force_full && row_nnz_limit > 0 && !is_diag && !is_nnz1_gate) {
        const long long predicted = static_cast<long long>(running_max_row_nnz) * gate_row_nnz_ub;
        if (predicted > static_cast<long long>(row_nnz_limit)) {
          break;
        }
      }

      if (is_diag) {
        ++total_skip_count;
        impl->ensureNextCapacity(impl->nnz, nDim);
        CUDA_CHECK(cudaMemcpyAsync(impl->d_next_row_ptr,
                                   impl->d_csr_row_ptr,
                                   (nDim + 1) * sizeof(int),
                                   cudaMemcpyDeviceToDevice,
                                   impl->stream));
        CUDA_CHECK(cudaMemcpyAsync(impl->d_next_col,
                                   impl->d_csr_col,
                                   impl->nnz * sizeof(int),
                                   cudaMemcpyDeviceToDevice,
                                   impl->stream));
        const auto td0 = std::chrono::high_resolution_clock::now();
        apply_left_diagonal_gate_csr_kernel<<<blocks, threads, 0, impl->stream>>>(
            impl->d_csr_row_ptr,
            impl->d_csr_val,
            impl->d_next_val,
            static_cast<int>(nDim),
            gates[g]);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(impl->stream));
        const auto td1 = std::chrono::high_resolution_clock::now();
        total_nnz1_ms += std::chrono::duration<double, std::milli>(td1 - td0).count();

        std::swap(impl->d_csr_row_ptr, impl->d_next_row_ptr);
        std::swap(impl->d_csr_col, impl->d_next_col);
        std::swap(impl->d_csr_val, impl->d_next_val);
        std::swap(impl->csr_nnz_capacity, impl->next_nnz_capacity);
        ++fused;
        continue;
      }

      if (is_nnz1_gate) {
        ++total_skip_count;
        impl->ensureInvMapCapacity(nDim);
        impl->ensureNextCapacity(impl->nnz, nDim);
        CUDA_CHECK(cudaMemsetAsync(impl->d_inv_rows, 0xFF, nDim * sizeof(int), impl->stream));
        CUDA_CHECK(cudaMemsetAsync(impl->d_gate_row_nnz, 0, nDim * sizeof(int), impl->stream));

        const auto tn0 = std::chrono::high_resolution_clock::now();
        build_inverse_rowmap_for_nnz1_gate_kernel<<<blocks, threads, 0, impl->stream>>>(
            gates[g],
            static_cast<int>(nDim),
            impl->d_inv_rows,
            impl->d_inv_scales);
        CUDA_CHECK(cudaGetLastError());
        build_row_nnz_from_inverse_map_kernel<<<blocks, threads, 0, impl->stream>>>(
            impl->d_csr_row_ptr,
            impl->d_inv_rows,
            static_cast<int>(nDim),
            impl->d_gate_row_nnz);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(impl->d_scan_temp,
                                                 impl->scan_temp_capacity,
                                                 impl->d_gate_row_nnz,
                                                 impl->d_next_row_ptr,
                                                 nDim,
                                                 impl->stream));
        gate_finalize_row_ptr_kernel<<<1, 1, 0, impl->stream>>>(
            impl->d_next_row_ptr, impl->d_gate_row_nnz, static_cast<int>(nDim));
        CUDA_CHECK(cudaGetLastError());
        apply_nnz1_gate_via_rowmap_csr_kernel<<<blocks, threads, 0, impl->stream>>>(
            impl->d_csr_row_ptr,
            impl->d_csr_col,
            impl->d_csr_val,
            static_cast<int>(nDim),
            impl->d_inv_rows,
            impl->d_inv_scales,
            impl->d_next_row_ptr,
            impl->d_next_col,
            impl->d_next_val);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(impl->stream));
        const auto tn1 = std::chrono::high_resolution_clock::now();
        total_nnz1_ms += std::chrono::duration<double, std::milli>(tn1 - tn0).count();

        std::swap(impl->d_csr_row_ptr, impl->d_next_row_ptr);
        std::swap(impl->d_csr_col, impl->d_next_col);
        std::swap(impl->d_csr_val, impl->d_next_val);
        std::swap(impl->csr_nnz_capacity, impl->next_nnz_capacity);
        ++fused;
        continue;
      }

      auto tg0 = std::chrono::high_resolution_clock::now();
      const std::size_t gate_cap = static_cast<std::size_t>(nDim) * 4;
      impl->ensureGateCapacity(gate_cap, nDim);
      std::size_t g_nnz = 0;
      buildGateCSRDevice(gates[g], impl->d_gate_row_ptr, impl->d_gate_col, impl->d_gate_val, g_nnz);
      auto tg1 = std::chrono::high_resolution_clock::now();
      total_csr_build_ms += std::chrono::duration<double, std::milli>(tg1 - tg0).count();

      cusparseSpMatDescr_t matA = nullptr;
      cusparseSpMatDescr_t matB = nullptr;
      cusparseSpMatDescr_t matC = nullptr;
      cusparseSpGEMMDescr_t spgemmDesc = nullptr;

#if defined(BQSIM_RT_FP64)
      const cudaDataType value_type = CUDA_C_64F;
      const auto alpha = make_cuDoubleComplex(1.0, 0.0);
      const auto beta = make_cuDoubleComplex(0.0, 0.0);
#else
      const cudaDataType value_type = CUDA_C_32F;
      const auto alpha = make_cuFloatComplex(1.0f, 0.0f);
      const auto beta = make_cuFloatComplex(0.0f, 0.0f);
#endif

      CUSPARSE_CHECK(cusparseCreateCsr(&matA,
                                       static_cast<int64_t>(nDim),
                                       static_cast<int64_t>(nDim),
                                       static_cast<int64_t>(g_nnz),
                                       impl->d_gate_row_ptr,
                                       impl->d_gate_col,
                                       impl->d_gate_val,
                                       CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_BASE_ZERO,
                                       value_type));
      CUSPARSE_CHECK(cusparseCreateCsr(&matB,
                                       static_cast<int64_t>(nDim),
                                       static_cast<int64_t>(nDim),
                                       static_cast<int64_t>(impl->nnz),
                                       impl->d_csr_row_ptr,
                                       impl->d_csr_col,
                                       impl->d_csr_val,
                                       CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_BASE_ZERO,
                                       value_type));
      impl->ensureNextCapacity(1, nDim);

      CUSPARSE_CHECK(cusparseCreateCsr(&matC,
                                       static_cast<int64_t>(nDim),
                                       static_cast<int64_t>(nDim),
                                       0,
                                       impl->d_next_row_ptr,
                                       nullptr,
                                       nullptr,
                                       CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_BASE_ZERO,
                                       value_type));
      CUSPARSE_CHECK(cusparseSpGEMM_createDescr(&spgemmDesc));

      std::size_t bufferSize1 = 0;
      std::size_t bufferSize2 = 0;
      auto tc0 = std::chrono::high_resolution_clock::now();
      CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(impl->handle,
                                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   &alpha,
                                                   matA,
                                                   matB,
                                                   &beta,
                                                   matC,
                                                   value_type,
                                                   CUSPARSE_SPGEMM_DEFAULT,
                                                   spgemmDesc,
                                                   &bufferSize1,
                                                   nullptr));
      impl->ensureSpgemmWorkEstimationCapacity(bufferSize1);
      CUSPARSE_CHECK(cusparseSpGEMM_workEstimation(impl->handle,
                                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   &alpha,
                                                   matA,
                                                   matB,
                                                   &beta,
                                                   matC,
                                                   value_type,
                                                   CUSPARSE_SPGEMM_DEFAULT,
                                                   spgemmDesc,
                                                   &bufferSize1,
                                                   impl->spgemm_work_estimation));

      CUSPARSE_CHECK(cusparseSpGEMM_compute(impl->handle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha,
                                            matA,
                                            matB,
                                            &beta,
                                            matC,
                                            value_type,
                                            CUSPARSE_SPGEMM_DEFAULT,
                                            spgemmDesc,
                                            &bufferSize2,
                                            nullptr));
      impl->ensureSpgemmComputeCapacity(bufferSize2);
      CUSPARSE_CHECK(cusparseSpGEMM_compute(impl->handle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha,
                                            matA,
                                            matB,
                                            &beta,
                                            matC,
                                            value_type,
                                            CUSPARSE_SPGEMM_DEFAULT,
                                            spgemmDesc,
                                            &bufferSize2,
                                            impl->spgemm_compute));

      int64_t cRows = 0;
      int64_t cCols = 0;
      int64_t cNnz = 0;
      CUSPARSE_CHECK(cusparseSpMatGetSize(matC, &cRows, &cCols, &cNnz));

      impl->ensureNextCapacity(static_cast<std::size_t>(cNnz), nDim);
      CUSPARSE_CHECK(cusparseCsrSetPointers(matC, impl->d_next_row_ptr, impl->d_next_col, impl->d_next_val));

      CUSPARSE_CHECK(cusparseSpGEMM_copy(impl->handle,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha,
                                         matA,
                                         matB,
                                         &beta,
                                         matC,
                                         value_type,
                                         CUSPARSE_SPGEMM_DEFAULT,
                                         spgemmDesc));
      CUDA_CHECK(cudaStreamSynchronize(impl->stream));
      auto tc1 = std::chrono::high_resolution_clock::now();
      total_spgemm_ms += std::chrono::duration<double, std::milli>(tc1 - tc0).count();

      const auto scan0 = std::chrono::high_resolution_clock::now();
      std::vector<int> h_c_row_ptr(nDim + 1, 0);
      CUDA_CHECK(cudaMemcpy(h_c_row_ptr.data(), impl->d_next_row_ptr, (nDim + 1) * sizeof(int), cudaMemcpyDeviceToHost));
      running_max_row_nnz = 0;
      for (std::size_t r = 0; r < nDim; ++r) {
        running_max_row_nnz = std::max(running_max_row_nnz, h_c_row_ptr[r + 1] - h_c_row_ptr[r]);
      }
      const auto scan1 = std::chrono::high_resolution_clock::now();
      total_row_scan_ms += std::chrono::duration<double, std::milli>(scan1 - scan0).count();

      std::swap(impl->d_csr_row_ptr, impl->d_next_row_ptr);
      std::swap(impl->d_csr_col, impl->d_next_col);
      std::swap(impl->d_csr_val, impl->d_next_val);
      std::swap(impl->csr_nnz_capacity, impl->next_nnz_capacity);
      impl->nnz = static_cast<std::size_t>(cNnz);
      CUSPARSE_CHECK(cusparseSpGEMM_destroyDescr(spgemmDesc));
      CUSPARSE_CHECK(cusparseDestroySpMat(matA));
      CUSPARSE_CHECK(cusparseDestroySpMat(matB));
      CUSPARSE_CHECK(cusparseDestroySpMat(matC));

      ++fused;
      if (!force_full && row_nnz_limit > 0 && running_max_row_nnz >= row_nnz_limit) {
        break;
      }
    }

    impl->last_fused_gates = fused;
    impl->max_row_nnz = std::max(1, running_max_row_nnz);
    impl->precomputed_result = true;
    impl->nDim = nDim;
    const auto stage_stop = std::chrono::high_resolution_clock::now();
    const double stage_total_ms =
        std::chrono::duration<double, std::milli>(stage_stop - stage_start).count();
    const double accounted_ms =
        total_csr_build_ms + total_h2d_ms + total_spgemm_ms + total_row_scan_ms + total_nnz1_ms;
    const double other_ms = std::max(0.0, stage_total_ms - accounted_ms);
    if (collect_breakdown) {
      last_stats.ray_gen_ms = total_csr_build_ms;
      last_stats.h2d_ms = total_h2d_ms;
      last_stats.launch_ms = total_spgemm_ms;
      last_stats.compute_ms = total_spgemm_ms;
      last_stats.compact_ms = total_row_scan_ms;
      last_stats.diagonal_ms = total_nnz1_ms;
      last_stats.overhead_ms = other_ms;
      last_stats.bvh_skip_count = total_skip_count;
    }
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[CuSparseSpGEMM] Exception: " << e.what() << std::endl;
    if (impl) {
      impl->releaseBuffers();
      impl->resetState();
    }
    return false;
  }
}

bool CuSparseSpGEMMEngine::launchRTMultiply() {
  return available && impl && impl->precomputed_result && impl->nnz > 0;
}

bool CuSparseSpGEMMEngine::collectResultToELL(bqsim_rt::Complex* values,
                                              int* indices,
                                              int num_non_zeros,
                                              std::size_t nDim) {
  if (!available || !impl || !values || !indices || !impl->d_csr_row_ptr || !impl->d_csr_col || !impl->d_csr_val) {
    return false;
  }
  if (nDim != impl->nDim || impl->nnz == 0 || num_non_zeros <= 0) {
    return false;
  }

  try {
    auto t0 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemset(values, 0, sizeof(bqsim_rt::Complex) * nDim * static_cast<std::size_t>(num_non_zeros)));
    CUDA_CHECK(cudaMemset(indices, 0, sizeof(int) * nDim * static_cast<std::size_t>(num_non_zeros)));

    const int threads = 256;
    const int blocks = static_cast<int>((nDim + threads - 1) / threads);
    csr_to_ell_kernel<<<blocks, threads, 0, impl->stream>>>(impl->d_csr_row_ptr,
                                                             impl->d_csr_col,
                                                             impl->d_csr_val,
                                                             static_cast<int>(nDim),
                                                             num_non_zeros,
                                                             values,
                                                             indices);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(impl->stream));

    auto t1 = std::chrono::high_resolution_clock::now();
    if (envFlagDefaultTrue("BQSIM_ENABLE_BREAKDOWN")) {
      last_stats.ell_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

int CuSparseSpGEMMEngine::maxRowNNZ() const {
  return impl ? impl->max_row_nnz : 0;
}

std::size_t CuSparseSpGEMMEngine::lastFusedGateCount() const {
  return impl ? impl->last_fused_gates : 0;
}

const CuSparseSpGEMMEngine::Stats& CuSparseSpGEMMEngine::lastStats() const { return last_stats; }
void CuSparseSpGEMMEngine::resetStats() { last_stats = {}; }
void CuSparseSpGEMMEngine::warmup() {
  if (available && impl) {
    impl->ensureRuntime();
  }
}
void CuSparseSpGEMMEngine::setDebugContext(const std::string& circuit_name, std::size_t block_start_gate) {
  if (!impl) return;
  impl->debug_circuit_name = circuit_name;
  impl->debug_block_start_gate = block_start_gate;
}
