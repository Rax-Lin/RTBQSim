#include "RTSpMSpMEngine.hpp"

#if !defined(BQSIM_USE_RTSPMSPM)

RTSpMSpMEngine::RTSpMSpMEngine() = default;
RTSpMSpMEngine::~RTSpMSpMEngine() = default;

bool RTSpMSpMEngine::isAvailable() const {
  return available;
}

void RTSpMSpMEngine::setAvailable(bool value) {
  available = value;
}

bool RTSpMSpMEngine::prepareGeometryFromGates(const qc::GatePrimitive*,
                                              std::size_t,
                                              int,
                                              std::size_t,
                                              bool) {
  last_stats = {};
  return false;
}

std::size_t RTSpMSpMEngine::lastFusedGateCount() const {
  return 0;
}

bool RTSpMSpMEngine::launchRTMultiply() {
  return false;
}

bool RTSpMSpMEngine::collectResultToELL(bqsim_rt::Complex*,
                                        int*,
                                        int,
                                        std::size_t) {
  return false;
}

int RTSpMSpMEngine::maxRowNNZ() const {
  return 0;
}

const RTSpMSpMEngine::Stats& RTSpMSpMEngine::lastStats() const {
  return last_stats;
}

void RTSpMSpMEngine::resetStats() {
  last_stats = {};
}

void RTSpMSpMEngine::warmup() {
  // No-op when RTSpMSpM is disabled.
}

void RTSpMSpMEngine::setDebugContext(const std::string&, std::size_t) {
  // No-op when RTSpMSpM is disabled.
}
#else

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

#include <chrono>
#include <cub/cub.cuh>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <thrust/system/cuda/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include <cstring>
#include <nvtx3/nvToolsExt.h>

#include "optixSpMSpM.h"
#include "GatePrimitive.hpp"

using RayDataRec = SbtRecord<RayData>;
using MissSbtRecord = SbtRecord<MissData>;
using SphereDataRec = SbtRecord<SphereData>;

namespace {

constexpr size_t kOptixLogSize = 2048;

inline void checkCuda(cudaError_t rc, const char* msg) {
  if (rc != cudaSuccess) {
    std::ostringstream oss;
    oss << msg << ": " << cudaGetErrorString(rc);
    throw std::runtime_error(oss.str());
  }
}

template <typename T>
inline void safeCudaFree(T*& ptr, const char* label) {
  if (!ptr) {
    return;
  }
  cudaError_t rc = cudaFree(reinterpret_cast<void*>(ptr));
  if (rc == cudaErrorInvalidValue || rc == cudaErrorInvalidDevicePointer) {
    cudaGetLastError();
    ptr = nullptr;
    return;
  }
  checkCuda(rc, label);
  ptr = nullptr;
}

inline void checkOptix(OptixResult rc, const char* msg) {
  if (rc != OPTIX_SUCCESS) {
    std::ostringstream oss;
    oss << msg << " (OptiX error " << static_cast<int>(rc) << ")";
    throw std::runtime_error(oss.str());
  }
}

#define CUDA_CHECK(call) checkCuda((call), #call)
#define OPTIX_CHECK(call) checkOptix((call), #call)

inline void checkOptixLog(OptixResult rc, const char* msg, const char* log, size_t log_size) {
  if (rc != OPTIX_SUCCESS) {
    std::ostringstream oss;
    oss << msg << " (OptiX error " << static_cast<int>(rc) << ")";
    if (log_size > 1 && log) {
      oss << ": " << log;
    }
    throw std::runtime_error(oss.str());
  }
}

static void context_log_cb(unsigned int level, const char* tag, const char* message, void*) {
  std::cerr << "[" << level << "][" << tag << "] " << message << "\n";
}

bool envFlag(const char* name) {
  const char* value = std::getenv(name);
  if (!value) {
    return false;
  }
  if (std::strcmp(value, "1") == 0) {
    return true;
  }
  if (std::strcmp(value, "true") == 0 || std::strcmp(value, "TRUE") == 0 ||
      std::strcmp(value, "on") == 0 || std::strcmp(value, "ON") == 0) {
    return true;
  }
  return false;
}

uint64_t envUInt64(const char* name, uint64_t fallback) {
  const char* value = std::getenv(name);
  if (!value) {
    return fallback;
  }
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end == value) {
    return fallback;
  }
  return static_cast<uint64_t>(parsed);
}

std::string sanitizeFileToken(std::string value) {
  for (char& ch : value) {
    const bool ok = (ch >= 'a' && ch <= 'z') ||
                    (ch >= 'A' && ch <= 'Z') ||
                    (ch >= '0' && ch <= '9') ||
                    ch == '-' || ch == '_' || ch == '.';
    if (!ok) {
      ch = '_';
    }
  }
  return value;
}

std::string loadPtxFromFile(const std::string& path) {
  std::ifstream file(path.c_str(), std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open PTX file: " + path);
  }
  std::ostringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

inline bool isZeroMatrixEntry(const bqsim_rt::MatrixElem& value) {
  return value.x == 0.0 && value.y == 0.0;
}

// Detect diagonal gates so we can reuse topology and only update values.
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

// Detect permutation-like 1-to-1 mappings; these can skip expensive duplicate-merge work.
bool isCollisionFreeGate(const qc::GatePrimitive& gate) {
  if (gate.target_count != 1) {
    return false;
  }
  if (gate.matrix_dim <= 0 || gate.matrix_dim > 4) {
    return false;
  }
  const int dim = gate.matrix_dim;
  if (dim * dim > 16) {
    return false;
  }

  for (int row = 0; row < dim; ++row) {
    int row_nnz = 0;
    for (int col = 0; col < dim; ++col) {
      if (!isZeroMatrixEntry(gate.matrix[row * dim + col])) {
        ++row_nnz;
      }
    }
    if (row_nnz > 1) {
      return false;
    }
  }

  for (int col = 0; col < dim; ++col) {
    int col_nnz = 0;
    for (int row = 0; row < dim; ++row) {
      if (!isZeroMatrixEntry(gate.matrix[row * dim + col])) {
        ++col_nnz;
      }
    }
    if (col_nnz > 1) {
      return false;
    }
  }
  return true;
}

// Sample one row's nonzeros for adaptive fusion stop / culling decisions.
int sampleGateRowNNZ(const qc::GatePrimitive& gate, int row) {
  if (gate.target_count != 1 || gate.matrix_dim != 2) {
    return 2;
  }
  bool controls_ok = true;
  for (int c = 0; c < gate.control_count; ++c) {
    const int qb = gate.controls[c];
    if (((row >> qb) & 1) == 0) {
      controls_ok = false;
      break;
    }
  }
  if (!controls_ok) {
    return 1;
  }

  const int target = gate.targets[0];
  const int bit = (row >> target) & 1;
  const int m = bit * 2;
  int row_nnz = 0;
  if (!isZeroMatrixEntry(gate.matrix[m])) {
    ++row_nnz;
  }
  if (!isZeroMatrixEntry(gate.matrix[m + 1])) {
    ++row_nnz;
  }
  return row_nnz;
}

// Upper bound on per-row nnz growth contributed by one primitive gate.
int gateRowNNZUpperBound(const qc::GatePrimitive& gate) {
  if (gate.target_count != 1 || gate.matrix_dim <= 0 || gate.matrix_dim > 4) {
    return 2;
  }
  const int dim = gate.matrix_dim;
  int max_row_nnz = 0;
  for (int r = 0; r < dim; ++r) {
    int row_nnz = 0;
    for (int c = 0; c < dim; ++c) {
      if (!isZeroMatrixEntry(gate.matrix[r * dim + c])) {
        ++row_nnz;
      }
    }
    if (row_nnz > max_row_nnz) {
      max_row_nnz = row_nnz;
    }
  }
  if (gate.control_count > 0 && max_row_nnz < 1) {
    max_row_nnz = 1;
  }
  return (max_row_nnz > 0) ? max_row_nnz : 1;
}

// Decide whether to cull zero gate entries before launching RT.
bool shouldCullGateRays(const qc::GatePrimitive& gate, int sample_row) {
  if (gate.target_count != 1 || gate.matrix_dim != 2) {
    return false;
  }
  return sampleGateRowNNZ(gate, sample_row) < 2;
}

struct ComplexAdd {
  __host__ __device__ bqsim_rt::Complex operator()(const bqsim_rt::Complex& a,
                                                const bqsim_rt::Complex& b) const {
    return bqsim_rt::cadd(a, b);
  }
};

// Map COO (row,col) points to sphere centers/radii for OptiX GAS build/update.
__global__ void coo_to_sphere_kernel(const int* rows,
                                     const int* cols,
                                     float3* out_points,
                                     float* out_radius,
                                     int nnz) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nnz) {
    return;
  }
  const int row = rows[tid];
  const int col = cols[tid];
  out_points[tid] = make_float3(static_cast<float>(col) + 0.5f,
                                static_cast<float>(row) + 0.5f,
                                0.5f);
  out_radius[tid] = 0.5f;
}

// Materialize one gate into COO rows/cols/values (2 entries per row for 1-qubit gates).
__global__ void build_gate_coo_kernel(const qc::GatePrimitive* gates,
                                      int gate_idx,
                                      int nDim,
                                      int* rows,
                                      int* cols,
                                      bqsim_rt::Complex* vals) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= nDim) {
    return;
  }
  const qc::GatePrimitive gate = gates[gate_idx];
  if (gate.target_count != 1) {
    return;
  }
  const int target = gate.targets[0];
  bool controls_ok = true;
  for (int c = 0; c < gate.control_count; ++c) {
    const int qb = gate.controls[c];
    if (((row >> qb) & 1) == 0) {
      controls_ok = false;
      break;
    }
  }

  const int bit = (row >> target) & 1;
  const int base = row & ~(1 << target);
  const int col0 = base;
  const int col1 = base | (1 << target);
  bqsim_rt::Complex v0{};
  bqsim_rt::Complex v1{};

  if (!controls_ok) {
    v0 = bqsim_rt::make_complex(1.0, 0.0);
    v1 = bqsim_rt::make_complex(0.0, 0.0);
  } else {
    const int m = bit * 2;
    const bqsim_rt::MatrixElem a0 = gate.matrix[m];
    const bqsim_rt::MatrixElem a1 = gate.matrix[m + 1];
    v0 = bqsim_rt::make_complex(a0.x, a0.y);
    v1 = bqsim_rt::make_complex(a1.x, a1.y);
  }

  const int idx = row * 2;
  rows[idx] = row;
  cols[idx] = controls_ok ? col0 : row;
  vals[idx] = v0;
  rows[idx + 1] = row;
  cols[idx + 1] = controls_ok ? col1 : row;
  vals[idx + 1] = v1;
}

__global__ void mark_nonzero_gate_entries_kernel(const bqsim_rt::Complex* vals,
                                                 int* flags,
                                                 int nnz) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nnz) {
    return;
  }
  const bqsim_rt::Complex v = vals[tid];
  flags[tid] = (v.x != 0.0 || v.y != 0.0) ? 1 : 0;
}

__global__ void make_key_kernel(const int* rows,
                                const int* cols,
                                uint64_t* keys,
                                uint64_t nDim,
                                int nnz) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nnz) {
    return;
  }
  const uint64_t row = static_cast<uint64_t>(rows[tid]);
  const uint64_t col = static_cast<uint64_t>(cols[tid]);
  keys[tid] = row * nDim + col;
}

__global__ void unpack_key_kernel(const uint64_t* keys,
                                  int* rows,
                                  int* cols,
                                  uint64_t nDim,
                                  int nnz) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nnz) {
    return;
  }
  const uint64_t key = keys[tid];
  rows[tid] = static_cast<int>(key / nDim);
  cols[tid] = static_cast<int>(key % nDim);
}

__global__ void accumulate_position_delta_kernel(const int* curr_rows,
                                                 const int* curr_cols,
                                                 const int* prev_rows,
                                                 const int* prev_cols,
                                                 std::size_t nnz,
                                                 unsigned long long* out_sum) {
  const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= nnz) {
    return;
  }
  const int cr = curr_rows[tid];
  const int cc = curr_cols[tid];
  const int pr = prev_rows[tid];
  const int pc = prev_cols[tid];
  const unsigned long long drow = (cr >= pr) ? static_cast<unsigned long long>(cr - pr)
                                              : static_cast<unsigned long long>(pr - cr);
  const unsigned long long dcol = (cc >= pc) ? static_cast<unsigned long long>(cc - pc)
                                              : static_cast<unsigned long long>(pc - cc);
  atomicAdd(out_sum, drow + dcol);
}

__global__ void init_identity_kernel(int nDim,
                                     int* rows,
                                     int* cols,
                                     bqsim_rt::Complex* vals) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nDim) {
    return;
  }
  rows[tid] = tid;
  cols[tid] = tid;
  vals[tid] = bqsim_rt::make_complex(1.0, 0.0);
}

__global__ void count_rows_kernel(const int* rows, int nnz, int* row_counts) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nnz) {
    return;
  }
  const int row = rows[tid];
  atomicAdd(&row_counts[row], 1);
}

} // namespace

// Pack merged COO into fixed-width ELL layout used by Stage-2 batched SpMV.
__global__ void coo_to_ell_kernel(const int* rows,
                                 const int* cols,
                                 const bqsim_rt::Complex* vals,
                                 int nnz,
                                 int num_mac,
                                 int nDim,
                                 bqsim_rt::Complex* ell_vals,
                                 int* ell_indices,
                                 int* row_counts) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nnz) {
    return;
  }
  int row = rows[tid];
  if (row < 0 || row >= nDim) {
    return;
  }
  int pos = atomicAdd(&row_counts[row], 1);
  if (pos < num_mac) {
    ell_vals[row * num_mac + pos] = vals[tid];
    ell_indices[row * num_mac + pos] = cols[tid];
  }
}

struct RTSpMSpMEngine::Impl {
  optixState state{};
  CUstream stream = 0;
  CUdeviceptr d_param = 0;
  uint64_t* d_merge_keys = nullptr;
  uint64_t* d_merge_keys_out = nullptr;
  bqsim_rt::Complex* d_merge_vals_sorted = nullptr;
  int* d_merge_unique_count = nullptr;
  void* d_merge_temp_storage = nullptr;
  size_t merge_key_capacity = 0;
  size_t merge_val_capacity = 0;
  size_t merge_temp_storage_bytes = 0;
  bool context_ready = false;
  bool pipeline_ready = false;
  bool gas_ready = false;
  bool merge_collision_free_hint = false;
  int max_row_nnz = 0;
  bool precomputed_result = false;
  bool gas_allow_update = true;
  bool gas_enable_compaction = false;
  bool gas_reuse_output_buffer = true;
  bool reuse_geometry_buffer = true;
  bool diag_value_only = true;
  uint64_t gas_update_interval = 16;
  uint64_t gas_updates_since_rebuild = 0;
  size_t nDim = 0;
  size_t num_rays = 0;
  size_t sphere_capacity = 0;
  size_t gas_prim_count = 0;
  size_t gas_output_capacity = 0;
  CUdeviceptr d_gas_temp_workspace = 0;
  size_t gas_temp_workspace_capacity = 0;
  bool gas_last_update = false;
  size_t last_fused_gates = 0;
  CUdeviceptr raygen_record = 0;
  CUdeviceptr miss_record = 0;
  CUdeviceptr hitgroup_record = 0;
  std::string debug_circuit_name;
  size_t debug_block_start_gate = 0;

  void resetState() {
    nDim = 0;
    num_rays = 0;
    merge_collision_free_hint = false;
    max_row_nnz = 0;
    precomputed_result = false;
    gas_allow_update = envFlag("BQSIM_RT_GAS_ALLOW_UPDATE");
    if (!std::getenv("BQSIM_RT_GAS_ALLOW_UPDATE")) {
      gas_allow_update = true;
    }
    gas_enable_compaction = envFlag("BQSIM_RT_GAS_COMPACT");
    gas_reuse_output_buffer = envFlag("BQSIM_RT_GAS_REUSE_OUTPUT_BUFFER");
    if (!std::getenv("BQSIM_RT_GAS_REUSE_OUTPUT_BUFFER")) {
      gas_reuse_output_buffer = true;
    }
    reuse_geometry_buffer = envFlag("BQSIM_RT_REUSE_GEOMETRY_BUFFER");
    if (!std::getenv("BQSIM_RT_REUSE_GEOMETRY_BUFFER")) {
      // Keep geometry-buffer behavior aligned with GAS output-buffer reuse unless explicitly overridden.
      reuse_geometry_buffer = gas_reuse_output_buffer;
    }
    diag_value_only = envFlag("BQSIM_RT_DIAG_VALUE_ONLY");
    if (!std::getenv("BQSIM_RT_DIAG_VALUE_ONLY")) {
      diag_value_only = true;
    }
    gas_update_interval = envUInt64("BQSIM_RT_GAS_UPDATE_INTERVAL", 16);
    gas_updates_since_rebuild = 0;
    last_fused_gates = 0;
    gas_prim_count = 0;
    gas_last_update = false;
  }

  void cleanupGeometry() {
    if (state.d_ray_rows) {
      safeCudaFree(state.d_ray_rows, "cudaFree(state.d_ray_rows)");
    }
    if (state.d_ray_cols) {
      safeCudaFree(state.d_ray_cols, "cudaFree(state.d_ray_cols)");
    }
    if (state.d_ray_vals) {
      safeCudaFree(state.d_ray_vals, "cudaFree(state.d_ray_vals)");
    }
    if (state.spherePoints) {
      safeCudaFree(state.spherePoints, "cudaFree(state.spherePoints)");
    }
    if (state.sphereRadius) {
      safeCudaFree(state.sphereRadius, "cudaFree(state.sphereRadius)");
    }
    if (state.sphereValues) {
      safeCudaFree(state.sphereValues, "cudaFree(state.sphereValues)");
    }
    if (state.d_result) {
      safeCudaFree(state.d_result, "cudaFree(state.d_result)");
    }
    if (state.d_ray_counts) {
      safeCudaFree(state.d_ray_counts, "cudaFree(state.d_ray_counts)");
    }
    if (state.d_row_counts) {
      safeCudaFree(state.d_row_counts, "cudaFree(state.d_row_counts)");
    }
    if (state.d_ray_offsets) {
      safeCudaFree(state.d_ray_offsets, "cudaFree(state.d_ray_offsets)");
    }
    if (state.d_ray_write_pos) {
      safeCudaFree(state.d_ray_write_pos, "cudaFree(state.d_ray_write_pos)");
    }
    if (state.d_out_rows) {
      safeCudaFree(state.d_out_rows, "cudaFree(state.d_out_rows)");
    }
    if (state.d_out_cols) {
      safeCudaFree(state.d_out_cols, "cudaFree(state.d_out_cols)");
    }
    if (state.d_out_vals) {
      safeCudaFree(state.d_out_vals, "cudaFree(state.d_out_vals)");
    }
    if (d_merge_keys) {
      safeCudaFree(d_merge_keys, "cudaFree(d_merge_keys)");
    }
    if (d_merge_keys_out) {
      safeCudaFree(d_merge_keys_out, "cudaFree(d_merge_keys_out)");
    }
    if (d_merge_vals_sorted) {
      safeCudaFree(d_merge_vals_sorted, "cudaFree(d_merge_vals_sorted)");
    }
    if (d_merge_unique_count) {
      safeCudaFree(d_merge_unique_count, "cudaFree(d_merge_unique_count)");
    }
    if (d_merge_temp_storage) {
      safeCudaFree(d_merge_temp_storage, "cudaFree(d_merge_temp_storage)");
    }
    merge_key_capacity = 0;
    merge_val_capacity = 0;
    merge_temp_storage_bytes = 0;
    state.out_capacity = 0;
    state.d_result_buf_size = 0;
    state.d_size = 0;
    state.sphere_size = 0;
    state.aabb_size = 0;
    sphere_capacity = 0;
  }

  void cleanupSbt() {
    if (raygen_record) {
      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(raygen_record)));
      raygen_record = 0;
    }
    if (miss_record) {
      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(miss_record)));
      miss_record = 0;
    }
    if (hitgroup_record) {
      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(hitgroup_record)));
      hitgroup_record = 0;
    }
  }

  void cleanupGas() {
    if (state.d_gas_output_buffer) {
      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_gas_output_buffer)));
      state.d_gas_output_buffer = 0;
    }
    if (d_gas_temp_workspace) {
      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gas_temp_workspace)));
      d_gas_temp_workspace = 0;
    }
    state.gas_handle = 0;
    gas_output_capacity = 0;
    gas_temp_workspace_capacity = 0;
    gas_prim_count = 0;
    gas_updates_since_rebuild = 0;
    gas_last_update = false;
    gas_ready = false;
  }

  void cleanupPipeline() {
    cleanupSbt();
    cleanupGas();
    if (state.pipeline) {
      OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
      state.pipeline = 0;
    }
    if (state.raygen_prog_group) {
      OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_prog_group));
      state.raygen_prog_group = 0;
    }
    if (state.miss_prog_group) {
      OPTIX_CHECK(optixProgramGroupDestroy(state.miss_prog_group));
      state.miss_prog_group = 0;
    }
    if (state.hit_prog_group) {
      OPTIX_CHECK(optixProgramGroupDestroy(state.hit_prog_group));
      state.hit_prog_group = 0;
    }
    if (state.module) {
      OPTIX_CHECK(optixModuleDestroy(state.module));
      state.module = 0;
    }
    if (state.sphere_module) {
      OPTIX_CHECK(optixModuleDestroy(state.sphere_module));
      state.sphere_module = 0;
    }
    pipeline_ready = false;
  }

  void cleanupContext() {
    cleanupPipeline();
    if (state.context) {
      OPTIX_CHECK(optixDeviceContextDestroy(state.context));
      state.context = nullptr;
    }
    context_ready = false;
  }

  ~Impl() {
    if (stream) {
      cudaStreamDestroy(stream);
      stream = 0;
    }
    if (d_param) {
      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_param)));
      d_param = 0;
    }
    cleanupGeometry();
    cleanupContext();
  }

  void ensureContext() {
    if (context_ready) {
      return;
    }
    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    CUcontext cuCtx = 0;
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &state.context));
    context_ready = true;
  }

  void ensurePipeline() {
    if (pipeline_ready) {
      return;
    }
    ensureContext();

    OptixModuleCompileOptions module_compile_options = {};
#if !defined(NDEBUG)
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif

    state.pipeline_compile_options.usesMotionBlur = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    state.pipeline_compile_options.numPayloadValues = 2;
    state.pipeline_compile_options.numAttributeValues = 1;
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    state.pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;

#ifndef BQSIM_RTSPMSPM_PTX
    throw std::runtime_error("BQSIM_RTSPMSPM_PTX not defined; PTX path required");
#endif
    std::string ptx = loadPtxFromFile(BQSIM_RTSPMSPM_PTX);

    char log[kOptixLogSize];
    size_t log_size = sizeof(log);
    OptixResult rc = optixModuleCreate(state.context,
                                       &module_compile_options,
                                       &state.pipeline_compile_options,
                                       ptx.c_str(),
                                       ptx.size(),
                                       log,
                                       &log_size,
                                       &state.module);
    checkOptixLog(rc, "optixModuleCreate", log, log_size);
    OptixBuiltinISOptions builtin_is_options = {};
    builtin_is_options.usesMotionBlur = false;
    builtin_is_options.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
    rc = optixBuiltinISModuleGet(state.context,
                                 &module_compile_options,
                                 &state.pipeline_compile_options,
                                 &builtin_is_options,
                                 &state.sphere_module);
    checkOptixLog(rc, "optixBuiltinISModuleGet(sphere)", log, log_size);

    OptixProgramGroupOptions program_group_options = {};

    OptixProgramGroupDesc raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = state.module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    log_size = sizeof(log);
    rc = optixProgramGroupCreate(state.context,
                                 &raygen_prog_group_desc,
                                 1,
                                 &program_group_options,
                                 log,
                                 &log_size,
                                 &state.raygen_prog_group);
    checkOptixLog(rc, "optixProgramGroupCreate(raygen)", log, log_size);

    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = state.module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    log_size = sizeof(log);
    rc = optixProgramGroupCreate(state.context,
                                 &miss_prog_group_desc,
                                 1,
                                 &program_group_options,
                                 log,
                                 &log_size,
                                 &state.miss_prog_group);
    checkOptixLog(rc, "optixProgramGroupCreate(miss)", log, log_size);

    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleAH = state.module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ch";
    hitgroup_prog_group_desc.hitgroup.moduleCH = nullptr;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = nullptr;
    hitgroup_prog_group_desc.hitgroup.moduleIS = state.sphere_module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = nullptr;
    log_size = sizeof(log);
    rc = optixProgramGroupCreate(state.context,
                                 &hitgroup_prog_group_desc,
                                 1,
                                 &program_group_options,
                                 log,
                                 &log_size,
                                 &state.hit_prog_group);
    checkOptixLog(rc, "optixProgramGroupCreate(hitgroup)", log, log_size);

    const uint32_t max_trace_depth = 1;
    OptixProgramGroup program_groups[] = {state.raygen_prog_group, state.miss_prog_group, state.hit_prog_group};
    state.pipeline_link_options.maxTraceDepth = max_trace_depth;
    log_size = sizeof(log);
    rc = optixPipelineCreate(state.context,
                             &state.pipeline_compile_options,
                             &state.pipeline_link_options,
                             program_groups,
                             sizeof(program_groups) / sizeof(program_groups[0]),
                             log,
                             &log_size,
                             &state.pipeline);
    checkOptixLog(rc, "optixPipelineCreate", log, log_size);

    OptixStackSizes stack_sizes = {};
    for (auto& pg : program_groups) {
      OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stack_sizes, state.pipeline));
    }

    uint32_t direct_callable_stack_size_from_traversal = 0;
    uint32_t direct_callable_stack_size_from_state = 0;
    uint32_t continuation_stack_size = 0;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes,
                                           max_trace_depth,
                                           0,
                                           0,
                                           &direct_callable_stack_size_from_traversal,
                                           &direct_callable_stack_size_from_state,
                                           &continuation_stack_size));
    OPTIX_CHECK(optixPipelineSetStackSize(state.pipeline,
                                          direct_callable_stack_size_from_traversal,
                                          direct_callable_stack_size_from_state,
                                          continuation_stack_size,
                                          1));

    pipeline_ready = true;
  }

  // Ensure/reuse sphere geometry buffers for current primitive count.
  void ensureSphereBuffers(size_t required) {
    if (required == 0) {
      return;
    }
    if (reuse_geometry_buffer && state.spherePoints && state.sphereRadius &&
        sphere_capacity >= required) {
      return;
    }
    if (state.spherePoints) {
      safeCudaFree(state.spherePoints, "cudaFree(state.spherePoints)");
    }
    if (state.sphereRadius) {
      safeCudaFree(state.sphereRadius, "cudaFree(state.sphereRadius)");
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.spherePoints), required * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.sphereRadius), required * sizeof(float)));
    sphere_capacity = required;
  }

  // Build or update GAS from current sphere geometry, with optional compaction/reuse policy.
  void buildGas(bool try_update = false) {
    if (!state.spherePoints || !state.sphereRadius || state.sphere_size == 0) {
      throw std::runtime_error("buildGas: sphere geometry is not ready");
    }

    const bool allow_update = gas_allow_update;
    const bool use_compaction = gas_enable_compaction && !allow_update;
    const bool same_prim_count = (gas_prim_count == state.sphere_size);
    bool do_update = try_update && gas_ready && allow_update && same_prim_count;
    if (do_update && gas_update_interval > 0 && gas_updates_since_rebuild >= gas_update_interval) {
      do_update = false;
    }

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    if (allow_update) {
      accel_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    }
    if (use_compaction) {
      accel_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    }
    accel_options.operation = do_update ? OPTIX_BUILD_OPERATION_UPDATE : OPTIX_BUILD_OPERATION_BUILD;

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
    state.devicePoints = reinterpret_cast<CUdeviceptr>(state.spherePoints);
    state.deviceRadius = reinterpret_cast<CUdeviceptr>(state.sphereRadius);
    build_input.sphereArray.numVertices = static_cast<unsigned int>(state.sphere_size);
    build_input.sphereArray.vertexBuffers = &state.devicePoints;
    build_input.sphereArray.vertexStrideInBytes = sizeof(float3);
    build_input.sphereArray.radiusBuffers = &state.deviceRadius;
    build_input.sphereArray.radiusStrideInBytes = sizeof(float);
    uint32_t build_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    build_input.sphereArray.flags = build_input_flags;
    build_input.sphereArray.numSbtRecords = 1;

    OptixAccelBufferSizes gas_buffer_sizes{};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(state.context, &accel_options, &build_input, 1, &gas_buffer_sizes));
    if (do_update && gas_output_capacity < gas_buffer_sizes.outputSizeInBytes) {
      do_update = false;
      accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
      OPTIX_CHECK(optixAccelComputeMemoryUsage(state.context, &accel_options, &build_input, 1, &gas_buffer_sizes));
    }
    if (do_update && !state.d_gas_output_buffer) {
      do_update = false;
      accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
      OPTIX_CHECK(optixAccelComputeMemoryUsage(state.context, &accel_options, &build_input, 1, &gas_buffer_sizes));
    }

    CUstream accel_stream = stream ? stream : 0;
    const size_t required_temp_size = gas_buffer_sizes.tempSizeInBytes;
    if (required_temp_size > gas_temp_workspace_capacity) {
      if (d_gas_temp_workspace) {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gas_temp_workspace)));
        d_gas_temp_workspace = 0;
      }
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_temp_workspace), required_temp_size));
      gas_temp_workspace_capacity = required_temp_size;
    }
    CUdeviceptr gas_temp_workspace = d_gas_temp_workspace;
    if (required_temp_size == 0) {
      gas_temp_workspace = 0;
    }

    bool built_with_update = false;
    if (do_update) {
      OPTIX_CHECK(optixAccelBuild(state.context,
                                  accel_stream,
                                  &accel_options,
                                  &build_input,
                                  1,
                                  gas_temp_workspace,
                                  gas_buffer_sizes.tempSizeInBytes,
                                  state.d_gas_output_buffer,
                                  gas_output_capacity,
                                  &state.gas_handle,
                                  nullptr,
                                  0));
      built_with_update = true;
    } else {
      const bool need_new_output_buffer =
          (!state.d_gas_output_buffer) ||
          (gas_output_capacity < gas_buffer_sizes.outputSizeInBytes) ||
          (!gas_reuse_output_buffer);
      if (need_new_output_buffer) {
        if (state.d_gas_output_buffer) {
          CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_gas_output_buffer)));
          state.d_gas_output_buffer = 0;
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_gas_output_buffer),
                              gas_buffer_sizes.outputSizeInBytes));
        gas_output_capacity = gas_buffer_sizes.outputSizeInBytes;
      }

      if (use_compaction) {
        CUdeviceptr d_compacted_size_buf = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_compacted_size_buf), sizeof(size_t)));
        OptixAccelEmitDesc emit_property{};
        emit_property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit_property.result = d_compacted_size_buf;
        OPTIX_CHECK(optixAccelBuild(state.context,
                                    accel_stream,
                                    &accel_options,
                                    &build_input,
                                    1,
                                    gas_temp_workspace,
                                    gas_buffer_sizes.tempSizeInBytes,
                                    state.d_gas_output_buffer,
                                    gas_output_capacity,
                                    &state.gas_handle,
                                    &emit_property,
                                    1));
        size_t compacted_size = 0;
        CUDA_CHECK(cudaMemcpy(&compacted_size,
                              reinterpret_cast<void*>(d_compacted_size_buf),
                              sizeof(size_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_compacted_size_buf)));
        if (compacted_size > 0 && compacted_size < gas_output_capacity) {
          CUdeviceptr compacted_buffer = 0;
          CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&compacted_buffer), compacted_size));
          OPTIX_CHECK(optixAccelCompact(state.context,
                                        accel_stream,
                                        state.gas_handle,
                                        compacted_buffer,
                                        compacted_size,
                                        &state.gas_handle));
          CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_gas_output_buffer)));
          state.d_gas_output_buffer = compacted_buffer;
          gas_output_capacity = compacted_size;
        }
      } else {
        OPTIX_CHECK(optixAccelBuild(state.context,
                                    accel_stream,
                                    &accel_options,
                                    &build_input,
                                    1,
                                    gas_temp_workspace,
                                    gas_buffer_sizes.tempSizeInBytes,
                                    state.d_gas_output_buffer,
                                    gas_output_capacity,
                                    &state.gas_handle,
                                    nullptr,
                                    0));
      }
    }
    gas_ready = true;
    gas_prim_count = state.sphere_size;
    gas_last_update = built_with_update;
    if (built_with_update) {
      ++gas_updates_since_rebuild;
    } else {
      gas_updates_since_rebuild = 0;
    }
  }

  // Refresh SBT records to bind current ray/sphere/result buffers before each launch.
  void buildSbt() {
    RayDataRec rg_sbt = {};
    rg_sbt.data.rows = state.d_ray_rows;
    rg_sbt.data.cols = state.d_ray_cols;
    rg_sbt.data.values = state.d_ray_vals;
    rg_sbt.data.size = state.d_size;
    if (!raygen_record) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), sizeof(RayDataRec)));
    }
    OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prog_group, &rg_sbt));
    if (stream) {
      CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(raygen_record),
                                 &rg_sbt,
                                 sizeof(RayDataRec),
                                 cudaMemcpyHostToDevice,
                                 stream));
    } else {
      CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(raygen_record),
                            &rg_sbt,
                            sizeof(RayDataRec),
                            cudaMemcpyHostToDevice));
    }

    MissSbtRecord ms_sbt = {};
    ms_sbt.data = {0.0f, 0.0f, 0.0f};
    if (!miss_record) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), sizeof(MissSbtRecord)));
    }
    OPTIX_CHECK(optixSbtRecordPackHeader(state.miss_prog_group, &ms_sbt));
    if (stream) {
      CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(miss_record),
                                 &ms_sbt,
                                 sizeof(MissSbtRecord),
                                 cudaMemcpyHostToDevice,
                                 stream));
    } else {
      CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(miss_record),
                            &ms_sbt,
                            sizeof(MissSbtRecord),
                            cudaMemcpyHostToDevice));
    }

    SphereDataRec hg_sbt = {};
    hg_sbt.data.aabbs = nullptr;
    hg_sbt.data.sphereColor = state.sphereValues;
    hg_sbt.data.rayValues = state.d_ray_vals;
    hg_sbt.data.result = state.d_result;
    hg_sbt.data.resultNumRow = state.m_result_dim.x;
    hg_sbt.data.resultNumCol = state.m_result_dim.y;
    hg_sbt.data.matrix1size = state.d_size;
    hg_sbt.data.matrix2size = state.sphere_size;
    hg_sbt.data.rayRows = state.d_ray_rows;
    hg_sbt.data.rayCols = state.d_ray_cols;
    hg_sbt.data.rayCounts = state.d_ray_counts;
    hg_sbt.data.rowCounts = state.d_row_counts;
    hg_sbt.data.rayOffsets = state.d_ray_offsets;
    hg_sbt.data.rayWritePos = state.d_ray_write_pos;
    hg_sbt.data.outRows = state.d_out_rows;
    hg_sbt.data.outCols = state.d_out_cols;
    hg_sbt.data.outVals = state.d_out_vals;
    hg_sbt.data.outCapacity = state.out_capacity;
    hg_sbt.data.mode = state.rt_mode;

    if (!hitgroup_record) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), sizeof(SphereDataRec)));
    }
    OPTIX_CHECK(optixSbtRecordPackHeader(state.hit_prog_group, &hg_sbt));
    if (stream) {
      CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(hitgroup_record),
                                 &hg_sbt,
                                 sizeof(SphereDataRec),
                                 cudaMemcpyHostToDevice,
                                 stream));
    } else {
      CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(hitgroup_record),
                            &hg_sbt,
                            sizeof(SphereDataRec),
                            cudaMemcpyHostToDevice));
    }

    state.sbt.raygenRecord = raygen_record;
    state.sbt.missRecordBase = miss_record;
    state.sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    state.sbt.missRecordCount = 1;
    state.sbt.hitgroupRecordBase = hitgroup_record;
    state.sbt.hitgroupRecordStrideInBytes = sizeof(SphereDataRec);
    state.sbt.hitgroupRecordCount = 1;

  }

  // Merge duplicated (row,col) hits on GPU: sort-by-key + reduce-by-key.
  // When gpu_ms_out is provided, return pure merge-stream elapsed time.
  bool runCudaMerge(float* gpu_ms_out = nullptr) {
    if (!state.d_ray_rows || !state.d_ray_cols || !state.sphereValues || !state.d_result) {
      return false;
    }
    const int nnz = static_cast<int>(num_rays);
    if (nnz <= 0) {
      return false;
    }

    try {
      if (!stream) {
        CUDA_CHECK(cudaStreamCreate(&stream));
      }
      cudaEvent_t merge_start = nullptr;
      cudaEvent_t merge_stop = nullptr;
      if (gpu_ms_out) {
        *gpu_ms_out = 0.0f;
        CUDA_CHECK(cudaEventCreate(&merge_start));
        CUDA_CHECK(cudaEventCreate(&merge_stop));
        CUDA_CHECK(cudaEventRecord(merge_start, stream));
      }
      if (merge_collision_free_hint) {
        if (state.d_result != state.sphereValues) {
          CUDA_CHECK(cudaMemcpyAsync(state.d_result,
                                     state.sphereValues,
                                     static_cast<size_t>(nnz) * sizeof(bqsim_rt::Complex),
                                     cudaMemcpyDeviceToDevice,
                                     stream));
        }
        if (gpu_ms_out) {
          CUDA_CHECK(cudaEventRecord(merge_stop, stream));
          CUDA_CHECK(cudaEventSynchronize(merge_stop));
          CUDA_CHECK(cudaEventElapsedTime(gpu_ms_out, merge_start, merge_stop));
          CUDA_CHECK(cudaEventDestroy(merge_start));
          CUDA_CHECK(cudaEventDestroy(merge_stop));
        }
        num_rays = static_cast<size_t>(nnz);
        state.d_size = num_rays;
        state.sphere_size = num_rays;
        return true;
      }
      const size_t required = static_cast<size_t>(nnz);
      if (merge_key_capacity < required) {
        if (d_merge_keys) {
          safeCudaFree(d_merge_keys, "cudaFree(d_merge_keys)");
        }
        if (d_merge_keys_out) {
          safeCudaFree(d_merge_keys_out, "cudaFree(d_merge_keys_out)");
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_merge_keys), required * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_merge_keys_out), required * sizeof(uint64_t)));
        merge_key_capacity = required;
      }
      if (merge_val_capacity < required) {
        if (d_merge_vals_sorted) {
          safeCudaFree(d_merge_vals_sorted, "cudaFree(d_merge_vals_sorted)");
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_merge_vals_sorted),
                              required * sizeof(bqsim_rt::Complex)));
        merge_val_capacity = required;
      }
      if (!d_merge_unique_count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_merge_unique_count), sizeof(int)));
      }
      unsigned int sort_end_bit = 64u;
      {
        const uint64_t n_dim_u64 = static_cast<uint64_t>(nDim);
        const __uint128_t key_space = static_cast<__uint128_t>(n_dim_u64) *
                                      static_cast<__uint128_t>(n_dim_u64);
        if (key_space <= 1u) {
          sort_end_bit = 1u;
        } else if (key_space > static_cast<__uint128_t>(std::numeric_limits<uint64_t>::max())) {
          sort_end_bit = 64u;
        } else {
          const uint64_t max_key = static_cast<uint64_t>(key_space - 1u);
          sort_end_bit = 64u - static_cast<unsigned int>(__builtin_clzll(max_key));
        }
      }

      const int threads = 256;
      const int blocks = static_cast<int>((nnz + threads - 1) / threads);
      make_key_kernel<<<blocks, threads, 0, stream>>>(state.d_ray_rows,
                                                       state.d_ray_cols,
                                                       d_merge_keys,
                                                       static_cast<uint64_t>(nDim),
                                                       nnz);
      CUDA_CHECK(cudaGetLastError());

      size_t sort_temp_bytes = 0;
      CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr,
                                                 sort_temp_bytes,
                                                 d_merge_keys,
                                                 d_merge_keys_out,
                                                 state.sphereValues,
                                                 d_merge_vals_sorted,
                                                 nnz,
                                                 0,
                                                 static_cast<int>(sort_end_bit),
                                                 stream));

      size_t reduce_temp_bytes = 0;
      CUDA_CHECK(cub::DeviceReduce::ReduceByKey(nullptr,
                                                reduce_temp_bytes,
                                                d_merge_keys_out,
                                                d_merge_keys,
                                                d_merge_vals_sorted,
                                                state.d_result,
                                                d_merge_unique_count,
                                                ComplexAdd(),
                                                nnz,
                                                stream));
      const size_t required_temp_bytes =
          (sort_temp_bytes > reduce_temp_bytes) ? sort_temp_bytes : reduce_temp_bytes;
      if (merge_temp_storage_bytes < required_temp_bytes) {
        if (d_merge_temp_storage) {
          safeCudaFree(d_merge_temp_storage, "cudaFree(d_merge_temp_storage)");
        }
        CUDA_CHECK(cudaMalloc(&d_merge_temp_storage, required_temp_bytes));
        merge_temp_storage_bytes = required_temp_bytes;
      }

      CUDA_CHECK(cub::DeviceRadixSort::SortPairs(d_merge_temp_storage,
                                                 sort_temp_bytes,
                                                 d_merge_keys,
                                                 d_merge_keys_out,
                                                 state.sphereValues,
                                                 d_merge_vals_sorted,
                                                 nnz,
                                                 0,
                                                 static_cast<int>(sort_end_bit),
                                                 stream));

      CUDA_CHECK(cub::DeviceReduce::ReduceByKey(d_merge_temp_storage,
                                                reduce_temp_bytes,
                                                d_merge_keys_out,
                                                d_merge_keys,
                                                d_merge_vals_sorted,
                                                state.d_result,
                                                d_merge_unique_count,
                                                ComplexAdd(),
                                                nnz,
                                                stream));

      unpack_key_kernel<<<blocks, threads, 0, stream>>>(d_merge_keys,
                                                        state.d_ray_rows,
                                                        state.d_ray_cols,
                                                        static_cast<uint64_t>(nDim),
                                                        nnz);
      CUDA_CHECK(cudaGetLastError());

      int unique = 0;
      CUDA_CHECK(cudaMemcpyAsync(&unique,
                                 d_merge_unique_count,
                                 sizeof(int),
                                 cudaMemcpyDeviceToHost,
                                 stream));
      if (gpu_ms_out) {
        CUDA_CHECK(cudaEventRecord(merge_stop, stream));
        CUDA_CHECK(cudaEventSynchronize(merge_stop));
        CUDA_CHECK(cudaEventElapsedTime(gpu_ms_out, merge_start, merge_stop));
        CUDA_CHECK(cudaEventDestroy(merge_start));
        CUDA_CHECK(cudaEventDestroy(merge_stop));
      } else {
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
      if (unique <= 0) {
        return false;
      }
      num_rays = static_cast<size_t>(unique);
      state.d_size = num_rays;
      state.sphere_size = num_rays;
      return true;
    } catch (const std::exception&) {
      return false;
    }
  }
};

RTSpMSpMEngine::RTSpMSpMEngine() : impl(std::make_unique<Impl>()) {}

RTSpMSpMEngine::~RTSpMSpMEngine() = default;

bool RTSpMSpMEngine::isAvailable() const {
  return available;
}

void RTSpMSpMEngine::setAvailable(bool value) {
  available = value;
}

// Stage 1 main pipeline:
// 1) gate->COO generation, 2) OptiX launch, 3) COO merge, 4) keep fused result on device.
bool RTSpMSpMEngine::prepareGeometryFromGates(const qc::GatePrimitive* gates,
                                              std::size_t gate_count,
                                              int num_qubits,
                                              std::size_t nDim,
                                              bool force_full) {
  last_stats = {};
  if (!available || gates == nullptr || gate_count == 0) {
    return false;
  }

  try {
    const auto setup_start = std::chrono::high_resolution_clock::now();
    impl->cleanupGeometry();
    impl->resetState();
    impl->last_fused_gates = 0;
    last_stats.build_gate_events.clear();
    last_stats.gate_traversal_events.clear();

    const uint64_t max_gates_env = envUInt64("BQSIM_RT_SPM_MAX_GATES", static_cast<uint64_t>(gate_count));
    const size_t max_gates = std::min(static_cast<size_t>(max_gates_env), gate_count);
    const bool verbose = envFlag("BQSIM_RT_SPM_VERBOSE");
    const int row_nnz_limit = static_cast<int>(envUInt64("BQSIM_RT_SPM_ROW_NNZ_LIMIT", 4ULL));
    bool enable_refit_shift_metric = envFlag("BQSIM_RT_REFIT_SHIFT_METRIC");
    if (!std::getenv("BQSIM_RT_REFIT_SHIFT_METRIC")) {
      enable_refit_shift_metric = true;
    }
    bool sync_stage_timing = envFlag("BQSIM_RT_SYNC_STAGE_TIMING");
    if (!std::getenv("BQSIM_RT_SYNC_STAGE_TIMING")) {
      sync_stage_timing = false;
    }
    const bool serial_prep_stream = envFlag("BQSIM_RT_SERIAL_PREP_STREAM");
    const int sample_row = (nDim > 1) ? 1 : 0;
    int running_max_row_nnz = 1;
    double total_geom_ms = 0.0;
    double total_gas_ms = 0.0;
    double total_launch_ms = 0.0;
    double total_ray_gen_ms = 0.0;
    double total_merge_ms = 0.0;
    double total_overhead_ms = 0.0;
    double total_refit_shift_sum = 0.0;
    std::size_t total_refit_shift_samples = 0;
    std::size_t total_bvh_rebuild_count = 0;
    std::size_t total_bvh_update_count = 0;
    std::size_t total_bvh_skip_count = 0;
    std::size_t fused_gates_applied = 0;
    const uint64_t target_global_gate =
        envUInt64("BQSIM_RT_TARGET_GLOBAL_GATE", std::numeric_limits<uint64_t>::max());
    const bool has_target_global_gate =
        target_global_gate != std::numeric_limits<uint64_t>::max();
    const bool dump_target_bvh = envFlag("BQSIM_RT_DUMP_TARGET_BVH");
    const bool enable_nvtx_profile = envFlag("BQSIM_RT_ENABLE_NVTX_PROFILE");
    impl->nDim = nDim;
    qc::GatePrimitive* d_gates = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gates), gate_count * sizeof(qc::GatePrimitive)));
    const auto h2d_start = std::chrono::high_resolution_clock::now();
    total_overhead_ms += std::chrono::duration<double, std::milli>(h2d_start - setup_start).count();
    CUDA_CHECK(cudaMemcpy(d_gates, gates, gate_count * sizeof(qc::GatePrimitive), cudaMemcpyHostToDevice));
    const auto h2d_stop = std::chrono::high_resolution_clock::now();
    last_stats.h2d_ms = std::chrono::duration<double, std::milli>(h2d_stop - h2d_start).count();
    const auto pre_loop_start = std::chrono::high_resolution_clock::now();

    const int threads = 256;
    const int blocks_n = static_cast<int>((nDim + threads - 1) / threads);
    const size_t G_nnz = static_cast<size_t>(nDim) * 2;

    int* d_M_rows = nullptr;
    int* d_M_cols = nullptr;
    bqsim_rt::Complex* d_M_vals = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_M_rows), G_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_M_cols), G_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_M_vals), G_nnz * sizeof(bqsim_rt::Complex)));
    size_t M_nnz = 0;
    int* d_G_rows[2] = {nullptr, nullptr};
    int* d_G_cols[2] = {nullptr, nullptr};
    bqsim_rt::Complex* d_G_vals[2] = {nullptr, nullptr};
    int* d_G_rows_culled[2] = {nullptr, nullptr};
    int* d_G_cols_culled[2] = {nullptr, nullptr};
    bqsim_rt::Complex* d_G_vals_culled[2] = {nullptr, nullptr};
    int* d_G_flags[2] = {nullptr, nullptr};
    int* d_G_selected_count[2] = {nullptr, nullptr};
    int* h_G_selected_count = nullptr;
    bool h_gate_use_cull[2] = {false, false};
    void* d_gate_select_temp_storage = nullptr;
    size_t gate_select_temp_storage_bytes = 0;
    for (int slot = 0; slot < 2; ++slot) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_G_rows[slot]), G_nnz * sizeof(int)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_G_cols[slot]), G_nnz * sizeof(int)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_G_vals[slot]), G_nnz * sizeof(bqsim_rt::Complex)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_G_rows_culled[slot]), G_nnz * sizeof(int)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_G_cols_culled[slot]), G_nnz * sizeof(int)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_G_vals_culled[slot]), G_nnz * sizeof(bqsim_rt::Complex)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_G_flags[slot]), G_nnz * sizeof(int)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_G_selected_count[slot]), sizeof(int)));
    }
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_G_selected_count), 2 * sizeof(int)));
    h_G_selected_count[0] = 0;
    h_G_selected_count[1] = 0;
    size_t select_temp_int_bytes = 0;
    size_t select_temp_val_bytes = 0;
    CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr,
                                          select_temp_int_bytes,
                                          d_G_rows[0],
                                          d_G_flags[0],
                                          d_G_rows_culled[0],
                                          d_G_selected_count[0],
                                          G_nnz));
    CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr,
                                          select_temp_val_bytes,
                                          d_G_vals[0],
                                          d_G_flags[0],
                                          d_G_vals_culled[0],
                                          d_G_selected_count[0],
                                          G_nnz));
    gate_select_temp_storage_bytes =
        (select_temp_int_bytes > select_temp_val_bytes) ? select_temp_int_bytes : select_temp_val_bytes;
    if (gate_select_temp_storage_bytes > 0) {
      CUDA_CHECK(cudaMalloc(&d_gate_select_temp_storage, gate_select_temp_storage_bytes));
    }
    cudaStream_t prep_stream = nullptr;
    cudaEvent_t gate_ready[2] = {nullptr, nullptr};
    CUDA_CHECK(cudaStreamCreateWithFlags(&prep_stream, cudaStreamNonBlocking));
    for (int slot = 0; slot < 2; ++slot) {
      CUDA_CHECK(cudaEventCreateWithFlags(&gate_ready[slot], cudaEventDisableTiming));
    }
    cudaEvent_t gate_gen_start[2] = {nullptr, nullptr};
    cudaEvent_t gate_gen_stop[2] = {nullptr, nullptr};
    if (sync_stage_timing) {
      for (int slot = 0; slot < 2; ++slot) {
        CUDA_CHECK(cudaEventCreate(&gate_gen_start[slot]));
        CUDA_CHECK(cudaEventCreate(&gate_gen_stop[slot]));
      }
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->state.d_ray_counts), G_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->state.d_row_counts), nDim * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->state.d_ray_offsets), G_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->state.d_ray_write_pos), G_nnz * sizeof(int)));

    std::size_t M_capacity = G_nnz;
    int* d_N_rows = nullptr;
    int* d_N_cols = nullptr;
    bqsim_rt::Complex* d_N_vals = nullptr;
    std::size_t N_capacity = 0;
    bqsim_rt::Complex* d_tmp_vals = nullptr;
    std::size_t tmp_vals_capacity = 0;
    int* d_rebuild_snapshot_rows = nullptr;
    int* d_rebuild_snapshot_cols = nullptr;
    std::size_t rebuild_snapshot_capacity = 0;
    std::size_t rebuild_snapshot_size = 0;
    bool has_rebuild_snapshot = false;
    unsigned long long* d_position_delta_sum = nullptr;

    auto ensure_next_capacity = [&](std::size_t required) {
      if (required == 0) {
        return;
      }
      if (impl->reuse_geometry_buffer && N_capacity >= required) {
        return;
      }
      if (d_N_rows) {
        safeCudaFree(d_N_rows, "cudaFree(d_N_rows)");
      }
      if (d_N_cols) {
        safeCudaFree(d_N_cols, "cudaFree(d_N_cols)");
      }
      if (d_N_vals) {
        safeCudaFree(d_N_vals, "cudaFree(d_N_vals)");
      }
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_N_rows), required * sizeof(int)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_N_cols), required * sizeof(int)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_N_vals), required * sizeof(bqsim_rt::Complex)));
      N_capacity = required;
    };

    auto ensure_tmp_vals_capacity = [&](std::size_t required) {
      if (required == 0) {
        return;
      }
      if (impl->reuse_geometry_buffer && tmp_vals_capacity >= required) {
        return;
      }
      if (d_tmp_vals) {
        safeCudaFree(d_tmp_vals, "cudaFree(d_tmp_vals)");
      }
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_tmp_vals), required * sizeof(bqsim_rt::Complex)));
      tmp_vals_capacity = required;
    };

    auto ensure_rebuild_snapshot_capacity = [&](std::size_t required) {
      if (required == 0) {
        return;
      }
      if (impl->reuse_geometry_buffer && rebuild_snapshot_capacity >= required) {
        return;
      }
      if (d_rebuild_snapshot_rows) {
        safeCudaFree(d_rebuild_snapshot_rows, "cudaFree(d_rebuild_snapshot_rows)");
      }
      if (d_rebuild_snapshot_cols) {
        safeCudaFree(d_rebuild_snapshot_cols, "cudaFree(d_rebuild_snapshot_cols)");
      }
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rebuild_snapshot_rows), required * sizeof(int)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rebuild_snapshot_cols), required * sizeof(int)));
      rebuild_snapshot_capacity = required;
      has_rebuild_snapshot = false;
      rebuild_snapshot_size = 0;
    };

    auto release_workspace = [&]() {
      if (impl->state.d_ray_counts) {
        CUDA_CHECK(cudaFree(impl->state.d_ray_counts));
        impl->state.d_ray_counts = nullptr;
      }
      if (impl->state.d_row_counts) {
        CUDA_CHECK(cudaFree(impl->state.d_row_counts));
        impl->state.d_row_counts = nullptr;
      }
      if (impl->state.d_ray_offsets) {
        CUDA_CHECK(cudaFree(impl->state.d_ray_offsets));
        impl->state.d_ray_offsets = nullptr;
      }
      if (impl->state.d_ray_write_pos) {
        CUDA_CHECK(cudaFree(impl->state.d_ray_write_pos));
        impl->state.d_ray_write_pos = nullptr;
      }
      if (d_N_rows && d_N_rows != d_M_rows) {
        safeCudaFree(d_N_rows, "cudaFree(d_N_rows)");
      }
      if (d_N_cols && d_N_cols != d_M_cols) {
        safeCudaFree(d_N_cols, "cudaFree(d_N_cols)");
      }
      if (d_N_vals && d_N_vals != d_M_vals) {
        safeCudaFree(d_N_vals, "cudaFree(d_N_vals)");
      }
      if (d_tmp_vals) {
        safeCudaFree(d_tmp_vals, "cudaFree(d_tmp_vals)");
      }
      if (d_rebuild_snapshot_rows) {
        safeCudaFree(d_rebuild_snapshot_rows, "cudaFree(d_rebuild_snapshot_rows)");
      }
      if (d_rebuild_snapshot_cols) {
        safeCudaFree(d_rebuild_snapshot_cols, "cudaFree(d_rebuild_snapshot_cols)");
      }
      if (d_position_delta_sum) {
        safeCudaFree(d_position_delta_sum, "cudaFree(d_position_delta_sum)");
      }
      impl->state.d_out_rows = nullptr;
      impl->state.d_out_cols = nullptr;
      impl->state.d_out_vals = nullptr;
      impl->state.out_capacity = 0;
    };

    auto release_prebuild_resources = [&]() {
      for (int slot = 0; slot < 2; ++slot) {
        if (gate_ready[slot]) {
          cudaEventDestroy(gate_ready[slot]);
          gate_ready[slot] = nullptr;
        }
        if (gate_gen_start[slot]) {
          cudaEventDestroy(gate_gen_start[slot]);
          gate_gen_start[slot] = nullptr;
        }
        if (gate_gen_stop[slot]) {
          cudaEventDestroy(gate_gen_stop[slot]);
          gate_gen_stop[slot] = nullptr;
        }
      }
      if (prep_stream) {
        cudaStreamDestroy(prep_stream);
        prep_stream = nullptr;
      }
      if (d_gate_select_temp_storage) {
        CUDA_CHECK(cudaFree(d_gate_select_temp_storage));
        d_gate_select_temp_storage = nullptr;
        gate_select_temp_storage_bytes = 0;
      }
      if (h_G_selected_count) {
        CUDA_CHECK(cudaFreeHost(h_G_selected_count));
        h_G_selected_count = nullptr;
      }
      for (int slot = 0; slot < 2; ++slot) {
        if (d_G_rows[slot]) {
          if (impl->state.d_ray_rows == d_G_rows[slot]) {
            impl->state.d_ray_rows = nullptr;
          }
          if (impl->state.d_out_rows == d_G_rows[slot]) {
            impl->state.d_out_rows = nullptr;
          }
          CUDA_CHECK(cudaFree(d_G_rows[slot]));
          d_G_rows[slot] = nullptr;
        }
        if (d_G_cols[slot]) {
          if (impl->state.d_ray_cols == d_G_cols[slot]) {
            impl->state.d_ray_cols = nullptr;
          }
          if (impl->state.d_out_cols == d_G_cols[slot]) {
            impl->state.d_out_cols = nullptr;
          }
          CUDA_CHECK(cudaFree(d_G_cols[slot]));
          d_G_cols[slot] = nullptr;
        }
        if (d_G_vals[slot]) {
          if (impl->state.d_ray_vals == d_G_vals[slot]) {
            impl->state.d_ray_vals = nullptr;
          }
          CUDA_CHECK(cudaFree(d_G_vals[slot]));
          d_G_vals[slot] = nullptr;
        }
        if (d_G_rows_culled[slot]) {
          if (impl->state.d_ray_rows == d_G_rows_culled[slot]) {
            impl->state.d_ray_rows = nullptr;
          }
          if (impl->state.d_out_rows == d_G_rows_culled[slot]) {
            impl->state.d_out_rows = nullptr;
          }
          CUDA_CHECK(cudaFree(d_G_rows_culled[slot]));
          d_G_rows_culled[slot] = nullptr;
        }
        if (d_G_cols_culled[slot]) {
          if (impl->state.d_ray_cols == d_G_cols_culled[slot]) {
            impl->state.d_ray_cols = nullptr;
          }
          if (impl->state.d_out_cols == d_G_cols_culled[slot]) {
            impl->state.d_out_cols = nullptr;
          }
          CUDA_CHECK(cudaFree(d_G_cols_culled[slot]));
          d_G_cols_culled[slot] = nullptr;
        }
        if (d_G_vals_culled[slot]) {
          if (impl->state.d_ray_vals == d_G_vals_culled[slot]) {
            impl->state.d_ray_vals = nullptr;
          }
          CUDA_CHECK(cudaFree(d_G_vals_culled[slot]));
          d_G_vals_culled[slot] = nullptr;
        }
        if (d_G_flags[slot]) {
          CUDA_CHECK(cudaFree(d_G_flags[slot]));
          d_G_flags[slot] = nullptr;
        }
        if (d_G_selected_count[slot]) {
          CUDA_CHECK(cudaFree(d_G_selected_count[slot]));
          d_G_selected_count[slot] = nullptr;
        }
      }
    };

    if (!impl->stream) {
      CUDA_CHECK(cudaStreamCreate(&impl->stream));
    }
    if (enable_refit_shift_metric) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_position_delta_sum), sizeof(unsigned long long)));
    }
    cudaEvent_t pending_launch_start = nullptr;
    cudaEvent_t pending_launch_stop = nullptr;
    size_t current_gate_idx = 0;
    std::size_t gate_launch_sample_count = 0;
    bool current_gate_is_target = false;
    std::string current_gate_nvtx_prefix;
    auto push_nvtx = [&](const std::string& label, bool enabled) {
      if (enabled && enable_nvtx_profile) {
        nvtxRangePushA(label.c_str());
      }
    };
    auto pop_nvtx = [&](bool enabled) {
      if (enabled && enable_nvtx_profile) {
        nvtxRangePop();
      }
    };
    auto dump_target_tree_csv = [&](std::size_t global_gate_idx, int row_nnz_before_gate) {
      if (!dump_target_bvh) {
        return;
      }
      std::vector<int> host_rows(M_nnz);
      std::vector<int> host_cols(M_nnz);
      std::vector<bqsim_rt::Complex> host_vals(M_nnz);
      CUDA_CHECK(cudaMemcpy(host_rows.data(),
                            d_M_rows,
                            M_nnz * sizeof(int),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(host_cols.data(),
                            d_M_cols,
                            M_nnz * sizeof(int),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(host_vals.data(),
                            d_M_vals,
                            M_nnz * sizeof(bqsim_rt::Complex),
                            cudaMemcpyDeviceToHost));

      const std::filesystem::path out_dir = "../../log/target_gate_bvh";
      std::filesystem::create_directories(out_dir);
      const std::string circuit_token =
          sanitizeFileToken(impl->debug_circuit_name.empty() ? "unknown_circuit" : impl->debug_circuit_name);
      const std::filesystem::path csv_path =
          out_dir / (circuit_token + "_gate" + std::to_string(global_gate_idx) + "_tree.csv");
      std::ofstream out(csv_path, std::ios::trunc);
      if (!out.is_open()) {
        std::cerr << "[SPMSPM] Failed to open target-gate BVH CSV: " << csv_path << std::endl;
        return;
      }
      out << "circuit,block_start_gate,global_gate_idx,row_nnz_before_gate,primitive_count,"
             "primitive_idx,row,col,val_real,val_imag,center_x,center_y,aabb_min_x,aabb_max_x,aabb_min_y,aabb_max_y\n";
      for (std::size_t i = 0; i < M_nnz; ++i) {
        const double center_x = static_cast<double>(host_cols[i]) + 0.5;
        const double center_y = static_cast<double>(host_rows[i]) + 0.5;
        out << circuit_token << ','
            << impl->debug_block_start_gate << ','
            << global_gate_idx << ','
            << row_nnz_before_gate << ','
            << M_nnz << ','
            << i << ','
            << host_rows[i] << ','
            << host_cols[i] << ','
            << host_vals[i].x << ','
            << host_vals[i].y << ','
            << center_x << ','
            << center_y << ','
            << (center_x - 0.5) << ','
            << (center_x + 0.5) << ','
            << (center_y - 0.5) << ','
            << (center_y + 0.5) << '\n';
      }
      std::cout << "[SPMSPM] Dumped target gate tree CSV: " << csv_path << std::endl;
    };

    auto finalize_pending_launch = [&]() {
      if (!pending_launch_start || !pending_launch_stop) {
        return;
      }
      CUDA_CHECK(cudaEventSynchronize(pending_launch_stop));
      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, pending_launch_start, pending_launch_stop));
      total_launch_ms += ms;
      CUDA_CHECK(cudaEventDestroy(pending_launch_start));
      CUDA_CHECK(cudaEventDestroy(pending_launch_stop));
      pending_launch_start = nullptr;
      pending_launch_stop = nullptr;
    };

    auto launch_optix = [&](size_t rays, bool refresh_gas, bool sync_after_launch) {
      const auto pipeline_start = std::chrono::high_resolution_clock::now();
      impl->ensurePipeline();
      const auto pipeline_stop = std::chrono::high_resolution_clock::now();
      total_overhead_ms += std::chrono::duration<double, std::milli>(pipeline_stop - pipeline_start).count();
      if (refresh_gas) {
        const std::string build_nvtx_label = current_gate_nvtx_prefix + " build";
        push_nvtx(build_nvtx_label, current_gate_is_target);
        if (sync_stage_timing) {
          cudaEvent_t gas_start = nullptr;
          cudaEvent_t gas_stop = nullptr;
          CUDA_CHECK(cudaEventCreate(&gas_start));
          CUDA_CHECK(cudaEventCreate(&gas_stop));
          CUDA_CHECK(cudaEventRecord(gas_start, impl->stream));
          impl->buildGas(true);
          CUDA_CHECK(cudaEventRecord(gas_stop, impl->stream));
          CUDA_CHECK(cudaEventSynchronize(gas_stop));
          float gas_ms = 0.0f;
          CUDA_CHECK(cudaEventElapsedTime(&gas_ms, gas_start, gas_stop));
          total_gas_ms += gas_ms;
          CUDA_CHECK(cudaEventDestroy(gas_start));
          CUDA_CHECK(cudaEventDestroy(gas_stop));
        } else {
          const auto gas_start = std::chrono::high_resolution_clock::now();
          impl->buildGas(true);
          const auto gas_stop = std::chrono::high_resolution_clock::now();
          total_gas_ms += std::chrono::duration<double, std::milli>(gas_stop - gas_start).count();
        }
        if (impl->gas_last_update) {
          ++total_bvh_update_count;
        } else {
          ++total_bvh_rebuild_count;
          if (envFlag("BQSIM_RT_DUMP_TREE_OWNER_AVG") && current_gate_idx > 0) {
            BuildGateEvent ev{};
            ev.gate_idx = current_gate_idx - 1;
            ev.traversal_begin_sample_idx = gate_launch_sample_count;
            ev.gate = gates[ev.gate_idx];
            ev.tree_build_row_nnz = running_max_row_nnz;
            ev.tree_final_row_nnz = running_max_row_nnz;
            last_stats.build_gate_events.push_back(ev);
          }
        }
        pop_nvtx(current_gate_is_target);
      }
      const auto sbt_setup_start = std::chrono::high_resolution_clock::now();
      impl->buildSbt();
      if (!impl->d_param) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->d_param), sizeof(Params)));
      }
      impl->state.params = {};
      impl->state.params.handle = impl->state.gas_handle;
      impl->state.params.mode = impl->state.rt_mode;
      CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(impl->d_param),
                                 &impl->state.params,
                                 sizeof(Params),
                                 cudaMemcpyHostToDevice,
                                 impl->stream));
      const auto sbt_setup_stop = std::chrono::high_resolution_clock::now();
      total_overhead_ms += std::chrono::duration<double, std::milli>(sbt_setup_stop - sbt_setup_start).count();

      cudaEvent_t launch_start = nullptr;
      cudaEvent_t launch_stop = nullptr;
      CUDA_CHECK(cudaEventCreate(&launch_start));
      CUDA_CHECK(cudaEventCreate(&launch_stop));
      const std::string launch_nvtx_label =
          current_gate_nvtx_prefix + " launch_mode" + std::to_string(impl->state.rt_mode);
      push_nvtx(launch_nvtx_label, current_gate_is_target);
      CUDA_CHECK(cudaEventRecord(launch_start, impl->stream));
      OPTIX_CHECK(optixLaunch(impl->state.pipeline,
                              impl->stream,
                              impl->d_param,
                              sizeof(Params),
                              &impl->state.sbt,
                              static_cast<unsigned int>(rays),
                              1,
                              1));
      CUDA_CHECK(cudaEventRecord(launch_stop, impl->stream));
      if (sync_after_launch) {
        CUDA_CHECK(cudaEventSynchronize(launch_stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, launch_start, launch_stop));
        total_launch_ms += ms;
        CUDA_CHECK(cudaEventDestroy(launch_start));
        CUDA_CHECK(cudaEventDestroy(launch_stop));
      } else {
        if (pending_launch_start || pending_launch_stop) {
          finalize_pending_launch();
        }
        pending_launch_start = launch_start;
        pending_launch_stop = launch_stop;
      }
      pop_nvtx(current_gate_is_target);
    };
    const auto pre_loop_stop = std::chrono::high_resolution_clock::now();
    total_overhead_ms +=
        std::chrono::duration<double, std::milli>(pre_loop_stop - pre_loop_start).count();
    auto launch_gate_generation = [&](size_t gate_idx, int slot) {
      const bool do_cull = shouldCullGateRays(gates[gate_idx], sample_row);
      h_gate_use_cull[slot] = do_cull;
      const int blocks_g = static_cast<int>((G_nnz + threads - 1) / threads);
      if (sync_stage_timing) {
        CUDA_CHECK(cudaEventRecord(gate_gen_start[slot], prep_stream));
      }
      build_gate_coo_kernel<<<blocks_n, threads, 0, prep_stream>>>(d_gates,
                                                                    static_cast<int>(gate_idx),
                                                                    static_cast<int>(nDim),
                                                                    d_G_rows[slot],
                                                                    d_G_cols[slot],
                                                                    d_G_vals[slot]);
      CUDA_CHECK(cudaGetLastError());
      if (do_cull) {
        mark_nonzero_gate_entries_kernel<<<blocks_g, threads, 0, prep_stream>>>(d_G_vals[slot],
                                                                                 d_G_flags[slot],
                                                                                 static_cast<int>(G_nnz));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cub::DeviceSelect::Flagged(d_gate_select_temp_storage,
                                              gate_select_temp_storage_bytes,
                                              d_G_rows[slot],
                                              d_G_flags[slot],
                                              d_G_rows_culled[slot],
                                              d_G_selected_count[slot],
                                              G_nnz,
                                              prep_stream));
        CUDA_CHECK(cub::DeviceSelect::Flagged(d_gate_select_temp_storage,
                                              gate_select_temp_storage_bytes,
                                              d_G_cols[slot],
                                              d_G_flags[slot],
                                              d_G_cols_culled[slot],
                                              d_G_selected_count[slot],
                                              G_nnz,
                                              prep_stream));
        CUDA_CHECK(cub::DeviceSelect::Flagged(d_gate_select_temp_storage,
                                              gate_select_temp_storage_bytes,
                                              d_G_vals[slot],
                                              d_G_flags[slot],
                                              d_G_vals_culled[slot],
                                              d_G_selected_count[slot],
                                              G_nnz,
                                              prep_stream));
        CUDA_CHECK(cudaMemcpyAsync(h_G_selected_count + slot,
                                   d_G_selected_count[slot],
                                   sizeof(int),
                                   cudaMemcpyDeviceToHost,
                                   prep_stream));
      } else {
        h_G_selected_count[slot] = static_cast<int>(G_nnz);
      }
      if (sync_stage_timing) {
        CUDA_CHECK(cudaEventRecord(gate_gen_stop[slot], prep_stream));
      }
      CUDA_CHECK(cudaEventRecord(gate_ready[slot], prep_stream));
    };
    launch_gate_generation(0, 0);
    CUDA_CHECK(cudaEventSynchronize(gate_ready[0]));
    const bool first_gate_use_cull = h_gate_use_cull[0];
    const int first_gate_nnz = h_G_selected_count[0];
    if (first_gate_nnz <= 0) {
      std::cerr << "[SPMSPM] gate has zero valid rays at gate 0" << std::endl;
      release_prebuild_resources();
      if (d_gates) {
        CUDA_CHECK(cudaFree(d_gates));
        d_gates = nullptr;
      }
      release_workspace();
      impl->cleanupGeometry();
      return false;
    }
    if (sync_stage_timing) {
      float raygen_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&raygen_ms, gate_gen_start[0], gate_gen_stop[0]));
      total_ray_gen_ms += raygen_ms;
    } else {
      CUDA_CHECK(cudaStreamSynchronize(prep_stream));
    }
    M_nnz = static_cast<size_t>(first_gate_nnz);
    CUDA_CHECK(cudaMemcpyAsync(d_M_rows,
                               first_gate_use_cull ? d_G_rows_culled[0] : d_G_rows[0],
                               M_nnz * sizeof(int),
                               cudaMemcpyDeviceToDevice,
                               impl->stream));
    CUDA_CHECK(cudaMemcpyAsync(d_M_cols,
                               first_gate_use_cull ? d_G_cols_culled[0] : d_G_cols[0],
                               M_nnz * sizeof(int),
                               cudaMemcpyDeviceToDevice,
                               impl->stream));
    CUDA_CHECK(cudaMemcpyAsync(d_M_vals,
                               first_gate_use_cull ? d_G_vals_culled[0] : d_G_vals[0],
                               M_nnz * sizeof(bqsim_rt::Complex),
                               cudaMemcpyDeviceToDevice,
                               impl->stream));
    CUDA_CHECK(cudaMemsetAsync(impl->state.d_row_counts, 0, nDim * sizeof(int), impl->stream));
    {
      const int blocks_cnt = static_cast<int>((M_nnz + threads - 1) / threads);
      count_rows_kernel<<<blocks_cnt, threads>>>(d_M_rows,
                                                 static_cast<int>(M_nnz),
                                                 impl->state.d_row_counts);
      CUDA_CHECK(cudaGetLastError());
      auto exec = thrust::cuda::par.on(impl->stream);
      auto row_counts_ptr = thrust::device_pointer_cast(impl->state.d_row_counts);
      running_max_row_nnz = static_cast<int>(
          thrust::reduce(exec, row_counts_ptr, row_counts_ptr + nDim, 0, thrust::maximum<int>()));
    }

    fused_gates_applied = 1;
    impl->last_fused_gates = fused_gates_applied;
    if (envFlag("BQSIM_RT_DUMP_GATE_TRAVERSAL")) {
      GateTraversalEvent seed_ev{};
      seed_ev.gate_idx = 0;
      seed_ev.gate = gates[0];
      seed_ev.tree_row_nnz_before = running_max_row_nnz;
      seed_ev.result_row_nnz_after = running_max_row_nnz;
      seed_ev.traversal_ms = 0.0;
      seed_ev.has_traversal = false;
      last_stats.gate_traversal_events.push_back(seed_ev);
    }
    bool previous_gate_was_diagonal = false;
    bool has_gas_tree = false;
    std::vector<double> gate_launch_ms;
    gate_launch_ms.reserve(max_gates);
    std::vector<int> gate_result_row_nnz_after;
    gate_result_row_nnz_after.reserve(max_gates);
    auto collect_shift_sample = [&](bool gate_refreshed_gas) {
      if (!enable_refit_shift_metric) {
        return;
      }
      double shift_sample = 0.0;
      if (gate_refreshed_gas && !impl->gas_last_update) {
        ensure_rebuild_snapshot_capacity(M_nnz);
        CUDA_CHECK(cudaMemcpyAsync(d_rebuild_snapshot_rows,
                                   d_M_rows,
                                   M_nnz * sizeof(int),
                                   cudaMemcpyDeviceToDevice,
                                   impl->stream));
        CUDA_CHECK(cudaMemcpyAsync(d_rebuild_snapshot_cols,
                                   d_M_cols,
                                   M_nnz * sizeof(int),
                                   cudaMemcpyDeviceToDevice,
                                   impl->stream));
        has_rebuild_snapshot = true;
        rebuild_snapshot_size = M_nnz;
      } else if (has_rebuild_snapshot && rebuild_snapshot_size == M_nnz && nDim > 1) {
        CUDA_CHECK(cudaMemsetAsync(d_position_delta_sum,
                                   0,
                                   sizeof(unsigned long long),
                                   impl->stream));
        const int blocks_delta = static_cast<int>((M_nnz + threads - 1) / threads);
        accumulate_position_delta_kernel<<<blocks_delta, threads, 0, impl->stream>>>(
            d_M_rows,
            d_M_cols,
            d_rebuild_snapshot_rows,
            d_rebuild_snapshot_cols,
            M_nnz,
            d_position_delta_sum);
        CUDA_CHECK(cudaGetLastError());
        unsigned long long host_delta = 0;
        CUDA_CHECK(cudaMemcpyAsync(&host_delta,
                                   d_position_delta_sum,
                                   sizeof(unsigned long long),
                                   cudaMemcpyDeviceToHost,
                                   impl->stream));
        CUDA_CHECK(cudaStreamSynchronize(impl->stream));
        const double max_axis_shift = 2.0 * static_cast<double>(nDim - 1);
        const double denom = static_cast<double>(M_nnz) * max_axis_shift;
        if (denom > 0.0) {
          shift_sample = static_cast<double>(host_delta) / denom;
        }
      }
      total_refit_shift_sum += shift_sample;
      ++total_refit_shift_samples;
    };

    size_t effective_max_gates = max_gates;
    if (!force_full && row_nnz_limit > 0 && running_max_row_nnz >= row_nnz_limit) {
      effective_max_gates = 1;
    }
    if (effective_max_gates > 1) {
      if (serial_prep_stream) {
        CUDA_CHECK(cudaStreamSynchronize(impl->stream));
      }
      launch_gate_generation(1, 1);
    }

    for (size_t g = 1; g < effective_max_gates; ++g) {
      current_gate_idx = g;
      const std::size_t global_gate_idx = impl->debug_block_start_gate + g;
      current_gate_is_target =
          has_target_global_gate && global_gate_idx == static_cast<std::size_t>(target_global_gate);
      current_gate_nvtx_prefix = "gate " + std::to_string(global_gate_idx);
      const double launch_before_gate = total_launch_ms;
      const auto raygen_start = std::chrono::high_resolution_clock::now();
      const int curr_slot = static_cast<int>(g & 1ULL);
      const int next_slot = curr_slot ^ 1;
      CUDA_CHECK(cudaEventSynchronize(gate_ready[curr_slot]));
      if (!serial_prep_stream && g + 1 < effective_max_gates) {
        launch_gate_generation(g + 1, next_slot);
      }
      const bool use_gate_cull = h_gate_use_cull[curr_slot];
      const int gate_nnz = h_G_selected_count[curr_slot];
      if (gate_nnz <= 0) {
        if (verbose) {
          std::cerr << "[SPMSPM] gate has zero valid rays at gate " << g << std::endl;
        }
        break;
      }
      if (verbose) {
        CUDA_CHECK(cudaStreamSynchronize(prep_stream));
      }
      if (sync_stage_timing) {
        float raygen_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&raygen_ms, gate_gen_start[curr_slot], gate_gen_stop[curr_slot]));
        total_ray_gen_ms += raygen_ms;
      } else {
        const auto raygen_stop = std::chrono::high_resolution_clock::now();
        total_ray_gen_ms += std::chrono::duration<double, std::milli>(raygen_stop - raygen_start).count();
      }

      bool is_diag = impl->diag_value_only && isDiagonalGate(gates[g]);
      const int row_nnz_before_gate = running_max_row_nnz;

      // Conservative pre-check: avoid entering a multiply that can push row nnz far beyond limit.
      if (!force_full && row_nnz_limit > 0 && !is_diag && fused_gates_applied > 0) {
        const int gate_row_nnz_ub = gateRowNNZUpperBound(gates[g]);
        const long long predicted_upper =
            static_cast<long long>(running_max_row_nnz) * static_cast<long long>(gate_row_nnz_ub);
        if (predicted_upper > static_cast<long long>(row_nnz_limit)) {
          break;
        }
      }

      if (current_gate_is_target) {
        dump_target_tree_csv(global_gate_idx, row_nnz_before_gate);
      }

      const bool need_gas_refresh = !has_gas_tree || !previous_gate_was_diagonal;

      if (need_gas_refresh) {
        impl->state.sphere_size = M_nnz;
        if (sync_stage_timing) {
          cudaEvent_t geom_start = nullptr;
          cudaEvent_t geom_stop = nullptr;
          CUDA_CHECK(cudaEventCreate(&geom_start));
          CUDA_CHECK(cudaEventCreate(&geom_stop));
          CUDA_CHECK(cudaEventRecord(geom_start, impl->stream));
          impl->ensureSphereBuffers(M_nnz);
          const int blocks_sphere = static_cast<int>((M_nnz + threads - 1) / threads);
          coo_to_sphere_kernel<<<blocks_sphere, threads, 0, impl->stream>>>(d_M_rows,
                                                                             d_M_cols,
                                                                             impl->state.spherePoints,
                                                                             impl->state.sphereRadius,
                                                                             static_cast<int>(M_nnz));
          CUDA_CHECK(cudaGetLastError());
          CUDA_CHECK(cudaEventRecord(geom_stop, impl->stream));
          CUDA_CHECK(cudaEventSynchronize(geom_stop));
          float geom_ms = 0.0f;
          CUDA_CHECK(cudaEventElapsedTime(&geom_ms, geom_start, geom_stop));
          total_geom_ms += geom_ms;
          CUDA_CHECK(cudaEventDestroy(geom_start));
          CUDA_CHECK(cudaEventDestroy(geom_stop));
        } else {
          const auto geom_start = std::chrono::high_resolution_clock::now();
          impl->ensureSphereBuffers(M_nnz);
          const int blocks_sphere = static_cast<int>((M_nnz + threads - 1) / threads);
          coo_to_sphere_kernel<<<blocks_sphere, threads, 0, impl->stream>>>(d_M_rows,
                                                                             d_M_cols,
                                                                             impl->state.spherePoints,
                                                                             impl->state.sphereRadius,
                                                                             static_cast<int>(M_nnz));
          CUDA_CHECK(cudaGetLastError());
          if (verbose) {
            CUDA_CHECK(cudaStreamSynchronize(impl->stream));
          }
          const auto geom_stop = std::chrono::high_resolution_clock::now();
          total_gas_ms += std::chrono::duration<double, std::milli>(geom_stop - geom_start).count();
        }
      } else {
        ++total_bvh_skip_count;
      }
      if (is_diag) {
          impl->num_rays = static_cast<size_t>(gate_nnz);
          impl->state.d_size = static_cast<uint64_t>(gate_nnz);
          impl->state.m_result_dim = make_int2(static_cast<int>(nDim), static_cast<int>(nDim));

          impl->state.d_ray_rows = use_gate_cull ? d_G_rows_culled[curr_slot] : d_G_rows[curr_slot];
          impl->state.d_ray_cols = use_gate_cull ? d_G_cols_culled[curr_slot] : d_G_cols[curr_slot];
          impl->state.d_ray_vals = use_gate_cull ? d_G_vals_culled[curr_slot] : d_G_vals[curr_slot];
          impl->state.sphereValues = d_M_vals;

          ensure_next_capacity(M_nnz);
          CUDA_CHECK(cudaMemsetAsync(d_N_vals, 0, M_nnz * sizeof(bqsim_rt::Complex), impl->stream));
          CUDA_CHECK(cudaMemcpyAsync(d_N_rows,
                                     d_M_rows,
                                     M_nnz * sizeof(int),
                                     cudaMemcpyDeviceToDevice,
                                     impl->stream));
          CUDA_CHECK(cudaMemcpyAsync(d_N_cols,
                                     d_M_cols,
                                     M_nnz * sizeof(int),
                                     cudaMemcpyDeviceToDevice,
                                     impl->stream));

          impl->state.d_out_vals = d_N_vals;
          impl->state.rt_mode = 3;
          launch_optix(static_cast<size_t>(gate_nnz), need_gas_refresh, true);
          collect_shift_sample(need_gas_refresh);

          impl->state.d_result = d_N_vals;
          impl->state.d_result_buf_size = M_nnz * sizeof(bqsim_rt::Complex);
          impl->num_rays = M_nnz;
          impl->merge_collision_free_hint = true;
      } else {
          const auto loop_overhead_start = std::chrono::high_resolution_clock::now();
          impl->num_rays = static_cast<size_t>(gate_nnz);
          impl->state.d_size = static_cast<uint64_t>(gate_nnz);
          impl->state.m_result_dim = make_int2(static_cast<int>(nDim), static_cast<int>(nDim));
          impl->state.d_ray_rows = use_gate_cull ? d_G_rows_culled[curr_slot] : d_G_rows[curr_slot];
          impl->state.d_ray_cols = use_gate_cull ? d_G_cols_culled[curr_slot] : d_G_cols[curr_slot];
          impl->state.d_ray_vals = use_gate_cull ? d_G_vals_culled[curr_slot] : d_G_vals[curr_slot];
          impl->state.sphereValues = d_M_vals;

          CUDA_CHECK(cudaMemsetAsync(impl->state.d_ray_counts,
                                     0,
                                     static_cast<size_t>(gate_nnz) * sizeof(int),
                                     impl->stream));
          CUDA_CHECK(cudaMemsetAsync(impl->state.d_row_counts, 0, nDim * sizeof(int), impl->stream));

          impl->state.rt_mode = 0;
          const auto sym_start = std::chrono::high_resolution_clock::now();
          total_overhead_ms += std::chrono::duration<double, std::milli>(sym_start - loop_overhead_start).count();
          launch_optix(static_cast<size_t>(gate_nnz), need_gas_refresh, true);
          collect_shift_sample(need_gas_refresh);
          auto exec = thrust::cuda::par.on(impl->stream);
          auto ray_counts_ptr = thrust::device_pointer_cast(impl->state.d_ray_counts);
          int total_hits = 0;
          auto scan_phase_end = std::chrono::high_resolution_clock::now();
          if (sync_stage_timing) {
            cudaEvent_t scan_start = nullptr;
            cudaEvent_t scan_stop = nullptr;
            CUDA_CHECK(cudaEventCreate(&scan_start));
            CUDA_CHECK(cudaEventCreate(&scan_stop));
            CUDA_CHECK(cudaEventRecord(scan_start, impl->stream));
            total_hits = static_cast<int>(thrust::reduce(exec,
                                                         ray_counts_ptr,
                                                         ray_counts_ptr + gate_nnz,
                                                         0));
            thrust::exclusive_scan(exec,
                                   ray_counts_ptr,
                                   ray_counts_ptr + gate_nnz,
                                   thrust::device_pointer_cast(impl->state.d_ray_offsets));
            CUDA_CHECK(cudaEventRecord(scan_stop, impl->stream));
            CUDA_CHECK(cudaEventSynchronize(scan_stop));
            float scan_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&scan_ms, scan_start, scan_stop));
            total_merge_ms += scan_ms;
            CUDA_CHECK(cudaEventDestroy(scan_start));
            CUDA_CHECK(cudaEventDestroy(scan_stop));
            scan_phase_end = std::chrono::high_resolution_clock::now();
          } else {
            const auto scan_start = std::chrono::high_resolution_clock::now();
            total_hits = static_cast<int>(thrust::reduce(exec,
                                                         ray_counts_ptr,
                                                         ray_counts_ptr + gate_nnz,
                                                         0));
            thrust::exclusive_scan(exec,
                                   ray_counts_ptr,
                                   ray_counts_ptr + gate_nnz,
                                   thrust::device_pointer_cast(impl->state.d_ray_offsets));
            const auto scan_stop = std::chrono::high_resolution_clock::now();
            total_merge_ms += std::chrono::duration<double, std::milli>(scan_stop - scan_start).count();
            scan_phase_end = scan_stop;
          }

          ensure_next_capacity(static_cast<std::size_t>(total_hits));
          ensure_tmp_vals_capacity(static_cast<std::size_t>(total_hits));
          impl->state.d_out_rows = d_N_rows;
          impl->state.d_out_cols = d_N_cols;
          impl->state.d_out_vals = d_tmp_vals;
          impl->state.out_capacity = static_cast<uint64_t>(N_capacity);
          CUDA_CHECK(cudaMemsetAsync(impl->state.d_ray_write_pos,
                                     0,
                                     static_cast<size_t>(gate_nnz) * sizeof(int),
                                     impl->stream));

          impl->state.rt_mode = 1;
          const auto num_start = std::chrono::high_resolution_clock::now();
          total_overhead_ms += std::chrono::duration<double, std::milli>(num_start - scan_phase_end).count();
          launch_optix(static_cast<size_t>(gate_nnz), false, false);
          const auto num_stop = std::chrono::high_resolution_clock::now();

          const bool collision_free_gate = isCollisionFreeGate(gates[g]);
          impl->state.d_result = d_N_vals;
          impl->state.d_result_buf_size = static_cast<size_t>(total_hits) * sizeof(bqsim_rt::Complex);
          if (!collision_free_gate) {
            CUDA_CHECK(cudaMemset(impl->state.d_result, 0, impl->state.d_result_buf_size));
          }

          impl->state.d_ray_rows = d_N_rows;
          impl->state.d_ray_cols = d_N_cols;
          impl->state.sphereValues = d_tmp_vals;
          impl->num_rays = static_cast<size_t>(total_hits);
          impl->state.d_size = impl->num_rays;
          impl->merge_collision_free_hint = false; 

          const auto merge_start = std::chrono::high_resolution_clock::now();
          total_overhead_ms += std::chrono::duration<double, std::milli>(merge_start - num_stop).count();
          const std::string merge_nvtx_label = current_gate_nvtx_prefix + " merge";
          push_nvtx(merge_nvtx_label, current_gate_is_target);
          if (sync_stage_timing) {
            finalize_pending_launch();
            float merge_ms = 0.0f;
            if (!impl->runCudaMerge(&merge_ms)) {
              pop_nvtx(current_gate_is_target);
              std::cerr << "[SPMSPM] CUDA merge failed at gate " << g << std::endl;
              release_prebuild_resources();
              if (d_gates) {
                CUDA_CHECK(cudaFree(d_gates));
                d_gates = nullptr;
              }
              release_workspace();
              impl->cleanupGeometry();
              return false;
            }
            total_merge_ms += merge_ms;
          } else {
            finalize_pending_launch();
            const auto merge_wall_start = std::chrono::high_resolution_clock::now();
            if (!impl->runCudaMerge()) {
              const auto merge_fail_stop = std::chrono::high_resolution_clock::now();
              total_merge_ms += std::chrono::duration<double, std::milli>(merge_fail_stop - merge_wall_start).count();
              pop_nvtx(current_gate_is_target);
              std::cerr << "[SPMSPM] CUDA merge failed at gate " << g << std::endl;
              release_prebuild_resources();
              if (d_gates) {
                CUDA_CHECK(cudaFree(d_gates));
                d_gates = nullptr;
              }
              release_workspace();
              impl->cleanupGeometry();
              return false;
            }
            const auto merge_stop = std::chrono::high_resolution_clock::now();
            total_merge_ms += std::chrono::duration<double, std::milli>(merge_stop - merge_wall_start).count();
          }
          pop_nvtx(current_gate_is_target);
      } // else is_diag

      const auto loop_tail_start = std::chrono::high_resolution_clock::now();

      std::swap(d_M_rows, d_N_rows);
      std::swap(d_M_cols, d_N_cols);
      std::swap(d_M_vals, d_N_vals);
      std::swap(M_capacity, N_capacity);
      M_nnz = impl->num_rays;
      impl->state.sphereValues = d_M_vals;
      impl->state.sphere_size = M_nnz;
      impl->state.d_out_rows = nullptr;
      impl->state.d_out_cols = nullptr;
      impl->state.d_out_vals = nullptr;
      impl->state.out_capacity = 0;

      ++fused_gates_applied;
      impl->last_fused_gates = fused_gates_applied;
      previous_gate_was_diagonal = is_diag;
      has_gas_tree = true;
      if (!is_diag && nDim > 0 && impl->state.d_row_counts) {
        auto exec = thrust::cuda::par.on(impl->stream);
        auto row_counts_ptr = thrust::device_pointer_cast(impl->state.d_row_counts);
        running_max_row_nnz = static_cast<int>(
            thrust::reduce(exec,
                           row_counts_ptr,
                           row_counts_ptr + nDim,
                           0,
                           thrust::maximum<int>()));
      }
      bool stop_after_this_gate =
          (!force_full && row_nnz_limit > 0 && running_max_row_nnz >= row_nnz_limit);
      const double gate_traversal_ms = total_launch_ms - launch_before_gate;
      gate_launch_ms.push_back(gate_traversal_ms);
      gate_result_row_nnz_after.push_back(running_max_row_nnz);
      if (envFlag("BQSIM_RT_DUMP_GATE_TRAVERSAL")) {
        GateTraversalEvent gate_ev{};
        gate_ev.gate_idx = g;
        gate_ev.traversal_sample_idx = gate_launch_sample_count;
        gate_ev.gate = gates[g];
        gate_ev.tree_row_nnz_before = row_nnz_before_gate;
        gate_ev.result_row_nnz_after = running_max_row_nnz;
        gate_ev.traversal_ms = gate_traversal_ms;
        gate_ev.has_traversal = true;
        last_stats.gate_traversal_events.push_back(gate_ev);
      }
      ++gate_launch_sample_count;
      if (stop_after_this_gate) {
        const auto loop_tail_stop = std::chrono::high_resolution_clock::now();
        total_overhead_ms +=
            std::chrono::duration<double, std::milli>(loop_tail_stop - loop_tail_start).count();
        break;
      }
      if (serial_prep_stream && g + 1 < effective_max_gates) {
        CUDA_CHECK(cudaStreamSynchronize(impl->stream));
        launch_gate_generation(g + 1, next_slot);
      }
      const auto loop_tail_stop = std::chrono::high_resolution_clock::now();
      total_overhead_ms +=
          std::chrono::duration<double, std::milli>(loop_tail_stop - loop_tail_start).count();
    }

    if (!last_stats.build_gate_events.empty() && !gate_launch_ms.empty()) {
      for (std::size_t i = 0; i < last_stats.build_gate_events.size(); ++i) {
        auto& ev = last_stats.build_gate_events[i];
        const std::size_t begin = ev.traversal_begin_sample_idx;
        std::size_t end = gate_launch_ms.size();
        if (i + 1 < last_stats.build_gate_events.size()) {
          end = std::min(end, last_stats.build_gate_events[i + 1].traversal_begin_sample_idx);
        }
        if (begin >= gate_launch_ms.size() || begin >= end) {
          ev.tree_final_row_nnz = ev.tree_build_row_nnz;
          ev.traversal_average_ms = 0.0;
          ev.traversal_sample_count = 0;
          continue;
        }
        const std::size_t last_sample_idx = end - 1;
        ev.tree_final_row_nnz =
            last_sample_idx < gate_result_row_nnz_after.size()
                ? gate_result_row_nnz_after[last_sample_idx]
                : ev.tree_build_row_nnz;
        double sum = 0.0;
        for (std::size_t gate_i = begin; gate_i < end; ++gate_i) {
          sum += gate_launch_ms[gate_i];
        }
        const std::size_t sample_count = end - begin;
        ev.traversal_sample_count = sample_count;
        ev.traversal_average_ms =
            sample_count > 0 ? (sum / static_cast<double>(sample_count)) : 0.0;
      }
    }

    const auto overhead_tail_start = std::chrono::high_resolution_clock::now();
    int* d_row_counts = impl->state.d_row_counts;
    CUDA_CHECK(cudaMemset(d_row_counts, 0, nDim * sizeof(int)));
    const int blocks_cnt = static_cast<int>((M_nnz + threads - 1) / threads);
    count_rows_kernel<<<blocks_cnt, threads>>>(d_M_rows, static_cast<int>(M_nnz), d_row_counts);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> host_counts;
    host_counts.resize(nDim);
    CUDA_CHECK(cudaMemcpy(host_counts.data(),
                          d_row_counts,
                          nDim * sizeof(int),
                          cudaMemcpyDeviceToHost));

    int max_row = 0;
    for (size_t i = 0; i < host_counts.size(); ++i) {
      if (host_counts[i] > max_row) {
        max_row = host_counts[i];
      }
    }
    impl->max_row_nnz = max_row;
    const auto overhead_tail_stop = std::chrono::high_resolution_clock::now();
    total_overhead_ms +=
        std::chrono::duration<double, std::milli>(overhead_tail_stop - overhead_tail_start).count();
    const auto cleanup_start = std::chrono::high_resolution_clock::now();

    impl->num_rays = M_nnz;
    impl->state.d_ray_rows = d_M_rows;
    impl->state.d_ray_cols = d_M_cols;
    impl->state.d_result = d_M_vals;
    impl->state.d_size = impl->num_rays;
    impl->state.d_result_buf_size = impl->num_rays * sizeof(bqsim_rt::Complex);
    impl->state.sphereValues = d_M_vals;
    impl->state.sphere_size = M_nnz;
    impl->state.m_result_dim = make_int2(static_cast<int>(nDim), static_cast<int>(nDim));
    impl->precomputed_result = true;
    finalize_pending_launch();

    impl->state.d_ray_vals = nullptr;
    release_prebuild_resources();
    CUDA_CHECK(cudaFree(d_gates));
    release_workspace();
    const auto cleanup_stop = std::chrono::high_resolution_clock::now();
    total_overhead_ms +=
        std::chrono::duration<double, std::milli>(cleanup_stop - cleanup_start).count();
    last_stats.geom_ms = total_geom_ms;
    last_stats.gas_ms = total_gas_ms;
    last_stats.launch_ms = total_launch_ms;
    last_stats.ray_gen_ms = total_ray_gen_ms;
    last_stats.merge_ms = total_merge_ms;
    last_stats.overhead_ms = total_overhead_ms;
    last_stats.compute_ms = total_launch_ms;
    last_stats.bvh_rebuild_count = total_bvh_rebuild_count;
    last_stats.bvh_update_count = total_bvh_update_count;
    last_stats.bvh_skip_count = total_bvh_skip_count;
    last_stats.bvh_refit_shift_sum = total_refit_shift_sum;
    last_stats.bvh_refit_shift_samples = total_refit_shift_samples;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[SPMSPM] Exception: " << e.what() << std::endl;
    impl->cleanupGeometry();
    return false;
  }
}

// Legacy entry is intentionally minimal in SPMSPM mode because Stage-1 precomputes results above.
bool RTSpMSpMEngine::launchRTMultiply() {
  if (!available || !impl) {
    return false;
  }
  if (impl->num_rays == 0) {
    return false;
  }
  if (impl->precomputed_result) {
    return true;
  }
  // In current SPMSPM flow, result is already prepared in prepareGeometryFromGates().
  return false;
}

// Convert final fused COO result on device into ELL buffers for Stage-2 simulation.
bool RTSpMSpMEngine::collectResultToELL(bqsim_rt::Complex* values,
                                        int* indices,
                                        int num_non_zeros,
                                        std::size_t nDim) {
  if (!available || !impl->state.d_result || !values || !indices) {
    return false;
  }
  if (impl->num_rays == 0) {
    return false;
  }

  int* d_row_counts = nullptr;
  try {
    auto ell_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemset(values, 0, sizeof(bqsim_rt::Complex) * nDim * num_non_zeros));
    CUDA_CHECK(cudaMemset(indices, 0, sizeof(int) * nDim * num_non_zeros));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_row_counts), nDim * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_row_counts, 0, nDim * sizeof(int)));

    int threads = 256;
    int blocks = static_cast<int>((impl->num_rays + threads - 1) / threads);
    coo_to_ell_kernel<<<blocks, threads>>>(impl->state.d_ray_rows,
                                           impl->state.d_ray_cols,
                                           impl->state.d_result,
                                           static_cast<int>(impl->num_rays),
                                           num_non_zeros,
                                           static_cast<int>(nDim),
                                           values,
                                           indices,
                                           d_row_counts);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_row_counts));
    auto ell_stop = std::chrono::high_resolution_clock::now();
    last_stats.ell_ms = std::chrono::duration<double, std::milli>(ell_stop - ell_start).count();
    return true;
  } catch (const std::exception&) {
    if (d_row_counts) {
      cudaFree(d_row_counts);
    }
    return false;
  }
}

int RTSpMSpMEngine::maxRowNNZ() const {
  return impl ? impl->max_row_nnz : 0;
}

std::size_t RTSpMSpMEngine::lastFusedGateCount() const {
  return impl ? impl->last_fused_gates : 0;
}

const RTSpMSpMEngine::Stats& RTSpMSpMEngine::lastStats() const {
  return last_stats;
}

void RTSpMSpMEngine::resetStats() {
  last_stats = {};
}

void RTSpMSpMEngine::warmup() {
  if (!available || !impl) {
    return;
  }
  impl->ensurePipeline();
}

void RTSpMSpMEngine::setDebugContext(const std::string& circuit_name, std::size_t block_start_gate) {
  if (!impl) {
    return;
  }
  impl->debug_circuit_name = circuit_name;
  impl->debug_block_start_gate = block_start_gate;
}

#endif  // BQSIM_USE_RTSPMSPM
