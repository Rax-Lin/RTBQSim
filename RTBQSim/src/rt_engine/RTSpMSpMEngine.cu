#include "RTSpMSpMEngine.hpp"

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

inline std::string cudaMemInfoSuffix() {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  const cudaError_t info_rc = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (info_rc != cudaSuccess) {
    cudaGetLastError();
    return std::string(" (cudaMemGetInfo failed: ") + cudaGetErrorString(info_rc) + ")";
  }
  std::ostringstream oss;
  oss << " (free=" << free_bytes << " bytes, total=" << total_bytes << " bytes)";
  return oss.str();
}

inline void checkCuda(cudaError_t rc, const char* msg) {
  if (rc != cudaSuccess) {
    std::ostringstream oss;
    oss << msg << ": " << cudaGetErrorString(rc);
    if (rc == cudaErrorMemoryAllocation) {
      oss << cudaMemInfoSuffix();
    }
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

inline void clearSpherePrimitiveBuffer(optixState& state) {
  state.spherePrimitiveBuffer = nullptr;
  state.spherePoints = nullptr;
  state.sphereRadius = nullptr;
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

bool envFlagDefaultTrue(const char* name) {
  const char* value = std::getenv(name);
  if (!value) {
    return true;
  }
  return envFlag(name);
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

enum class RTPrimitiveType {
  Triangle,
  Sphere,
};

RTPrimitiveType primitiveTypeFromEnv() {
  const char* value = std::getenv("RT_PRIMITIVE_TYPE");
  if (!value) {
    return RTPrimitiveType::Triangle;
  }
  if (std::strcmp(value, "sphere") == 0 || std::strcmp(value, "SPHERE") == 0) {
    return RTPrimitiveType::Sphere;
  }
  return RTPrimitiveType::Triangle;
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

__host__ __device__ inline bool isZeroMatrixEntry(const bqsim_rt::MatrixElem& value) {
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
// Heuristic:
// - uncontrolled dense 2x2 gates (e.g., X/H-like) have little to cull -> skip.
// - otherwise keep the previous conservative sample-row based rule.
bool shouldCullGateRays(const qc::GatePrimitive& gate, int sample_row) {
  if (gate.target_count != 1 || gate.matrix_dim != 2) {
    return false;
  }
  // Controlled 1-qubit gates produce identity rows when controls are not
  // satisfied, which introduces explicit zero-value rays in the 2-rays/row
  // materialization path. Force cull to drop those zero rays before launch.
  if (gate.control_count > 0) {
    return true;
  }
  int min_row_nnz = 2;
  for (int r = 0; r < 2; ++r) {
    int row_nnz = 0;
    for (int c = 0; c < 2; ++c) {
      if (!isZeroMatrixEntry(gate.matrix[r * 2 + c])) {
        ++row_nnz;
      }
    }
    if (row_nnz < min_row_nnz) {
      min_row_nnz = row_nnz;
    }
  }
  if (gate.control_count == 0 && min_row_nnz == 2) {
    return false;
  }

  bool controls_ok = true;
  for (int c = 0; c < gate.control_count; ++c) {
    const int qb = gate.controls[c];
    if (((sample_row >> qb) & 1) == 0) {
      controls_ok = false;
      break;
    }
  }
  if (!controls_ok) {
    return true;
  }
  const int target = gate.targets[0];
  const int bit = (sample_row >> target) & 1;
  const int m = bit * 2;
  int sample_row_nnz = 0;
  if (!isZeroMatrixEntry(gate.matrix[m])) {
    ++sample_row_nnz;
  }
  if (!isZeroMatrixEntry(gate.matrix[m + 1])) {
    ++sample_row_nnz;
  }
  return sample_row_nnz < 2;
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

// Map COO (row,col) points to one triangle (3 vertices) per NNZ for OptiX GAS.
// NOTE:
// Rays travel along +x with fixed y = col + 0.5 and fixed z = 0.5.
// To guarantee robust hits, each primitive triangle is placed on the plane
// x = col + 0.5 (not coplanar with ray direction) and its (y,z) footprint
// contains (row + 0.5, 0.5).
__global__ void coo_to_triangle_vertices_kernel(const int* rows,
                                                const int* cols,
                                                float3* out_vertices,
                                                int nnz) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nnz) {
    return;
  }
  const int row = rows[tid];
  const int col = cols[tid];
  const float c = static_cast<float>(col);
  const float r = static_cast<float>(row);
  const int base = tid * 3;
  out_vertices[base + 0] = make_float3(c + 0.5f, r + 0.1f, 0.1f);
  out_vertices[base + 1] = make_float3(c + 0.5f, r + 0.9f, 0.1f);
  out_vertices[base + 2] = make_float3(c + 0.5f, r + 0.5f, 0.9f);
}

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

__global__ void init_index_kernel(int n, int* indices) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }
  indices[tid] = tid;
}

__global__ void gather_selected_gate_entries_kernel(const int* in_rows,
                                                    const int* in_cols,
                                                    const bqsim_rt::Complex* in_vals,
                                                    const int* selected_indices,
                                                    const int* selected_count,
                                                    int* out_rows,
                                                    int* out_cols,
                                                    bqsim_rt::Complex* out_vals) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int nnz = *selected_count;
  if (tid >= nnz) {
    return;
  }
  const int src = selected_indices[tid];
  out_rows[tid] = in_rows[src];
  out_cols[tid] = in_cols[src];
  out_vals[tid] = in_vals[src];
}

__global__ void apply_left_diagonal_gate_kernel(const int* rows,
                                                const bqsim_rt::Complex* in_vals,
                                                bqsim_rt::Complex* out_vals,
                                                int nnz,
                                                qc::GatePrimitive gate) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nnz) {
    return;
  }
  const int row = rows[tid];
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
    const bqsim_rt::MatrixElem d = gate.matrix[diag_idx];
    factor = bqsim_rt::make_complex(d.x, d.y);
  }
  out_vals[tid] = bqsim_rt::cmul(factor, in_vals[tid]);
}

// Build inverse row map for 1-ray-per-row gates.
// For each destination row r, find source row src and scalar a such that:
// N[r, :] = a * M[src, :]. Store inverse by source index:
// inv_rows[src] = r, inv_scales[src] = a.
__global__ void build_inverse_rowmap_for_nnz1_gate_kernel(const qc::GatePrimitive gate,
                                                           int nDim,
                                                           int* inv_rows,
                                                           bqsim_rt::Complex* inv_scales) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
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
    const int bit = (r >> target) & 1;
    const int base = r & ~(1 << target);
    const int col0 = base;
    const int col1 = base | (1 << target);
    const int m = bit * 2;
    const bqsim_rt::MatrixElem a0 = gate.matrix[m];
    const bqsim_rt::MatrixElem a1 = gate.matrix[m + 1];
    const bool nz0 = (a0.x != 0.0 || a0.y != 0.0);
    if (nz0) {
      src = col0;
      a = bqsim_rt::make_complex(a0.x, a0.y);
    } else {
      src = col1;
      a = bqsim_rt::make_complex(a1.x, a1.y);
    }
  }

  inv_rows[src] = r;
  inv_scales[src] = a;
}

// Apply inverse row map to accumulated COO without RT traversal.
__global__ void apply_nnz1_gate_via_rowmap_kernel(const int* in_rows,
                                                   const int* in_cols,
                                                   const bqsim_rt::Complex* in_vals,
                                                   int nnz,
                                                   const int* inv_rows,
                                                   const bqsim_rt::Complex* inv_scales,
                                                   int* out_rows,
                                                   int* out_cols,
                                                   bqsim_rt::Complex* out_vals) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nnz) {
    return;
  }
  const int src_row = in_rows[tid];
  const int dst_row = inv_rows[src_row];
  const bool valid = (dst_row >= 0);
  out_rows[tid] = valid ? dst_row : src_row;
  out_cols[tid] = in_cols[tid];
  const bqsim_rt::Complex scale = valid ? inv_scales[src_row] : bqsim_rt::make_complex(1.0, 0.0);
  out_vals[tid] = bqsim_rt::cmul(scale, in_vals[tid]);
}

__global__ void count_rows_kernel(const int* rows, int nnz, int* row_counts) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nnz) {
    return;
  }
  const int row = rows[tid];
  atomicAdd(&row_counts[row], 1);
}

__global__ void compact_atomic_slots_kernel(const int* slot_cols,
                                            const bqsim_rt::Complex* slot_vals,
                                            const int* row_offsets,
                                            int nDim,
                                            int row_capacity,
                                            int* out_rows,
                                            int* out_cols,
                                            bqsim_rt::Complex* out_vals) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_slots = nDim * row_capacity;
  if (tid >= total_slots) {
    return;
  }
  const int row = tid / row_capacity;
  const int slot = tid % row_capacity;
  const int col = slot_cols[tid];
  if (col < 0) {
    return;
  }
  int local_pos = 0;
  const int base = row * row_capacity;
  for (int s = 0; s < slot; ++s) {
    if (slot_cols[base + s] >= 0) {
      ++local_pos;
    }
  }
  const int out_idx = row_offsets[row] + local_pos;
  out_rows[out_idx] = row;
  out_cols[out_idx] = col;
  out_vals[out_idx] = slot_vals[tid];
}

// Specialized compact kernel for fixed row_capacity=2.
// This avoids per-slot prefix loops in the generic compact kernel.
__global__ void compact_atomic_slots_row2_kernel(const int* slot_cols,
                                                 const bqsim_rt::Complex* slot_vals,
                                                 const int* row_offsets,
                                                 int nDim,
                                                 int* out_rows,
                                                 int* out_cols,
                                                 bqsim_rt::Complex* out_vals) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= nDim) {
    return;
  }
  const int base = row * 2;
  const int out_base = row_offsets[row];
  const int col0 = slot_cols[base];
  const int col1 = slot_cols[base + 1];

  int w = 0;
  if (col0 >= 0) {
    out_rows[out_base + w] = row;
    out_cols[out_base + w] = col0;
    out_vals[out_base + w] = slot_vals[base];
    ++w;
  }
  if (col1 >= 0) {
    out_rows[out_base + w] = row;
    out_cols[out_base + w] = col1;
    out_vals[out_base + w] = slot_vals[base + 1];
  }
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
  RTPrimitiveType primitive_type = RTPrimitiveType::Triangle;
  RTPrimitiveType pipeline_primitive_type = RTPrimitiveType::Triangle;
  optixState state{};
  CUstream stream = 0;
  CUdeviceptr d_param = 0;
  bool context_ready = false;
  bool pipeline_ready = false;
  bool gas_ready = false;
  int* d_atomic_row_offsets = nullptr;
  int* d_atomic_slot_cols = nullptr;
  bqsim_rt::Complex* d_atomic_slot_vals = nullptr;
  int* d_atomic_overflow = nullptr;
  size_t atomic_row_offset_capacity = 0;
  size_t atomic_slot_capacity = 0;
  int atomic_row_capacity = 4;
  int max_row_nnz = 0;
  bool precomputed_result = false;
  bool gas_allow_update = true;
  bool reuse_buffer = true;
  bool nnz1_special = false;
  bool require_single_anyhit_call = false;
  bool ell2_fast_path = true;
  qc::GatePrimitive* d_gates = nullptr;
  std::size_t gate_capacity = 0;
  int* d_work_rows[2] = {nullptr, nullptr};
  int* d_work_cols[2] = {nullptr, nullptr};
  bqsim_rt::Complex* d_work_vals[2] = {nullptr, nullptr};
  std::size_t work_capacity[2] = {0, 0};
  int* d_gate_rows[2] = {nullptr, nullptr};
  int* d_gate_cols[2] = {nullptr, nullptr};
  bqsim_rt::Complex* d_gate_vals[2] = {nullptr, nullptr};
  int* d_gate_rows_culled[2] = {nullptr, nullptr};
  int* d_gate_cols_culled[2] = {nullptr, nullptr};
  bqsim_rt::Complex* d_gate_vals_culled[2] = {nullptr, nullptr};
  int* d_gate_indices[2] = {nullptr, nullptr};
  int* d_gate_selected_indices[2] = {nullptr, nullptr};
  int* d_gate_flags[2] = {nullptr, nullptr};
  int* d_gate_selected_count[2] = {nullptr, nullptr};
  int* h_gate_selected_count = nullptr;
  void* d_gate_select_temp_storage = nullptr;
  std::size_t gate_select_temp_storage_bytes = 0;
  std::size_t gate_workspace_capacity = 0;
  cudaStream_t prep_stream = nullptr;
  bool prep_stream_alias_main = false;
  cudaEvent_t gate_ready[2] = {nullptr, nullptr};
  cudaEvent_t gate_gen_start[2] = {nullptr, nullptr};
  cudaEvent_t gate_gen_stop[2] = {nullptr, nullptr};
  bool gate_timing_events_ready = false;
  int* d_inv_rows = nullptr;
  bqsim_rt::Complex* d_inv_scales = nullptr;
  int* d_ell_row_counts = nullptr;
  size_t inv_map_capacity = 0;
  size_t ell_row_count_capacity = 0;
  size_t row_count_capacity = 0;
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
  RayDataRec raygen_record_host = {};
  MissSbtRecord miss_record_host = {};
  SphereDataRec hitgroup_record_host = {};
  bool raygen_header_ready = false;
  bool miss_header_ready = false;
  bool hitgroup_header_ready = false;
  std::string debug_circuit_name;
  size_t debug_block_start_gate = 0;

  void resetState() {
    const RTPrimitiveType requested_primitive_type = primitiveTypeFromEnv();
    if (requested_primitive_type != primitive_type) {
      cleanupPipeline();
      gas_ready = false;
      gas_prim_count = 0;
      if (state.triangleVertices) {
        safeCudaFree(state.triangleVertices, "cudaFree(state.triangleVertices)");
      }
      if (state.spherePrimitiveBuffer) {
        safeCudaFree(state.spherePrimitiveBuffer, "cudaFree(state.spherePrimitiveBuffer)");
        clearSpherePrimitiveBuffer(state);
      }
      sphere_capacity = 0;
      primitive_type = requested_primitive_type;
    }
    nDim = 0;
    num_rays = 0;
    max_row_nnz = 0;
    precomputed_result = false;
    state.primitiveCols = nullptr;
    gas_allow_update = envFlag("RT_GAS_ALLOW_UPDATE");
    if (!std::getenv("RT_GAS_ALLOW_UPDATE")) {
      gas_allow_update = true;
    }
    reuse_buffer = std::getenv("RT_REUSE_BUFFER")
                       ? envFlag("RT_REUSE_BUFFER")
                       : true;
    if (std::getenv("RT_NNZ1_SPECIAL")) {
      nnz1_special = envFlag("RT_NNZ1_SPECIAL");
    } else if (std::getenv("RT_DIAG_VALUE_ONLY")) {
      nnz1_special = envFlag("RT_DIAG_VALUE_ONLY");
    } else {
      nnz1_special = false;
    }
    require_single_anyhit_call = envFlag("REQUIRE_SINGLE_ANYHIT_CALL");
    if (!std::getenv("REQUIRE_SINGLE_ANYHIT_CALL")) {
      require_single_anyhit_call = false;
    }
    ell2_fast_path = envFlag("RT_ELL2_FAST_PATH");
    if (!std::getenv("RT_ELL2_FAST_PATH")) {
      ell2_fast_path = true;
    }
    atomic_row_capacity = static_cast<int>(envUInt64("BQSIM_RT_SPM_ROW_NNZ_LIMIT", 4ULL));
    if (atomic_row_capacity < 1) {
      atomic_row_capacity = 4;
    }
    last_fused_gates = 0;
    gas_prim_count = 0;
    gas_last_update = false;
  }

  void resetGeometryState() {
    state.d_ray_rows = nullptr;
    state.d_ray_cols = nullptr;
    state.d_ray_vals = nullptr;
    state.primitiveCols = nullptr;
    state.d_result = nullptr;
    state.d_out_rows = nullptr;
    state.d_out_cols = nullptr;
    state.d_out_vals = nullptr;
    state.out_capacity = 0;
    state.d_result_buf_size = 0;
    state.d_size = 0;
    state.sphere_size = 0;
    state.aabb_size = 0;
    state.m_result_dim = make_int2(0, 0);
    state.rt_mode = 0;
    state.procedural_raygen_mode = 0;
    state.current_gate = {};
    state.sphereValues = nullptr;
    state.deviceVertices = 0;
    state.devicePoints = 0;
    state.deviceRadius = 0;
    num_rays = 0;
    precomputed_result = false;
  }

  void cleanupGeometry() {
    resetGeometryState();
    if (state.spherePrimitiveBuffer) {
      safeCudaFree(state.spherePrimitiveBuffer, "cudaFree(state.spherePrimitiveBuffer)");
      clearSpherePrimitiveBuffer(state);
    }
    if (state.triangleVertices) {
      safeCudaFree(state.triangleVertices, "cudaFree(state.triangleVertices)");
    }
    if (d_gates) {
      safeCudaFree(d_gates, "cudaFree(d_gates)");
    }
    for (int slot = 0; slot < 2; ++slot) {
      if (d_work_rows[slot]) {
        safeCudaFree(d_work_rows[slot], "cudaFree(d_work_rows[slot])");
      }
      if (d_work_cols[slot]) {
        safeCudaFree(d_work_cols[slot], "cudaFree(d_work_cols[slot])");
      }
      if (d_work_vals[slot]) {
        safeCudaFree(d_work_vals[slot], "cudaFree(d_work_vals[slot])");
      }
      if (d_gate_rows[slot]) {
        safeCudaFree(d_gate_rows[slot], "cudaFree(d_gate_rows[slot])");
      }
      if (d_gate_cols[slot]) {
        safeCudaFree(d_gate_cols[slot], "cudaFree(d_gate_cols[slot])");
      }
      if (d_gate_vals[slot]) {
        safeCudaFree(d_gate_vals[slot], "cudaFree(d_gate_vals[slot])");
      }
      if (d_gate_rows_culled[slot]) {
        safeCudaFree(d_gate_rows_culled[slot], "cudaFree(d_gate_rows_culled[slot])");
      }
      if (d_gate_cols_culled[slot]) {
        safeCudaFree(d_gate_cols_culled[slot], "cudaFree(d_gate_cols_culled[slot])");
      }
      if (d_gate_vals_culled[slot]) {
        safeCudaFree(d_gate_vals_culled[slot], "cudaFree(d_gate_vals_culled[slot])");
      }
      if (d_gate_indices[slot]) {
        safeCudaFree(d_gate_indices[slot], "cudaFree(d_gate_indices[slot])");
      }
      if (d_gate_selected_indices[slot]) {
        safeCudaFree(d_gate_selected_indices[slot], "cudaFree(d_gate_selected_indices[slot])");
      }
      if (d_gate_flags[slot]) {
        safeCudaFree(d_gate_flags[slot], "cudaFree(d_gate_flags[slot])");
      }
      if (d_gate_selected_count[slot]) {
        safeCudaFree(d_gate_selected_count[slot], "cudaFree(d_gate_selected_count[slot])");
      }
      work_capacity[slot] = 0;
    }
    if (d_gate_select_temp_storage) {
      safeCudaFree(d_gate_select_temp_storage, "cudaFree(d_gate_select_temp_storage)");
    }
    if (h_gate_selected_count) {
      CUDA_CHECK(cudaFreeHost(h_gate_selected_count));
      h_gate_selected_count = nullptr;
    }
    if (prep_stream && !prep_stream_alias_main) {
      CUDA_CHECK(cudaStreamDestroy(prep_stream));
    }
    prep_stream = nullptr;
    prep_stream_alias_main = false;
    for (int slot = 0; slot < 2; ++slot) {
      if (gate_ready[slot]) {
        CUDA_CHECK(cudaEventDestroy(gate_ready[slot]));
        gate_ready[slot] = nullptr;
      }
      if (gate_gen_start[slot]) {
        CUDA_CHECK(cudaEventDestroy(gate_gen_start[slot]));
        gate_gen_start[slot] = nullptr;
      }
      if (gate_gen_stop[slot]) {
        CUDA_CHECK(cudaEventDestroy(gate_gen_stop[slot]));
        gate_gen_stop[slot] = nullptr;
      }
    }
    if (state.d_row_counts) {
      safeCudaFree(state.d_row_counts, "cudaFree(state.d_row_counts)");
    }
    if (state.d_out_count) {
      safeCudaFree(state.d_out_count, "cudaFree(state.d_out_count)");
    }
    if (d_atomic_row_offsets) {
      safeCudaFree(d_atomic_row_offsets, "cudaFree(d_atomic_row_offsets)");
    }
    if (d_atomic_slot_cols) {
      safeCudaFree(d_atomic_slot_cols, "cudaFree(d_atomic_slot_cols)");
    }
    if (d_atomic_slot_vals) {
      safeCudaFree(d_atomic_slot_vals, "cudaFree(d_atomic_slot_vals)");
    }
    if (d_atomic_overflow) {
      safeCudaFree(d_atomic_overflow, "cudaFree(d_atomic_overflow)");
    }
    if (d_inv_rows) {
      safeCudaFree(d_inv_rows, "cudaFree(d_inv_rows)");
    }
    if (d_inv_scales) {
      safeCudaFree(d_inv_scales, "cudaFree(d_inv_scales)");
    }
    if (d_ell_row_counts) {
      safeCudaFree(d_ell_row_counts, "cudaFree(d_ell_row_counts)");
    }
    gate_capacity = 0;
    gate_workspace_capacity = 0;
    gate_select_temp_storage_bytes = 0;
    gate_timing_events_ready = false;
    atomic_row_offset_capacity = 0;
    atomic_slot_capacity = 0;
    inv_map_capacity = 0;
    ell_row_count_capacity = 0;
    row_count_capacity = 0;
    sphere_capacity = 0;
  }

  void ensureGateCapacity(std::size_t gate_count) {
    if (gate_count == 0) {
      return;
    }
    if (d_gates && gate_capacity >= gate_count) {
      return;
    }
    if (d_gates) {
      safeCudaFree(d_gates, "cudaFree(d_gates)");
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gates), gate_count * sizeof(qc::GatePrimitive)));
    gate_capacity = gate_count;
  }

  void ensureWorkCapacity(int slot, std::size_t required) {
    if (required == 0) {
      return;
    }
    if (reuse_buffer && d_work_rows[slot] && work_capacity[slot] >= required) {
      return;
    }
    if (d_work_rows[slot]) {
      safeCudaFree(d_work_rows[slot], "cudaFree(d_work_rows[slot])");
    }
    if (d_work_cols[slot]) {
      safeCudaFree(d_work_cols[slot], "cudaFree(d_work_cols[slot])");
    }
    if (d_work_vals[slot]) {
      safeCudaFree(d_work_vals[slot], "cudaFree(d_work_vals[slot])");
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_work_rows[slot]), required * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_work_cols[slot]), required * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_work_vals[slot]), required * sizeof(bqsim_rt::Complex)));
    work_capacity[slot] = required;
  }

  void ensureGateWorkspaceCapacity(std::size_t required_nnz, int threads) {
    if (required_nnz == 0) {
      return;
    }
    if (d_gate_rows[0] && gate_workspace_capacity >= required_nnz) {
      return;
    }
    for (int slot = 0; slot < 2; ++slot) {
      if (d_gate_rows[slot]) {
        safeCudaFree(d_gate_rows[slot], "cudaFree(d_gate_rows[slot])");
      }
      if (d_gate_cols[slot]) {
        safeCudaFree(d_gate_cols[slot], "cudaFree(d_gate_cols[slot])");
      }
      if (d_gate_vals[slot]) {
        safeCudaFree(d_gate_vals[slot], "cudaFree(d_gate_vals[slot])");
      }
      if (d_gate_rows_culled[slot]) {
        safeCudaFree(d_gate_rows_culled[slot], "cudaFree(d_gate_rows_culled[slot])");
      }
      if (d_gate_cols_culled[slot]) {
        safeCudaFree(d_gate_cols_culled[slot], "cudaFree(d_gate_cols_culled[slot])");
      }
      if (d_gate_vals_culled[slot]) {
        safeCudaFree(d_gate_vals_culled[slot], "cudaFree(d_gate_vals_culled[slot])");
      }
      if (d_gate_indices[slot]) {
        safeCudaFree(d_gate_indices[slot], "cudaFree(d_gate_indices[slot])");
      }
      if (d_gate_selected_indices[slot]) {
        safeCudaFree(d_gate_selected_indices[slot], "cudaFree(d_gate_selected_indices[slot])");
      }
      if (d_gate_flags[slot]) {
        safeCudaFree(d_gate_flags[slot], "cudaFree(d_gate_flags[slot])");
      }
      if (d_gate_selected_count[slot]) {
        safeCudaFree(d_gate_selected_count[slot], "cudaFree(d_gate_selected_count[slot])");
      }

      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gate_rows[slot]), required_nnz * sizeof(int)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gate_cols[slot]), required_nnz * sizeof(int)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gate_vals[slot]), required_nnz * sizeof(bqsim_rt::Complex)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gate_rows_culled[slot]), required_nnz * sizeof(int)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gate_cols_culled[slot]), required_nnz * sizeof(int)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gate_vals_culled[slot]), required_nnz * sizeof(bqsim_rt::Complex)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gate_indices[slot]), required_nnz * sizeof(int)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gate_selected_indices[slot]), required_nnz * sizeof(int)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gate_flags[slot]), required_nnz * sizeof(int)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gate_selected_count[slot]), sizeof(int)));

      const int blocks_idx = static_cast<int>((required_nnz + threads - 1) / threads);
      init_index_kernel<<<blocks_idx, threads>>>(static_cast<int>(required_nnz), d_gate_indices[slot]);
      CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    if (!h_gate_selected_count) {
      CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_gate_selected_count), 2 * sizeof(int)));
    }
    h_gate_selected_count[0] = 0;
    h_gate_selected_count[1] = 0;

    size_t select_temp_index_bytes = 0;
    CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr,
                                          select_temp_index_bytes,
                                          d_gate_indices[0],
                                          d_gate_flags[0],
                                          d_gate_selected_indices[0],
                                          d_gate_selected_count[0],
                                          required_nnz));
    if (gate_select_temp_storage_bytes < select_temp_index_bytes) {
      if (d_gate_select_temp_storage) {
        safeCudaFree(d_gate_select_temp_storage, "cudaFree(d_gate_select_temp_storage)");
      }
      if (select_temp_index_bytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_gate_select_temp_storage, select_temp_index_bytes));
      }
      gate_select_temp_storage_bytes = select_temp_index_bytes;
    }
    gate_workspace_capacity = required_nnz;
  }

  void ensurePrepResources(bool enable_timing) {
    if (!stream) {
      CUDA_CHECK(cudaStreamCreate(&stream));
    }
    if (!prep_stream) {
      prep_stream = stream;
      prep_stream_alias_main = true;
    }
    for (int slot = 0; slot < 2; ++slot) {
      if (!gate_ready[slot]) {
        CUDA_CHECK(cudaEventCreateWithFlags(&gate_ready[slot], cudaEventDisableTiming));
      }
    }
    if (!enable_timing || gate_timing_events_ready) {
      return;
    }
    for (int slot = 0; slot < 2; ++slot) {
      if (!gate_gen_start[slot]) {
        CUDA_CHECK(cudaEventCreate(&gate_gen_start[slot]));
      }
      if (!gate_gen_stop[slot]) {
        CUDA_CHECK(cudaEventCreate(&gate_gen_stop[slot]));
      }
    }
    gate_timing_events_ready = true;
  }

  void ensureRowCountCapacity(std::size_t rows) {
    if (rows == 0) {
      return;
    }
    if (!state.d_row_counts || row_count_capacity < rows) {
      if (state.d_row_counts) {
        safeCudaFree(state.d_row_counts, "cudaFree(state.d_row_counts)");
      }
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_row_counts), rows * sizeof(int)));
      row_count_capacity = rows;
    }
    if (!state.d_out_count) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_out_count), sizeof(int)));
    }
  }

  void ensureEllRowCountCapacity(std::size_t rows) {
    if (rows == 0) {
      return;
    }
    if (d_ell_row_counts && ell_row_count_capacity >= rows) {
      return;
    }
    if (d_ell_row_counts) {
      safeCudaFree(d_ell_row_counts, "cudaFree(d_ell_row_counts)");
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_ell_row_counts), rows * sizeof(int)));
    ell_row_count_capacity = rows;
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
    raygen_header_ready = false;
    miss_header_ready = false;
    hitgroup_header_ready = false;
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
    state.pipeline_compile_options.usesPrimitiveTypeFlags =
        (primitive_type == RTPrimitiveType::Sphere)
            ? OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE
            : OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

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
    hitgroup_prog_group_desc.hitgroup.moduleIS =
        (primitive_type == RTPrimitiveType::Sphere) ? state.sphere_module : nullptr;
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
    pipeline_primitive_type = primitive_type;
  }

  void ensurePrimitiveBuffers(size_t required_primitives) {
    if (required_primitives == 0) {
      return;
    }
    if (primitive_type == RTPrimitiveType::Sphere) {
      if (reuse_buffer && state.spherePrimitiveBuffer && state.spherePoints && state.sphereRadius &&
          sphere_capacity >= required_primitives) {
        return;
      }
      if (state.spherePrimitiveBuffer) {
        safeCudaFree(state.spherePrimitiveBuffer, "cudaFree(state.spherePrimitiveBuffer)");
        clearSpherePrimitiveBuffer(state);
      }
      const std::size_t points_bytes = required_primitives * sizeof(float3);
      const std::size_t radius_bytes = required_primitives * sizeof(float);
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.spherePrimitiveBuffer),
                            points_bytes + radius_bytes));
      auto* sphere_buffer_bytes = reinterpret_cast<unsigned char*>(state.spherePrimitiveBuffer);
      state.spherePoints = reinterpret_cast<float3*>(sphere_buffer_bytes);
      state.sphereRadius = reinterpret_cast<float*>(sphere_buffer_bytes + points_bytes);
      sphere_capacity = required_primitives;
      return;
    }
    if (reuse_buffer && state.triangleVertices && sphere_capacity >= required_primitives) {
      return;
    }
    if (state.triangleVertices) {
      safeCudaFree(state.triangleVertices, "cudaFree(state.triangleVertices)");
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.triangleVertices),
                          required_primitives * 3 * sizeof(float3)));
    sphere_capacity = required_primitives;
  }

  // Build or update GAS from current primitive geometry.
  void buildGas(bool try_update = false) {
    if (primitive_type == RTPrimitiveType::Sphere) {
      if (!state.spherePoints || !state.sphereRadius || state.sphere_size == 0) {
        throw std::runtime_error("buildGas: sphere geometry is not ready");
      }
    } else if (!state.triangleVertices || state.sphere_size == 0) {
      throw std::runtime_error("buildGas: triangle geometry is not ready");
    }

    const bool allow_update = gas_allow_update;
    const bool same_prim_count = (gas_prim_count == state.sphere_size);
    bool do_update = try_update && gas_ready && allow_update && same_prim_count;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    if (allow_update) {
      accel_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    }
    accel_options.operation = do_update ? OPTIX_BUILD_OPERATION_UPDATE : OPTIX_BUILD_OPERATION_BUILD;

    OptixBuildInput build_input = {};
    uint32_t build_input_flags[1] = {
        require_single_anyhit_call
            ? OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL
            : OPTIX_GEOMETRY_FLAG_NONE};
    if (primitive_type == RTPrimitiveType::Sphere) {
      build_input.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
      state.devicePoints = reinterpret_cast<CUdeviceptr>(state.spherePoints);
      state.deviceRadius = reinterpret_cast<CUdeviceptr>(state.sphereRadius);
      build_input.sphereArray.numVertices = static_cast<unsigned int>(state.sphere_size);
      build_input.sphereArray.vertexBuffers = &state.devicePoints;
      build_input.sphereArray.vertexStrideInBytes = sizeof(float3);
      build_input.sphereArray.radiusBuffers = &state.deviceRadius;
      build_input.sphereArray.radiusStrideInBytes = sizeof(float);
      build_input.sphereArray.flags = build_input_flags;
      build_input.sphereArray.numSbtRecords = 1;
    } else {
      build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
      state.deviceVertices = reinterpret_cast<CUdeviceptr>(state.triangleVertices);
      build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
      build_input.triangleArray.vertexStrideInBytes = sizeof(float3);
      build_input.triangleArray.numVertices = static_cast<unsigned int>(state.sphere_size * 3ULL);
      build_input.triangleArray.vertexBuffers = &state.deviceVertices;
      build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_NONE;
      build_input.triangleArray.indexStrideInBytes = 0;
      build_input.triangleArray.numIndexTriplets = 0;
      build_input.triangleArray.indexBuffer = 0;
      build_input.triangleArray.flags = build_input_flags;
      build_input.triangleArray.numSbtRecords = 1;
    }

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
          (!reuse_buffer);
      if (need_new_output_buffer) {
        if (state.d_gas_output_buffer) {
          CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_gas_output_buffer)));
          state.d_gas_output_buffer = 0;
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_gas_output_buffer),
                              gas_buffer_sizes.outputSizeInBytes));
        gas_output_capacity = gas_buffer_sizes.outputSizeInBytes;
      }

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
    gas_ready = true;
    gas_prim_count = state.sphere_size;
    gas_last_update = built_with_update;
  }

  // Refresh SBT records to bind current ray/sphere/result buffers before each launch.
  void buildSbt() {
    if (!raygen_record) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), sizeof(RayDataRec)));
    }
    if (!raygen_header_ready) {
      std::memset(&raygen_record_host, 0, sizeof(raygen_record_host));
      OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prog_group, &raygen_record_host));
      raygen_header_ready = true;
    }
    raygen_record_host.data.rows = state.d_ray_rows;
    raygen_record_host.data.cols = state.d_ray_cols;
    raygen_record_host.data.values = state.d_ray_vals;
    raygen_record_host.data.size = state.d_size;
    raygen_record_host.data.nDim = nDim;
    raygen_record_host.data.proceduralMode = state.procedural_raygen_mode;
    raygen_record_host.data.gate = state.current_gate;
    if (stream) {
      CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(raygen_record),
                                 &raygen_record_host,
                                 sizeof(RayDataRec),
                                 cudaMemcpyHostToDevice,
                                 stream));
    } else {
      CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(raygen_record),
                            &raygen_record_host,
                            sizeof(RayDataRec),
                            cudaMemcpyHostToDevice));
    }

    if (!miss_record) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), sizeof(MissSbtRecord)));
    }
    if (!miss_header_ready) {
      std::memset(&miss_record_host, 0, sizeof(miss_record_host));
      OPTIX_CHECK(optixSbtRecordPackHeader(state.miss_prog_group, &miss_record_host));
      miss_record_host.data = {0.0f, 0.0f, 0.0f};
      if (stream) {
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(miss_record),
                                   &miss_record_host,
                                   sizeof(MissSbtRecord),
                                   cudaMemcpyHostToDevice,
                                   stream));
      } else {
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(miss_record),
                              &miss_record_host,
                              sizeof(MissSbtRecord),
                              cudaMemcpyHostToDevice));
      }
      miss_header_ready = true;
    }

    if (!hitgroup_record) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), sizeof(SphereDataRec)));
    }
    if (!hitgroup_header_ready) {
      std::memset(&hitgroup_record_host, 0, sizeof(hitgroup_record_host));
      OPTIX_CHECK(optixSbtRecordPackHeader(state.hit_prog_group, &hitgroup_record_host));
      hitgroup_header_ready = true;
    }
    hitgroup_record_host.data.aabbs = nullptr;
    hitgroup_record_host.data.sphereColor = state.sphereValues;
    hitgroup_record_host.data.primitiveCols = state.primitiveCols;
    hitgroup_record_host.data.rayValues = state.d_ray_vals;
    hitgroup_record_host.data.result = state.d_result;
    hitgroup_record_host.data.resultNumRow = state.m_result_dim.x;
    hitgroup_record_host.data.resultNumCol = state.m_result_dim.y;
    hitgroup_record_host.data.matrix1size = state.d_size;
    hitgroup_record_host.data.matrix2size = state.sphere_size;
    hitgroup_record_host.data.rayRows = state.d_ray_rows;
    hitgroup_record_host.data.rayCols = state.d_ray_cols;
    hitgroup_record_host.data.outRows = state.d_out_rows;
    hitgroup_record_host.data.outCols = state.d_out_cols;
    hitgroup_record_host.data.outVals = state.d_out_vals;
    hitgroup_record_host.data.outCount = state.d_out_count;
    hitgroup_record_host.data.outCapacity = state.out_capacity;
    hitgroup_record_host.data.mode = state.rt_mode;
    hitgroup_record_host.data.nDim = nDim;
    hitgroup_record_host.data.proceduralMode = state.procedural_raygen_mode;
    hitgroup_record_host.data.gate = state.current_gate;
    if (stream) {
      CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(hitgroup_record),
                                 &hitgroup_record_host,
                                 sizeof(SphereDataRec),
                                 cudaMemcpyHostToDevice,
                                 stream));
    } else {
      CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(hitgroup_record),
                            &hitgroup_record_host,
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
// 1) gate->COO generation, 2) OptiX launch, 3) compact row-slots to COO, 4) keep fused result on device.
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
    impl->resetGeometryState();
    impl->resetState();
    impl->last_fused_gates = 0;
    last_stats.build_gate_events.clear();
    last_stats.gate_traversal_events.clear();

    const uint64_t max_gates_env = envUInt64("BQSIM_RT_SPM_MAX_GATES", static_cast<uint64_t>(gate_count));
    const size_t max_gates = std::min(static_cast<size_t>(max_gates_env), gate_count);
    const bool verbose = envFlag("BQSIM_RT_SPM_VERBOSE");
    const int row_nnz_limit = static_cast<int>(envUInt64("BQSIM_RT_SPM_ROW_NNZ_LIMIT", 4ULL));
    // Fixed defaults: always use synchronized stage timing with aggressive
    // prep-stream overlap (no host-side serialization before gate generation).
    const bool collect_breakdown = envFlagDefaultTrue("BQSIM_ENABLE_BREAKDOWN");
    const bool sync_stage_timing = collect_breakdown;
    const int sample_row = (nDim > 1) ? 1 : 0;
    if (impl->pipeline_ready && impl->pipeline_primitive_type != impl->primitive_type) {
      impl->cleanupPipeline();
    }
    int running_max_row_nnz = 1;
    double total_geom_ms = 0.0;
    double total_gas_ms = 0.0;
    double total_launch_ms = 0.0;
    double total_ray_gen_ms = 0.0;
    double total_nnz1_mul_ms = 0.0;
    double total_compact_ms = 0.0;
    double total_overhead_ms = 0.0;
    double total_cleanup_ms = 0.0;
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
    impl->ensureGateCapacity(gate_count);
    qc::GatePrimitive* d_gates = impl->d_gates;
    const auto h2d_start = std::chrono::high_resolution_clock::now();
    total_overhead_ms += std::chrono::duration<double, std::milli>(h2d_start - setup_start).count();
    CUDA_CHECK(cudaMemcpy(d_gates, gates, gate_count * sizeof(qc::GatePrimitive), cudaMemcpyHostToDevice));
    const auto h2d_stop = std::chrono::high_resolution_clock::now();
    last_stats.h2d_ms = std::chrono::duration<double, std::milli>(h2d_stop - h2d_start).count();
    const auto pre_loop_start = std::chrono::high_resolution_clock::now();

    const int threads = 256;
    const int blocks_n = static_cast<int>((nDim + threads - 1) / threads);
    const size_t G_nnz = static_cast<size_t>(nDim) * 2;

    impl->ensureWorkCapacity(0, G_nnz);
    impl->ensureWorkCapacity(1, G_nnz);
    int M_slot = 0;
    int N_slot = 1;
    int* d_M_rows = impl->d_work_rows[M_slot];
    int* d_M_cols = impl->d_work_cols[M_slot];
    bqsim_rt::Complex* d_M_vals = impl->d_work_vals[M_slot];
    size_t M_nnz = 0;
    impl->ensureGateWorkspaceCapacity(G_nnz, threads);
    int* d_G_rows[2] = {impl->d_gate_rows[0], impl->d_gate_rows[1]};
    int* d_G_cols[2] = {impl->d_gate_cols[0], impl->d_gate_cols[1]};
    bqsim_rt::Complex* d_G_vals[2] = {impl->d_gate_vals[0], impl->d_gate_vals[1]};
    int* d_G_rows_culled[2] = {impl->d_gate_rows_culled[0], impl->d_gate_rows_culled[1]};
    int* d_G_cols_culled[2] = {impl->d_gate_cols_culled[0], impl->d_gate_cols_culled[1]};
    bqsim_rt::Complex* d_G_vals_culled[2] = {impl->d_gate_vals_culled[0], impl->d_gate_vals_culled[1]};
    int* d_G_indices[2] = {impl->d_gate_indices[0], impl->d_gate_indices[1]};
    int* d_G_selected_indices[2] = {impl->d_gate_selected_indices[0], impl->d_gate_selected_indices[1]};
    int* d_G_flags[2] = {impl->d_gate_flags[0], impl->d_gate_flags[1]};
    int* d_G_selected_count[2] = {impl->d_gate_selected_count[0], impl->d_gate_selected_count[1]};
    int* h_G_selected_count = impl->h_gate_selected_count;
    bool h_gate_use_cull[2] = {false, false};
    void* d_gate_select_temp_storage = impl->d_gate_select_temp_storage;
    size_t gate_select_temp_storage_bytes = impl->gate_select_temp_storage_bytes;
    impl->ensurePrepResources(sync_stage_timing);
    cudaStream_t prep_stream = impl->prep_stream;
    cudaEvent_t* gate_ready = impl->gate_ready;
    cudaEvent_t* gate_gen_start = impl->gate_gen_start;
    cudaEvent_t* gate_gen_stop = impl->gate_gen_stop;

    impl->ensureRowCountCapacity(nDim);

    std::size_t M_capacity = impl->work_capacity[M_slot];
    int* d_N_rows = impl->d_work_rows[N_slot];
    int* d_N_cols = impl->d_work_cols[N_slot];
    bqsim_rt::Complex* d_N_vals = impl->d_work_vals[N_slot];
    std::size_t N_capacity = impl->work_capacity[N_slot];

    auto ensure_next_capacity = [&](std::size_t required) {
      if (required == 0) {
        return;
      }
      impl->ensureWorkCapacity(N_slot, required);
      d_N_rows = impl->d_work_rows[N_slot];
      d_N_cols = impl->d_work_cols[N_slot];
      d_N_vals = impl->d_work_vals[N_slot];
      N_capacity = impl->work_capacity[N_slot];
    };

    auto ensure_inverse_map_capacity = [&](std::size_t rows) {
      if (rows == 0) {
        return;
      }
      if (impl->inv_map_capacity >= rows && impl->d_inv_rows && impl->d_inv_scales) {
        return;
      }
      if (impl->d_inv_rows) {
        safeCudaFree(impl->d_inv_rows, "cudaFree(d_inv_rows)");
      }
      if (impl->d_inv_scales) {
        safeCudaFree(impl->d_inv_scales, "cudaFree(d_inv_scales)");
      }
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->d_inv_rows), rows * sizeof(int)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->d_inv_scales), rows * sizeof(bqsim_rt::Complex)));
      impl->inv_map_capacity = rows;
    };

    // Pre-allocate once using row-NNZ limit baseline:
    // nDim * (row_limit * 2), with runtime expansion as safety fallback.
    const std::size_t prealloc_row_nnz = static_cast<std::size_t>(std::max(row_nnz_limit, 1)) * 2ULL;
    const std::size_t prealloc_capacity = static_cast<std::size_t>(nDim) * prealloc_row_nnz;
    ensure_next_capacity(prealloc_capacity);

    auto finalize_workspace_bindings = [&]() {
      impl->d_work_rows[M_slot] = d_M_rows;
      impl->d_work_cols[M_slot] = d_M_cols;
      impl->d_work_vals[M_slot] = d_M_vals;
      impl->work_capacity[M_slot] = M_capacity;
      impl->d_work_rows[N_slot] = d_N_rows;
      impl->d_work_cols[N_slot] = d_N_cols;
      impl->d_work_vals[N_slot] = d_N_vals;
      impl->work_capacity[N_slot] = N_capacity;
      impl->d_gate_select_temp_storage = d_gate_select_temp_storage;
      impl->gate_select_temp_storage_bytes = gate_select_temp_storage_bytes;
    };

    if (!impl->stream) {
      CUDA_CHECK(cudaStreamCreate(&impl->stream));
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

    const auto pipeline_setup_start = std::chrono::high_resolution_clock::now();
    impl->ensurePipeline();
    const auto pipeline_setup_stop = std::chrono::high_resolution_clock::now();
    total_overhead_ms +=
        std::chrono::duration<double, std::milli>(pipeline_setup_stop - pipeline_setup_start).count();

    auto launch_optix = [&](size_t rays, bool refresh_gas, bool sync_after_launch) {
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
          if (envFlag("RT_DUMP_TREE_OWNER_AVG") && current_gate_idx > 0) {
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
      if (!do_cull && gate_idx > 0) {
        const bool one_ray_per_row =
            impl->nnz1_special && gateHasOneRayPerRow(gates[gate_idx]);
        h_G_selected_count[slot] = static_cast<int>(one_ray_per_row ? nDim : G_nnz);
        if (sync_stage_timing) {
          CUDA_CHECK(cudaEventRecord(gate_gen_stop[slot], prep_stream));
        }
        CUDA_CHECK(cudaEventRecord(gate_ready[slot], prep_stream));
        return;
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
                                              d_G_indices[slot],
                                              d_G_flags[slot],
                                              d_G_selected_indices[slot],
                                              d_G_selected_count[slot],
                                              G_nnz,
                                              prep_stream));
        gather_selected_gate_entries_kernel<<<blocks_g, threads, 0, prep_stream>>>(
            d_G_rows[slot],
            d_G_cols[slot],
            d_G_vals[slot],
            d_G_selected_indices[slot],
            d_G_selected_count[slot],
            d_G_rows_culled[slot],
            d_G_cols_culled[slot],
            d_G_vals_culled[slot]);
        CUDA_CHECK(cudaGetLastError());
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
      finalize_workspace_bindings();
      impl->resetGeometryState();
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
      count_rows_kernel<<<blocks_cnt, threads, 0, impl->stream>>>(d_M_rows,
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
    // The seed gate initializes the accumulated matrix for this fusion block
    // and does not execute BVH update/rebuild. Count it as "skip" so
    // update+rebuild+skip matches applied gate count semantics.
    ++total_bvh_skip_count;
    if (envFlag("RT_DUMP_GATE_TRAVERSAL")) {
      GateTraversalEvent seed_ev{};
      seed_ev.gate_idx = 0;
      seed_ev.gate = gates[0];
      seed_ev.tree_row_nnz_before = running_max_row_nnz;
      seed_ev.result_row_nnz_after = running_max_row_nnz;
      seed_ev.traversal_ms = 0.0;
      seed_ev.has_traversal = false;
      last_stats.gate_traversal_events.push_back(seed_ev);
    }
    std::vector<double> gate_launch_ms;
    gate_launch_ms.reserve(max_gates);
    std::vector<int> gate_result_row_nnz_after;
    gate_result_row_nnz_after.reserve(max_gates);
    size_t effective_max_gates = max_gates;
    if (!force_full && row_nnz_limit > 0 && running_max_row_nnz >= row_nnz_limit) {
      effective_max_gates = 1;
    }
    const bool overlap_prep_with_compute = false;
    if (effective_max_gates > 1 && overlap_prep_with_compute) {
      launch_gate_generation(1, 1);
    }

    for (size_t g = 1; g < effective_max_gates; ++g) {
      current_gate_idx = g;
      const std::size_t global_gate_idx = impl->debug_block_start_gate + g;
      current_gate_is_target =
          has_target_global_gate && global_gate_idx == static_cast<std::size_t>(target_global_gate);
      current_gate_nvtx_prefix = "gate " + std::to_string(global_gate_idx);
      const double launch_before_gate = total_launch_ms;
      const bool is_diag = impl->nnz1_special && isDiagonalGate(gates[g]);
      const bool is_nnz1_gate =
          impl->nnz1_special && gateHasOneRayPerRow(gates[g]);
      const auto raygen_start = std::chrono::high_resolution_clock::now();
      const int curr_slot = static_cast<int>(g & 1ULL);
      const int next_slot = curr_slot ^ 1;
      bool use_gate_cull = false;
      bool use_procedural_raygen = true;
      int procedural_mode = 0;
      int gate_nnz = 0;
      if (!is_diag && !is_nnz1_gate) {
        if (!overlap_prep_with_compute) {
          launch_gate_generation(g, curr_slot);
        }
        CUDA_CHECK(cudaEventSynchronize(gate_ready[curr_slot]));
        if (overlap_prep_with_compute && g + 1 < effective_max_gates) {
          launch_gate_generation(g + 1, next_slot);
        }
        use_gate_cull = h_gate_use_cull[curr_slot];
        use_procedural_raygen = !use_gate_cull;
        procedural_mode = use_procedural_raygen
                              ? ((impl->nnz1_special && gateHasOneRayPerRow(gates[g])) ? 2 : 1)
                              : 0;
        gate_nnz = h_G_selected_count[curr_slot];
        if (gate_nnz <= 0) {
          if (verbose) {
            std::cerr << "[SPMSPM] gate has zero valid rays at gate " << g << std::endl;
          }
          finalize_workspace_bindings();
          impl->resetGeometryState();
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
      }
      const int row_nnz_before_gate = running_max_row_nnz;

      // Conservative pre-check: avoid entering a multiply that can push row nnz far beyond limit.
      if (!force_full && row_nnz_limit > 0 && !is_diag && !is_nnz1_gate && fused_gates_applied > 0) {
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

      const bool need_gas_refresh = !is_diag && !is_nnz1_gate;

      if (need_gas_refresh) {
        impl->state.sphere_size = M_nnz;
        if (sync_stage_timing) {
          cudaEvent_t geom_start = nullptr;
          cudaEvent_t geom_stop = nullptr;
          CUDA_CHECK(cudaEventCreate(&geom_start));
          CUDA_CHECK(cudaEventCreate(&geom_stop));
          CUDA_CHECK(cudaEventRecord(geom_start, impl->stream));
          impl->ensurePrimitiveBuffers(M_nnz);
          const int blocks_sphere = static_cast<int>((M_nnz + threads - 1) / threads);
          if (impl->primitive_type == RTPrimitiveType::Sphere) {
            coo_to_sphere_kernel<<<blocks_sphere, threads, 0, impl->stream>>>(d_M_rows,
                                                                               d_M_cols,
                                                                               impl->state.spherePoints,
                                                                               impl->state.sphereRadius,
                                                                               static_cast<int>(M_nnz));
          } else {
            coo_to_triangle_vertices_kernel<<<blocks_sphere, threads, 0, impl->stream>>>(d_M_rows,
                                                                                           d_M_cols,
                                                                                           impl->state.triangleVertices,
                                                                                           static_cast<int>(M_nnz));
          }
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
          impl->ensurePrimitiveBuffers(M_nnz);
          const int blocks_sphere = static_cast<int>((M_nnz + threads - 1) / threads);
          if (impl->primitive_type == RTPrimitiveType::Sphere) {
            coo_to_sphere_kernel<<<blocks_sphere, threads, 0, impl->stream>>>(d_M_rows,
                                                                               d_M_cols,
                                                                               impl->state.spherePoints,
                                                                               impl->state.sphereRadius,
                                                                               static_cast<int>(M_nnz));
          } else {
            coo_to_triangle_vertices_kernel<<<blocks_sphere, threads, 0, impl->stream>>>(d_M_rows,
                                                                                           d_M_cols,
                                                                                           impl->state.triangleVertices,
                                                                                           static_cast<int>(M_nnz));
          }
          CUDA_CHECK(cudaGetLastError());
          if (verbose) {
            CUDA_CHECK(cudaStreamSynchronize(impl->stream));
          }
          const auto geom_stop = std::chrono::high_resolution_clock::now();
          total_gas_ms += std::chrono::duration<double, std::milli>(geom_stop - geom_start).count();
        }
      }
      if (is_diag) {
          // Diagonal gate fast path:
          // skip RT traversal and multiply accumulated values directly by per-row
          // diagonal factors (left multiply: M' x M).
          ++total_bvh_skip_count;
          ensure_next_capacity(M_nnz);
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
          const int blocks_diag = static_cast<int>((M_nnz + threads - 1) / threads);
          if (sync_stage_timing) {
            cudaEvent_t diag_start = nullptr;
            cudaEvent_t diag_stop = nullptr;
            CUDA_CHECK(cudaEventCreate(&diag_start));
            CUDA_CHECK(cudaEventCreate(&diag_stop));
            CUDA_CHECK(cudaEventRecord(diag_start, impl->stream));
            apply_left_diagonal_gate_kernel<<<blocks_diag, threads, 0, impl->stream>>>(
                d_M_rows,
                d_M_vals,
                d_N_vals,
                static_cast<int>(M_nnz),
                gates[g]);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(diag_stop, impl->stream));
            CUDA_CHECK(cudaEventSynchronize(diag_stop));
            float diag_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&diag_ms, diag_start, diag_stop));
            total_nnz1_mul_ms += diag_ms;
            CUDA_CHECK(cudaEventDestroy(diag_start));
            CUDA_CHECK(cudaEventDestroy(diag_stop));
          } else {
            const auto diag_start = std::chrono::high_resolution_clock::now();
            apply_left_diagonal_gate_kernel<<<blocks_diag, threads, 0, impl->stream>>>(
                d_M_rows,
                d_M_vals,
                d_N_vals,
                static_cast<int>(M_nnz),
                gates[g]);
            CUDA_CHECK(cudaGetLastError());
            if (verbose) {
              CUDA_CHECK(cudaStreamSynchronize(impl->stream));
            }
            const auto diag_stop = std::chrono::high_resolution_clock::now();
            total_nnz1_mul_ms +=
                std::chrono::duration<double, std::milli>(diag_stop - diag_start).count();
          }
          impl->num_rays = M_nnz;
          impl->state.d_size = static_cast<uint64_t>(M_nnz);
          impl->state.m_result_dim = make_int2(static_cast<int>(nDim), static_cast<int>(nDim));
          impl->state.procedural_raygen_mode = procedural_mode;
          impl->state.current_gate = gates[g];
          impl->state.d_ray_rows = use_procedural_raygen ? nullptr
                                                         : (use_gate_cull ? d_G_rows_culled[curr_slot] : d_G_rows[curr_slot]);
          impl->state.d_ray_cols = use_procedural_raygen ? nullptr
                                                         : (use_gate_cull ? d_G_cols_culled[curr_slot] : d_G_cols[curr_slot]);
          impl->state.d_ray_vals = use_procedural_raygen ? nullptr
                                                         : (use_gate_cull ? d_G_vals_culled[curr_slot] : d_G_vals[curr_slot]);
          impl->state.sphereValues = d_M_vals;
          impl->state.primitiveCols = d_M_cols;
          impl->state.d_result = d_N_vals;
          impl->state.d_result_buf_size = M_nnz * sizeof(bqsim_rt::Complex);
      } else if (is_nnz1_gate) {
          // 1-ray-per-row gate fast path:
          // bypass RT and apply row remap + scaling directly on COO.
          cudaEvent_t nnz1_start_ev = nullptr;
          cudaEvent_t nnz1_stop_ev = nullptr;
          if (sync_stage_timing) {
            CUDA_CHECK(cudaEventCreate(&nnz1_start_ev));
            CUDA_CHECK(cudaEventCreate(&nnz1_stop_ev));
            CUDA_CHECK(cudaEventRecord(nnz1_start_ev, impl->stream));
          }
          ++total_bvh_skip_count;
          ensure_next_capacity(M_nnz);
          ensure_inverse_map_capacity(nDim);
          CUDA_CHECK(cudaMemsetAsync(impl->d_inv_rows, 0xFF, nDim * sizeof(int), impl->stream));

          const int blocks_map = static_cast<int>((nDim + threads - 1) / threads);
          build_inverse_rowmap_for_nnz1_gate_kernel<<<blocks_map, threads, 0, impl->stream>>>(
              gates[g],
              static_cast<int>(nDim),
              impl->d_inv_rows,
              impl->d_inv_scales);
          CUDA_CHECK(cudaGetLastError());

          const int blocks_apply = static_cast<int>((M_nnz + threads - 1) / threads);
          apply_nnz1_gate_via_rowmap_kernel<<<blocks_apply, threads, 0, impl->stream>>>(
              d_M_rows,
              d_M_cols,
              d_M_vals,
              static_cast<int>(M_nnz),
              impl->d_inv_rows,
              impl->d_inv_scales,
              d_N_rows,
              d_N_cols,
              d_N_vals);
          CUDA_CHECK(cudaGetLastError());

          CUDA_CHECK(cudaMemsetAsync(impl->state.d_row_counts, 0, nDim * sizeof(int), impl->stream));
          count_rows_kernel<<<blocks_apply, threads, 0, impl->stream>>>(
              d_N_rows,
              static_cast<int>(M_nnz),
              impl->state.d_row_counts);
          CUDA_CHECK(cudaGetLastError());

          if (sync_stage_timing) {
            CUDA_CHECK(cudaEventRecord(nnz1_stop_ev, impl->stream));
            CUDA_CHECK(cudaEventSynchronize(nnz1_stop_ev));
            float nnz1_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&nnz1_ms, nnz1_start_ev, nnz1_stop_ev));
            total_nnz1_mul_ms += nnz1_ms;
            CUDA_CHECK(cudaEventDestroy(nnz1_start_ev));
            CUDA_CHECK(cudaEventDestroy(nnz1_stop_ev));
          } else if (verbose) {
            CUDA_CHECK(cudaStreamSynchronize(impl->stream));
          }

          impl->num_rays = M_nnz;
          impl->state.d_size = static_cast<uint64_t>(M_nnz);
          impl->state.m_result_dim = make_int2(static_cast<int>(nDim), static_cast<int>(nDim));
          impl->state.procedural_raygen_mode = 0;
          impl->state.current_gate = gates[g];
          impl->state.d_ray_rows = nullptr;
          impl->state.d_ray_cols = nullptr;
          impl->state.d_ray_vals = nullptr;
          impl->state.sphereValues = d_M_vals;
          impl->state.primitiveCols = d_M_cols;
          impl->state.d_result = d_N_vals;
          impl->state.d_result_buf_size = M_nnz * sizeof(bqsim_rt::Complex);
      } else {
          const auto loop_overhead_start = std::chrono::high_resolution_clock::now();
          impl->num_rays = static_cast<size_t>(gate_nnz);
          impl->state.d_size = static_cast<uint64_t>(gate_nnz);
          impl->state.m_result_dim = make_int2(static_cast<int>(nDim), static_cast<int>(nDim));
          impl->state.procedural_raygen_mode = procedural_mode;
          impl->state.current_gate = gates[g];
          impl->state.d_ray_rows = use_procedural_raygen ? nullptr
                                                         : (use_gate_cull ? d_G_rows_culled[curr_slot] : d_G_rows[curr_slot]);
          impl->state.d_ray_cols = use_procedural_raygen ? nullptr
                                                         : (use_gate_cull ? d_G_cols_culled[curr_slot] : d_G_cols[curr_slot]);
          impl->state.d_ray_vals = use_procedural_raygen ? nullptr
                                                         : (use_gate_cull ? d_G_vals_culled[curr_slot] : d_G_vals[curr_slot]);
          impl->state.sphereValues = d_M_vals;
          impl->state.primitiveCols = d_M_cols;
          // Any-hit in-launch merge:
          // aggregate contributions directly into row-local slots, then compact once.
          const int gate_row_nnz_ub = gateRowNNZUpperBound(gates[g]);
          const int row2_predicted_upper = row_nnz_before_gate * gate_row_nnz_ub;
          const bool use_row2_fast =
              impl->ell2_fast_path &&
              row_nnz_before_gate > 0 &&
              row2_predicted_upper <= 2;

          const std::size_t row_cap = static_cast<std::size_t>(std::max(row_nnz_limit, 1));
          int row_capacity = use_row2_fast
                                 ? 2
                                 : static_cast<int>(std::max<std::size_t>(
                                       1,
                                       std::max<std::size_t>(row_cap, static_cast<std::size_t>(impl->atomic_row_capacity)) * 2ULL));
          bool launch_need_gas_refresh = need_gas_refresh;
          bool merge_completed = false;
          bool overhead_accounted = false;
          int unique = 0;
          const std::string merge_nvtx_label = current_gate_nvtx_prefix + " merge";
          while (!merge_completed) {
            const std::size_t slot_capacity =
                std::max<std::size_t>(1, nDim * static_cast<std::size_t>(row_capacity));
            ensure_next_capacity(slot_capacity);
            if (impl->atomic_row_offset_capacity < nDim) {
              if (impl->d_atomic_row_offsets) {
                safeCudaFree(impl->d_atomic_row_offsets, "cudaFree(d_atomic_row_offsets)");
              }
              CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->d_atomic_row_offsets),
                                    nDim * sizeof(int)));
              impl->atomic_row_offset_capacity = nDim;
            }
            if (impl->atomic_slot_capacity < slot_capacity) {
              if (impl->d_atomic_slot_cols) {
                safeCudaFree(impl->d_atomic_slot_cols, "cudaFree(d_atomic_slot_cols)");
              }
              if (impl->d_atomic_slot_vals) {
                safeCudaFree(impl->d_atomic_slot_vals, "cudaFree(d_atomic_slot_vals)");
              }
              CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->d_atomic_slot_cols),
                                    slot_capacity * sizeof(int)));
              CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->d_atomic_slot_vals),
                                    slot_capacity * sizeof(bqsim_rt::Complex)));
              impl->atomic_slot_capacity = slot_capacity;
            }
            CUDA_CHECK(cudaMemsetAsync(impl->state.d_row_counts, 0, nDim * sizeof(int), impl->stream));
            CUDA_CHECK(cudaMemsetAsync(impl->d_atomic_slot_cols,
                                       0xFF,
                                       slot_capacity * sizeof(int),
                                       impl->stream));
            CUDA_CHECK(cudaMemsetAsync(impl->d_atomic_slot_vals,
                                       0,
                                       slot_capacity * sizeof(bqsim_rt::Complex),
                                       impl->stream));
            CUDA_CHECK(cudaMemsetAsync(impl->state.d_out_count, 0, sizeof(int), impl->stream));

            impl->state.d_out_rows = impl->state.d_row_counts; // row-wise unique counters
            impl->state.d_out_cols = impl->d_atomic_slot_cols; // per-row slot col keys
            impl->state.d_out_vals = impl->d_atomic_slot_vals; // per-row slot accumulators
            impl->state.out_capacity = static_cast<uint64_t>(slot_capacity);
            impl->state.d_result = d_N_vals;
            impl->state.d_result_buf_size = slot_capacity * sizeof(bqsim_rt::Complex);
            impl->state.rt_mode = 1;

            if (!overhead_accounted) {
              const auto num_start = std::chrono::high_resolution_clock::now();
              total_overhead_ms += std::chrono::duration<double, std::milli>(num_start - loop_overhead_start).count();
              overhead_accounted = true;
            }
            launch_optix(static_cast<size_t>(gate_nnz), launch_need_gas_refresh, false);
            launch_need_gas_refresh = false;

            int overflow = 0;
            CUDA_CHECK(cudaMemcpyAsync(&overflow,
                                       impl->state.d_out_count,
                                       sizeof(int),
                                       cudaMemcpyDeviceToHost,
                                       impl->stream));
            CUDA_CHECK(cudaStreamSynchronize(impl->stream));
            finalize_pending_launch();
            if (overflow != 0) {
              const int next_row_capacity =
                  (row_capacity >= std::numeric_limits<int>::max() / 2)
                      ? std::numeric_limits<int>::max()
                      : row_capacity * 2;
              if (next_row_capacity <= row_capacity ||
                  static_cast<std::size_t>(row_capacity) >= nDim) {
                std::cerr << "[SPMSPM] anyhit in-launch merge overflow at gate " << g
                          << " (row_capacity=" << row_capacity
                          << "). Automatic growth exhausted." << std::endl;
                finalize_workspace_bindings();
                impl->resetGeometryState();
                return false;
              }
              if (verbose) {
                std::cout << "[SPMSPM] anyhit in-launch merge overflow at gate " << g
                          << " (row_capacity=" << row_capacity
                          << "); retrying with row_capacity=" << next_row_capacity
                          << std::endl;
              }
              row_capacity = next_row_capacity;
              impl->atomic_row_capacity =
                  std::max(impl->atomic_row_capacity, std::max(1, row_capacity / 2));
              continue;
            }

            push_nvtx(merge_nvtx_label, current_gate_is_target);
            const auto merge_wall_start = std::chrono::high_resolution_clock::now();
            auto exec = thrust::cuda::par.on(impl->stream);
            auto row_counts_ptr = thrust::device_pointer_cast(impl->state.d_row_counts);
            unique = static_cast<int>(thrust::reduce(exec,
                                                     row_counts_ptr,
                                                     row_counts_ptr + nDim,
                                                     0));
            thrust::exclusive_scan(exec,
                                   row_counts_ptr,
                                   row_counts_ptr + nDim,
                                   thrust::device_pointer_cast(impl->d_atomic_row_offsets));
            ensure_next_capacity(static_cast<std::size_t>(unique));
            if (use_row2_fast && row_capacity == 2) {
              const int blocks_row = static_cast<int>((nDim + threads - 1) / threads);
              compact_atomic_slots_row2_kernel<<<blocks_row, threads, 0, impl->stream>>>(
                  impl->d_atomic_slot_cols,
                  impl->d_atomic_slot_vals,
                  impl->d_atomic_row_offsets,
                  static_cast<int>(nDim),
                  d_N_rows,
                  d_N_cols,
                  d_N_vals);
            } else {
              const int total_slots = static_cast<int>(slot_capacity);
              const int blocks_compact = static_cast<int>((total_slots + threads - 1) / threads);
              compact_atomic_slots_kernel<<<blocks_compact, threads, 0, impl->stream>>>(
                  impl->d_atomic_slot_cols,
                  impl->d_atomic_slot_vals,
                  impl->d_atomic_row_offsets,
                  static_cast<int>(nDim),
                  row_capacity,
                  d_N_rows,
                  d_N_cols,
                  d_N_vals);
            }
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(impl->stream));
            const auto merge_wall_stop = std::chrono::high_resolution_clock::now();
            total_compact_ms += std::chrono::duration<double, std::milli>(merge_wall_stop - merge_wall_start).count();
            pop_nvtx(current_gate_is_target);
            merge_completed = true;
          }
          if (unique <= 0) {
            std::cerr << "[SPMSPM] in-launch merge produced no entries at gate " << g << std::endl;
            finalize_workspace_bindings();
            impl->resetGeometryState();
            return false;
          }
          impl->num_rays = static_cast<std::size_t>(unique);
          impl->state.d_size = impl->num_rays;
      } // else is_diag

      const auto loop_tail_start = std::chrono::high_resolution_clock::now();

      std::swap(d_M_rows, d_N_rows);
      std::swap(d_M_cols, d_N_cols);
      std::swap(d_M_vals, d_N_vals);
      std::swap(M_capacity, N_capacity);
      std::swap(M_slot, N_slot);
      M_nnz = impl->num_rays;
      impl->state.sphereValues = d_M_vals;
      impl->state.sphere_size = M_nnz;
      impl->state.d_out_rows = nullptr;
      impl->state.d_out_cols = nullptr;
      impl->state.d_out_vals = nullptr;
      impl->state.out_capacity = 0;

      ++fused_gates_applied;
      impl->last_fused_gates = fused_gates_applied;
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
      if (envFlag("RT_DUMP_GATE_TRAVERSAL")) {
        GateTraversalEvent gate_ev{};
        gate_ev.gate_idx = g;
        gate_ev.traversal_sample_idx = gate_launch_sample_count;
        gate_ev.gate = gates[g];
        gate_ev.tree_row_nnz_before = row_nnz_before_gate;
        gate_ev.result_row_nnz_after = running_max_row_nnz;
        gate_ev.traversal_ms = is_diag ? 0.0 : gate_traversal_ms;
        gate_ev.has_traversal = !is_diag;
        last_stats.gate_traversal_events.push_back(gate_ev);
      }
      ++gate_launch_sample_count;
      if (stop_after_this_gate) {
        const auto loop_tail_stop = std::chrono::high_resolution_clock::now();
        total_overhead_ms +=
            std::chrono::duration<double, std::milli>(loop_tail_stop - loop_tail_start).count();
        break;
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

    impl->max_row_nnz = std::max(1, running_max_row_nnz);
    const auto cleanup_start = std::chrono::high_resolution_clock::now();

    impl->num_rays = M_nnz;
    impl->state.d_ray_rows = d_M_rows;
    impl->state.d_ray_cols = d_M_cols;
    impl->state.d_result = d_M_vals;
    impl->state.d_size = impl->num_rays;
    impl->state.d_result_buf_size = impl->num_rays * sizeof(bqsim_rt::Complex);
    impl->state.sphereValues = d_M_vals;
    impl->state.primitiveCols = d_M_cols;
    impl->state.sphere_size = M_nnz;
    impl->state.m_result_dim = make_int2(static_cast<int>(nDim), static_cast<int>(nDim));
    impl->precomputed_result = true;
    finalize_pending_launch();

    impl->state.d_ray_vals = nullptr;
    finalize_workspace_bindings();
    const auto cleanup_stop = std::chrono::high_resolution_clock::now();
    total_cleanup_ms +=
        std::chrono::duration<double, std::milli>(cleanup_stop - cleanup_start).count();
    if (collect_breakdown) {
      last_stats.geom_ms = total_geom_ms;
      last_stats.gas_ms = total_gas_ms;
      last_stats.launch_ms = total_launch_ms;
      last_stats.ray_gen_ms = total_ray_gen_ms;
      last_stats.diagonal_ms = total_nnz1_mul_ms;
      last_stats.compact_ms = total_compact_ms;
      last_stats.overhead_ms = total_overhead_ms;
      last_stats.cleanup_ms = total_cleanup_ms;
      last_stats.compute_ms = total_launch_ms;
      last_stats.bvh_rebuild_count = total_bvh_rebuild_count;
      last_stats.bvh_update_count = total_bvh_update_count;
      last_stats.bvh_skip_count = total_bvh_skip_count;
    }
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[SPMSPM] Exception: " << e.what() << std::endl;
    impl->resetGeometryState();
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

  try {
    auto ell_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemsetAsync(values,
                               0,
                               sizeof(bqsim_rt::Complex) * nDim * num_non_zeros,
                               impl->stream));
    CUDA_CHECK(cudaMemsetAsync(indices,
                               0,
                               sizeof(int) * nDim * num_non_zeros,
                               impl->stream));

    impl->ensureEllRowCountCapacity(nDim);
    CUDA_CHECK(cudaMemsetAsync(impl->d_ell_row_counts, 0, nDim * sizeof(int), impl->stream));

    int threads = 256;
    int blocks = static_cast<int>((impl->num_rays + threads - 1) / threads);
    coo_to_ell_kernel<<<blocks, threads, 0, impl->stream>>>(impl->state.d_ray_rows,
                                                            impl->state.d_ray_cols,
                                                            impl->state.d_result,
                                                            static_cast<int>(impl->num_rays),
                                                            num_non_zeros,
                                                            static_cast<int>(nDim),
                                                            values,
                                                            indices,
                                                            impl->d_ell_row_counts);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(impl->stream));

    auto ell_stop = std::chrono::high_resolution_clock::now();
    if (envFlagDefaultTrue("BQSIM_ENABLE_BREAKDOWN")) {
      last_stats.ell_ms = std::chrono::duration<double, std::milli>(ell_stop - ell_start).count();
    }
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

int RTSpMSpMEngine::maxRowNNZ() const {
  return impl ? impl->max_row_nnz : 0;
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

const char* RTSpMSpMEngine::primitiveTypeName() const {
  if (!impl) {
    return "triangle";
  }
  return impl->primitive_type == RTPrimitiveType::Sphere ? "sphere" : "triangle";
}

std::size_t RTSpMSpMEngine::lastFusedGateCount() const {
  return impl ? impl->last_fused_gates : 0;
}
