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

bool RTSpMSpMEngine::prepareGeometry(const qc::FusedGate&,
                                     dd::Package<dd::DDPackageConfig>*,
                                     int,
                                     std::size_t) {
  last_stats = {};
  return false;
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

bool RTSpMSpMEngine::lastReachedDensity() const {
  return false;
}

bool RTSpMSpMEngine::launchRTMultiply() {
  return false;
}

bool RTSpMSpMEngine::collectResultToELL(cuComplex*,
                                        int*,
                                        int,
                                        std::size_t) {
  return false;
}

double RTSpMSpMEngine::densityEstimate() const {
  return 0.0;
}

int RTSpMSpMEngine::maxRowNNZ() const {
  return 0;
}

int RTSpMSpMEngine::ellWidthHint(int fallback) const {
  return fallback;
}

bool RTSpMSpMEngine::useDenseMV() const {
  return false;
}

std::size_t RTSpMSpMEngine::lastFusedGateCount() const {
  return 0;
}

bool RTSpMSpMEngine::lastReachedDensity() const {
  return false;
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
#else

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "operations/OpType.hpp"
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <cstring>

#include "optixSpMSpM.h"
#include "dd/Package.hpp"
#include "dd/RealNumber.hpp"
#include "FusedGate.hpp"
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

double envDouble(const char* name, double fallback) {
  const char* value = std::getenv(name);
  if (!value) {
    return fallback;
  }
  char* end = nullptr;
  const double parsed = std::strtod(value, &end);
  if (end == value) {
    return fallback;
  }
  return parsed;
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

std::string loadPtxFromFile(const std::string& path) {
  std::ifstream file(path.c_str(), std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open PTX file: " + path);
  }
  std::ostringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

struct DDNnz {
  int row;
  int col;
  cuComplex val;
};

struct ComplexAdd {
  __host__ __device__ cuComplex operator()(const cuComplex& a,
                                                const cuComplex& b) const {
    return cuCaddf(a, b);
  }
};

constexpr int kMaxDecodedMacs = 50;
constexpr int kMaxLev = 40;

inline cuComplex mulComplex(const cuComplex& a, const cuComplex& b) {
  return make_cuComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

void collectFromDD(const dd::mEdge& e,
                   std::size_t row_base,
                   std::size_t col_base,
                   cuComplex factor,
                   std::vector<DDNnz>& out) {
  if (e.w.exactlyZero()) {
    return;
  }

  const auto ar = dd::RealNumber::val(e.w.r);
  const auto ai = dd::RealNumber::val(e.w.i);
  cuComplex next_factor = mulComplex(factor, make_cuComplex(static_cast<float>(ar), static_cast<float>(ai)));

  if (e.isTerminal()) {
    out.push_back({static_cast<int>(row_base), static_cast<int>(col_base), next_factor});
    return;
  }

  const std::size_t half = 1ULL << e.p->v;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      const dd::mEdge& child = e.p->e[i * 2 + j];
      if (child.w.exactlyZero()) {
        continue;
      }
      collectFromDD(child,
                    row_base + static_cast<std::size_t>(i) * half,
                    col_base + static_cast<std::size_t>(j) * half,
                    next_factor,
                    out);
    }
  }
}

__global__ void dd_to_ell_kernel(const dd::GPU_DD_edge* dd_edges,
                                 const dd::GPU_DD_node* dd_nodes,
                                 int* out_cols,
                                 cuComplex* out_vals,
                                 int num_non_zeros,
                                 int num_qubits) {
  __shared__ int decoded_locs[kMaxDecodedMacs];
  __shared__ cuComplex decoded_factors[kMaxDecodedMacs];
  __shared__ uint8_t left_or_right[kMaxLev];
  __shared__ bool up_or_down[kMaxLev];
  __shared__ int decode_ptr;
  __shared__ int edge_stack[kMaxLev];

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (tid < num_qubits) {
    left_or_right[tid] = 0;
    up_or_down[num_qubits - 1 - tid] = (row & (1 << tid)) != 0;
  }
  __syncthreads();

  if (tid == 0) {
    int edge_ptr = 0;
    int node_ptr = 0;
    int stack_ptr = 0;
    decode_ptr = 0;
    edge_stack[stack_ptr] = 0;
    cuComplex rec_factor = make_cuComplex(1.0, 0.0);
    int rec_loc = 0;
    while (stack_ptr >= 0) {
      if (decode_ptr == num_non_zeros) {
        break;
      }
      edge_ptr = edge_stack[stack_ptr];
      if (edge_ptr == dd::const_zero_edge) {
        stack_ptr--;
        continue;
      }
      node_ptr = dd_edges[edge_ptr].DD_node_ptr;
      if (node_ptr == dd::const_one_node) {
        decoded_locs[decode_ptr] = rec_loc;
        decoded_factors[decode_ptr] = cuCmulf(rec_factor, dd_edges[edge_ptr].w);
        stack_ptr--;
        decode_ptr++;
        continue;
      }

      int child_idx = static_cast<int>(left_or_right[stack_ptr]) +
                      static_cast<int>(up_or_down[stack_ptr]) * 2;
      if (left_or_right[stack_ptr] == 2) {
        left_or_right[stack_ptr] = 0;
        rec_factor = cuCdivf(rec_factor, dd_edges[edge_ptr].w);
        rec_loc -= (1 << dd_nodes[node_ptr].qubit);
        stack_ptr--;
      } else {
        left_or_right[stack_ptr]++;
        if (left_or_right[stack_ptr] == 1) {
          rec_factor = cuCmulf(rec_factor, dd_edges[edge_ptr].w);
        }
        rec_loc += (1 << dd_nodes[node_ptr].qubit) *
                   static_cast<int>(left_or_right[stack_ptr] - 1);
        stack_ptr++;
        edge_stack[stack_ptr] = dd_nodes[node_ptr].outgoing_DD_edge_ptr[child_idx];
      }
    }
  }

  __syncthreads();
  if (tid < num_non_zeros) {
    const size_t idx = static_cast<size_t>(row) * num_non_zeros + tid;
    if (tid < decode_ptr) {
      out_cols[idx] = decoded_locs[tid];
      out_vals[idx] = decoded_factors[tid];
    } else {
      out_cols[idx] = -1;
      out_vals[idx] = make_cuComplex(0.0, 0.0);
    }
  }
}

__global__ void mark_valid_kernel(const int* cols, int* mask, size_t total_entries) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_entries) {
    return;
  }
  mask[idx] = (cols[idx] >= 0) ? 1 : 0;
}

__global__ void compact_aabb_kernel(const int* cols,
                                    const cuComplex* vals,
                                    const int* mask,
                                    const int* prefix,
                                    size_t total_entries,
                                    int num_non_zeros,
                                    int* out_rows,
                                    int* out_cols,
                                    cuComplex* out_ray_vals,
                                    cuComplex* out_aabb_vals,
                                    OptixAabb* out_aabbs) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_entries) {
    return;
  }
  if (mask[idx] == 0) {
    return;
  }
  const int out_idx = prefix[idx];
  const int row = static_cast<int>(idx / static_cast<size_t>(num_non_zeros));
  const int col = cols[idx];
  out_rows[out_idx] = row;
  out_cols[out_idx] = col;
  out_ray_vals[out_idx] = make_cuComplex(1.0, 0.0);
  out_aabb_vals[out_idx] = vals[idx];
  OptixAabb aabb{};
  aabb.minX = static_cast<float>(col);
  aabb.maxX = static_cast<float>(col + 1);
  aabb.minY = static_cast<float>(row);
  aabb.maxY = static_cast<float>(row + 1);
  aabb.minZ = 0.0f;
  aabb.maxZ = 1.0f;
  out_aabbs[out_idx] = aabb;
}

__global__ void coo_to_aabb_kernel(const int* rows,
                                   const int* cols,
                                   OptixAabb* out_aabbs,
                                   int nnz) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nnz) {
    return;
  }
  const int row = rows[tid];
  const int col = cols[tid];
  OptixAabb aabb{};
  aabb.minX = static_cast<float>(col);
  aabb.maxX = static_cast<float>(col + 1);
  aabb.minY = static_cast<float>(row);
  aabb.maxY = static_cast<float>(row + 1);
  aabb.minZ = 0.0f;
  aabb.maxZ = 1.0f;
  out_aabbs[tid] = aabb;
}

__global__ void build_gate_coo_kernel(const qc::GatePrimitive* gates,
                                      int gate_idx,
                                      int nDim,
                                      int* rows,
                                      int* cols,
                                      cuComplex* vals) {
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
  cuComplex v0{};
  cuComplex v1{};

  if (!controls_ok) {
    v0 = make_cuComplex(1.0, 0.0);
    v1 = make_cuComplex(0.0, 0.0);
  } else {
    const int m = bit * 2;
    const float2 a0 = gate.matrix[m];
    const float2 a1 = gate.matrix[m + 1];
    v0 = make_cuComplex(a0.x, a0.y);
    v1 = make_cuComplex(a1.x, a1.y);
  }

  const int idx = row * 2;
  rows[idx] = row;
  cols[idx] = controls_ok ? col0 : row;
  vals[idx] = v0;
  rows[idx + 1] = row;
  cols[idx + 1] = controls_ok ? col1 : row;
  vals[idx + 1] = v1;
}

__global__ void bfs_symbolic_kernel(const dd::GPU_DD_edge* dd_edges,
                                    const dd::GPU_DD_node* dd_nodes,
                                    const int* in_edge,
                                    const int* in_row,
                                    const int* in_col,
                                    int in_count,
                                    int* out_edge,
                                    int* out_row,
                                    int* out_col,
                                    int* out_count,
                                    int* row_counts,
                                    unsigned long long* total_nnz,
                                    int max_tasks,
                                    int* overflow) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= in_count) {
    return;
  }

  const int edge_idx = in_edge[tid];
  if (edge_idx == dd::const_zero_edge) {
    return;
  }
  const cuComplex w = dd_edges[edge_idx].w;
  if (w.x == 0.0 && w.y == 0.0) {
    return;
  }

  const int node_ptr = dd_edges[edge_idx].DD_node_ptr;
  if (node_ptr == dd::const_one_node) {
    const int row = in_row[tid];
    atomicAdd(&row_counts[row], 1);
    atomicAdd(total_nnz, 1ULL);
    return;
  }

  const int half = 1 << dd_nodes[node_ptr].qubit;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      const int child_edge = dd_nodes[node_ptr].outgoing_DD_edge_ptr[i * 2 + j];
      if (child_edge == dd::const_zero_edge) {
        continue;
      }
      const cuComplex cw = dd_edges[child_edge].w;
      if (cw.x == 0.0 && cw.y == 0.0) {
        continue;
      }
      const int next_idx = atomicAdd(out_count, 1);
      if (next_idx >= max_tasks) {
        *overflow = 1;
        return;
      }
      out_edge[next_idx] = child_edge;
      out_row[next_idx] = in_row[tid] + i * half;
      out_col[next_idx] = in_col[tid] + j * half;
    }
  }
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

__global__ void bfs_expand_kernel(const dd::GPU_DD_edge* dd_edges,
                                  const dd::GPU_DD_node* dd_nodes,
                                  const int* in_edge,
                                  const int* in_row,
                                  const int* in_col,
                                  const cuComplex* in_factor,
                                  int in_count,
                                  int* out_edge,
                                  int* out_row,
                                  int* out_col,
                                  cuComplex* out_factor,
                                  int* out_count,
                                  int* out_rows,
                                  int* out_cols,
                                  cuComplex* out_ray_vals,
                                  cuComplex* out_vals,
                                  OptixAabb* out_aabbs,
                                  int* out_total,
                                  int max_tasks,
                                  int max_out,
                                  int* overflow) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= in_count) {
    return;
  }

  const int edge_idx = in_edge[tid];
  if (edge_idx == dd::const_zero_edge) {
    return;
  }
  const cuComplex w = dd_edges[edge_idx].w;
  if (w.x == 0.0 && w.y == 0.0) {
    return;
  }

  const cuComplex next_factor = cuCmulf(in_factor[tid], w);
  const int node_ptr = dd_edges[edge_idx].DD_node_ptr;
  if (node_ptr == dd::const_one_node) {
    const int out_idx = atomicAdd(out_total, 1);
    if (out_idx >= max_out) {
      *overflow = 1;
      return;
    }
    const int row = in_row[tid];
    const int col = in_col[tid];
    out_rows[out_idx] = row;
    out_cols[out_idx] = col;
    out_ray_vals[out_idx] = make_cuComplex(1.0, 0.0);
    out_vals[out_idx] = next_factor;
    OptixAabb aabb{};
    aabb.minX = static_cast<float>(col);
    aabb.maxX = static_cast<float>(col + 1);
    aabb.minY = static_cast<float>(row);
    aabb.maxY = static_cast<float>(row + 1);
    aabb.minZ = 0.0f;
    aabb.maxZ = 1.0f;
    out_aabbs[out_idx] = aabb;
    return;
  }

  const int half = 1 << dd_nodes[node_ptr].qubit;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      const int child_edge = dd_nodes[node_ptr].outgoing_DD_edge_ptr[i * 2 + j];
      if (child_edge == dd::const_zero_edge) {
        continue;
      }
      const cuComplex cw = dd_edges[child_edge].w;
      if (cw.x == 0.0 && cw.y == 0.0) {
        continue;
      }
      const int next_idx = atomicAdd(out_count, 1);
      if (next_idx >= max_tasks) {
        *overflow = 1;
        return;
      }
      out_edge[next_idx] = child_edge;
      out_row[next_idx] = in_row[tid] + i * half;
      out_col[next_idx] = in_col[tid] + j * half;
      out_factor[next_idx] = next_factor;
    }
  }
}

__global__ void init_identity_kernel(int nDim,
                                     int* rows,
                                     int* cols,
                                     cuComplex* vals) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nDim) {
    return;
  }
  rows[tid] = tid;
  cols[tid] = tid;
  vals[tid] = make_cuComplex(1.0, 0.0);
}

__device__ __forceinline__ cuComplex gateLoad(const float2& v) {
  return make_cuComplex(v.x, v.y);
}

__global__ void apply_gate_kernel(const qc::GatePrimitive* gates,
                                  int gate_idx,
                                  const int* in_rows,
                                  const int* in_cols,
                                  const cuComplex* in_vals,
                                  int in_count,
                                  int* out_rows,
                                  int* out_cols,
                                  cuComplex* out_vals,
                                  int* out_count,
                                  int max_out,
                                  int* overflow) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= in_count) {
    return;
  }
  const cuComplex in_val = in_vals[tid];
  if (in_val.x == 0.0 && in_val.y == 0.0) {
    return;
  }
  const int row = in_rows[tid];
  const int col = in_cols[tid];
  const qc::GatePrimitive gate = gates[gate_idx];

  if (gate.target_count == 1 && gate.control_count == 0 && gate.matrix_dim == 2) {
    const int q = gate.targets[0];
    const int bit = (row >> q) & 1;
    const int row0 = row & ~(1 << q);
    const int row1 = row | (1 << q);
    const cuComplex m00 = gateLoad(gate.matrix[0]);
    const cuComplex m01 = gateLoad(gate.matrix[1]);
    const cuComplex m10 = gateLoad(gate.matrix[2]);
    const cuComplex m11 = gateLoad(gate.matrix[3]);
    cuComplex v0 = (bit == 0) ? cuCmulf(m00, in_val) : cuCmulf(m01, in_val);
    cuComplex v1 = (bit == 0) ? cuCmulf(m10, in_val) : cuCmulf(m11, in_val);
    if (v0.x != 0.0 || v0.y != 0.0) {
      const int idx = atomicAdd(out_count, 1);
      if (idx >= max_out) {
        *overflow = 1;
        return;
      }
      out_rows[idx] = row0;
      out_cols[idx] = col;
      out_vals[idx] = v0;
    }
    if (v1.x != 0.0 || v1.y != 0.0) {
      const int idx = atomicAdd(out_count, 1);
      if (idx >= max_out) {
        *overflow = 1;
        return;
      }
      out_rows[idx] = row1;
      out_cols[idx] = col;
      out_vals[idx] = v1;
    }
    return;
  }

  if (gate.control_count >= 1 && gate.target_count == 1) {
    bool ctrl_on = true;
    for (int i = 0; i < gate.control_count; ++i) {
      const int cb = (row >> gate.controls[i]) & 1;
      if (cb == 0) {
        ctrl_on = false;
        break;
      }
    }
    int out_row = row;
    cuComplex out_val = in_val;
    if (ctrl_on) {
      const int tq = gate.targets[0];
      if (gate.gate_type == qc::X) {
        out_row = row ^ (1 << tq);
      } else if (gate.gate_type == qc::Z) {
        if (((row >> tq) & 1) != 0) {
          out_val.x = -out_val.x;
          out_val.y = -out_val.y;
        }
      } else {
        const int bit = (row >> tq) & 1;
        const int row0 = row & ~(1 << tq);
        const int row1 = row | (1 << tq);
        const cuComplex m00 = gateLoad(gate.matrix[0]);
        const cuComplex m01 = gateLoad(gate.matrix[1]);
        const cuComplex m10 = gateLoad(gate.matrix[2]);
        const cuComplex m11 = gateLoad(gate.matrix[3]);
        const cuComplex v0 = (bit == 0) ? cuCmulf(m00, in_val) : cuCmulf(m01, in_val);
        const cuComplex v1 = (bit == 0) ? cuCmulf(m10, in_val) : cuCmulf(m11, in_val);
        if (v0.x != 0.0 || v0.y != 0.0) {
          const int idx = atomicAdd(out_count, 1);
          if (idx >= max_out) {
            *overflow = 1;
            return;
          }
          out_rows[idx] = row0;
          out_cols[idx] = col;
          out_vals[idx] = v0;
        }
        if (v1.x != 0.0 || v1.y != 0.0) {
          const int idx = atomicAdd(out_count, 1);
          if (idx >= max_out) {
            *overflow = 1;
            return;
          }
          out_rows[idx] = row1;
          out_cols[idx] = col;
          out_vals[idx] = v1;
        }
        return;
      }
    }
    if (out_val.x != 0.0 || out_val.y != 0.0) {
      const int idx = atomicAdd(out_count, 1);
      if (idx >= max_out) {
        *overflow = 1;
        return;
      }
      out_rows[idx] = out_row;
      out_cols[idx] = col;
      out_vals[idx] = out_val;
    }
    return;
  }

  if (gate.target_count == 2 && gate.control_count == 0 && gate.matrix_dim == 4) {
    const int q0 = gate.targets[0];
    const int q1 = gate.targets[1];
    const int b0 = (row >> q0) & 1;
    const int b1 = (row >> q1) & 1;
    const int in_idx = (b0 << 1) | b1;
    for (int out_idx = 0; out_idx < 4; ++out_idx) {
      const cuComplex m = gateLoad(gate.matrix[out_idx * 4 + in_idx]);
      if (m.x == 0.0 && m.y == 0.0) {
        continue;
      }
      const int nb0 = (out_idx >> 1) & 1;
      const int nb1 = out_idx & 1;
      int out_row = row;
      out_row = (out_row & ~(1 << q0)) | (nb0 << q0);
      out_row = (out_row & ~(1 << q1)) | (nb1 << q1);
      const cuComplex out_val = cuCmulf(m, in_val);
      if (out_val.x == 0.0 && out_val.y == 0.0) {
        continue;
      }
      const int idx = atomicAdd(out_count, 1);
      if (idx >= max_out) {
        *overflow = 1;
        return;
      }
      out_rows[idx] = out_row;
      out_cols[idx] = col;
      out_vals[idx] = out_val;
    }
    return;
  }
}

__global__ void coo_to_primitives_kernel(const int* rows,
                                         const int* cols,
                                         const cuComplex* vals,
                                         int nnz,
                                         cuComplex* ray_vals,
                                         OptixAabb* aabbs) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= nnz) {
    return;
  }
  ray_vals[tid] = make_cuComplex(1.0, 0.0);
  const int row = rows[tid];
  const int col = cols[tid];
  OptixAabb aabb{};
  aabb.minX = static_cast<float>(col);
  aabb.maxX = static_cast<float>(col + 1);
  aabb.minY = static_cast<float>(row);
  aabb.maxY = static_cast<float>(row + 1);
  aabb.minZ = 0.0f;
  aabb.maxZ = 1.0f;
  aabbs[tid] = aabb;
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

__global__ void coo_to_ell_kernel(const int* rows,
                                 const int* cols,
                                 const cuComplex* vals,
                                 int nnz,
                                 int num_mac,
                                 int nDim,
                                 cuComplex* ell_vals,
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
  bool context_ready = false;
  bool pipeline_ready = false;
  bool sbt_ready = false;
  bool gas_ready = false;
  bool use_cuda_merge = false;
  bool use_symbolic = false;
  double density_est = 0.0;
  double density_threshold = 0.02;
  int max_row_nnz = 0;
  int ell_width = 0;
  bool use_dense_mv = false;
  bool precomputed_result = false;
  int num_qubits = 0;
  int num_mac = 0;
  size_t nDim = 0;
  size_t num_rays = 0;
  size_t last_fused_gates = 0;
  bool last_reached_density = false;
  CUdeviceptr raygen_record = 0;
  CUdeviceptr miss_record = 0;
  CUdeviceptr hitgroup_record = 0;

  void resetState() {
    num_qubits = 0;
    num_mac = 0;
    nDim = 0;
    num_rays = 0;
    use_symbolic = false;
    density_est = 0.0;
    density_threshold = 0.02;
    max_row_nnz = 0;
    ell_width = 0;
    use_dense_mv = false;
    precomputed_result = false;
    last_fused_gates = 0;
    last_reached_density = false;
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
    if (state.aabbs) {
      safeCudaFree(state.aabbs, "cudaFree(state.aabbs)");
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
    state.out_capacity = 0;
    state.d_result_buf_size = 0;
    state.d_size = 0;
    state.sphere_size = 0;
    state.aabb_size = 0;
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
    sbt_ready = false;
  }

  void cleanupGas() {
    if (state.d_gas_output_buffer) {
      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_gas_output_buffer)));
      state.d_gas_output_buffer = 0;
    }
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
    state.pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

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

    // Custom AABB primitives; no builtin IS module needed.

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
    hitgroup_prog_group_desc.hitgroup.moduleIS = state.module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__is";
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

  void buildGas() {
    cleanupGas();

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    CUdeviceptr d_aabb = reinterpret_cast<CUdeviceptr>(state.aabbs);
    build_input.customPrimitiveArray.aabbBuffers = &d_aabb;
    build_input.customPrimitiveArray.numPrimitives = static_cast<unsigned int>(state.aabb_size);
    uint32_t build_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    build_input.customPrimitiveArray.flags = build_input_flags;
    build_input.customPrimitiveArray.numSbtRecords = 1;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(state.context, &accel_options, &build_input, 1, &gas_buffer_sizes));

    CUdeviceptr d_temp_buffer_gas = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));

    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size = 0;
    size_t compacted_size_offset = ((gas_buffer_sizes.outputSizeInBytes + 7ull) / 8ull) * 8ull;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),
                          compacted_size_offset + 8));

    OptixAccelEmitDesc emit_property = {};
    emit_property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_property.result = d_buffer_temp_output_gas_and_compacted_size + compacted_size_offset;

    OPTIX_CHECK(optixAccelBuild(state.context,
                                0,
                                &accel_options,
                                &build_input,
                                1,
                                d_temp_buffer_gas,
                                gas_buffer_sizes.tempSizeInBytes,
                                d_buffer_temp_output_gas_and_compacted_size,
                                gas_buffer_sizes.outputSizeInBytes,
                                &state.gas_handle,
                                &emit_property,
                                1));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));

    size_t compacted_gas_size = 0;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size,
                          reinterpret_cast<void*>(emit_property.result),
                          sizeof(size_t),
                          cudaMemcpyDeviceToHost));

    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_gas_output_buffer), compacted_gas_size));
      OPTIX_CHECK(optixAccelCompact(state.context,
                                    0,
                                    state.gas_handle,
                                    state.d_gas_output_buffer,
                                    compacted_gas_size,
                                    &state.gas_handle));
      CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_buffer_temp_output_gas_and_compacted_size)));
    } else {
      state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }

    gas_ready = true;
  }

  void buildSbt() {
    cleanupSbt();

    RayDataRec rg_sbt = {};
    rg_sbt.data.rows = state.d_ray_rows;
    rg_sbt.data.cols = state.d_ray_cols;
    rg_sbt.data.values = state.d_ray_vals;
    rg_sbt.data.size = state.d_size;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), sizeof(RayDataRec)));
    OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prog_group, &rg_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(raygen_record),
                          &rg_sbt,
                          sizeof(RayDataRec),
                          cudaMemcpyHostToDevice));

    MissSbtRecord ms_sbt = {};
    ms_sbt.data = {0.0f, 0.0f, 0.0f};
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), sizeof(MissSbtRecord)));
    OPTIX_CHECK(optixSbtRecordPackHeader(state.miss_prog_group, &ms_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(miss_record),
                          &ms_sbt,
                          sizeof(MissSbtRecord),
                          cudaMemcpyHostToDevice));

    SphereDataRec hg_sbt = {};
    hg_sbt.data.aabbs = state.aabbs;
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

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hitgroup_record), sizeof(SphereDataRec)));
    OPTIX_CHECK(optixSbtRecordPackHeader(state.hit_prog_group, &hg_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(hitgroup_record),
                          &hg_sbt,
                          sizeof(SphereDataRec),
                          cudaMemcpyHostToDevice));

    state.sbt.raygenRecord = raygen_record;
    state.sbt.missRecordBase = miss_record;
    state.sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    state.sbt.missRecordCount = 1;
    state.sbt.hitgroupRecordBase = hitgroup_record;
    state.sbt.hitgroupRecordStrideInBytes = sizeof(SphereDataRec);
    state.sbt.hitgroupRecordCount = 1;

    sbt_ready = true;
  }

  bool runCudaMerge() {
    if (!state.d_ray_rows || !state.d_ray_cols || !state.sphereValues || !state.d_result) {
      return false;
    }
    const int nnz = static_cast<int>(num_rays);
    if (nnz <= 0) {
      return false;
    }

    uint64_t* d_keys = nullptr;
    uint64_t* d_keys_out = nullptr;
    try {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_keys), nnz * sizeof(uint64_t)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_keys_out), nnz * sizeof(uint64_t)));

      const int threads = 256;
      const int blocks = static_cast<int>((nnz + threads - 1) / threads);
      make_key_kernel<<<blocks, threads>>>(state.d_ray_rows,
                                           state.d_ray_cols,
                                           d_keys,
                                           static_cast<uint64_t>(nDim),
                                           nnz);
      CUDA_CHECK(cudaGetLastError());

      auto keys_ptr = thrust::device_pointer_cast(d_keys);
      auto vals_ptr = thrust::device_pointer_cast(state.sphereValues);
      thrust::sort_by_key(thrust::device, keys_ptr, keys_ptr + nnz, vals_ptr);

      auto keys_out_ptr = thrust::device_pointer_cast(d_keys_out);
      auto vals_out_ptr = thrust::device_pointer_cast(state.d_result);
      auto end_pair = thrust::reduce_by_key(thrust::device,
                                            keys_ptr,
                                            keys_ptr + nnz,
                                            vals_ptr,
                                            keys_out_ptr,
                                            vals_out_ptr,
                                            thrust::equal_to<uint64_t>(),
                                            ComplexAdd());
      const int unique = static_cast<int>(end_pair.first - keys_out_ptr);
      if (unique <= 0) {
        CUDA_CHECK(cudaFree(d_keys_out));
        CUDA_CHECK(cudaFree(d_keys));
        return false;
      }

      const int blocks_unique = static_cast<int>((unique + threads - 1) / threads);
      unpack_key_kernel<<<blocks_unique, threads>>>(d_keys_out,
                                                    state.d_ray_rows,
                                                    state.d_ray_cols,
                                                    static_cast<uint64_t>(nDim),
                                                    unique);
      CUDA_CHECK(cudaGetLastError());

      num_rays = static_cast<size_t>(unique);
      state.d_size = num_rays;
      state.sphere_size = num_rays;
      state.aabb_size = num_rays;

      CUDA_CHECK(cudaFree(d_keys_out));
      CUDA_CHECK(cudaFree(d_keys));
      CUDA_CHECK(cudaDeviceSynchronize());
      return true;
    } catch (const std::exception&) {
      if (d_keys_out) {
        cudaFree(d_keys_out);
      }
      if (d_keys) {
        cudaFree(d_keys);
      }
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

bool RTSpMSpMEngine::prepareGeometry(const qc::FusedGate& gate,
                                     dd::Package<dd::DDPackageConfig>* dd,
                                     int num_qubits,
                                     std::size_t nDim) {
  last_stats = {};
  if (!available || dd == nullptr) {
    return false;
  }

  try {
    impl->cleanupGeometry();
    impl->resetState();

    impl->use_cuda_merge = envFlag("BQSIM_RT_CUDA_MERGE");
    impl->use_symbolic = envFlag("BQSIM_RT_FUSED_GATE_SPM") && (num_qubits >= 12);
    impl->density_threshold = envDouble("BQSIM_RT_DENSITY_TARGET", 0.02);
    impl->num_qubits = num_qubits;
    impl->nDim = nDim;
    impl->num_mac = gate.num_mac;

    auto dd_start = std::chrono::high_resolution_clock::now();
    bool gpu_success = false;

    if (gate.num_mac > 0 && num_qubits <= kMaxLev) {
      dd::GPU_DD_edge* h_edge_arr = nullptr;
      dd::GPU_DD_node* h_node_arr = nullptr;
      dd::GPU_DD_edge* d_edge_arr = nullptr;
      dd::GPU_DD_node* d_node_arr = nullptr;
      int* d_edge_curr = nullptr;
      int* d_row_curr = nullptr;
      int* d_col_curr = nullptr;
      cuComplex* d_factor_curr = nullptr;
      int* d_edge_next = nullptr;
      int* d_row_next = nullptr;
      int* d_col_next = nullptr;
      cuComplex* d_factor_next = nullptr;
      int* d_next_count = nullptr;
      int* d_out_total = nullptr;
      int* d_overflow = nullptr;
      int* d_row_counts = nullptr;
      unsigned long long* d_total_nnz = nullptr;

      auto free_temp = [&]() {
        if (d_total_nnz) {
          CUDA_CHECK(cudaFree(d_total_nnz));
        }
        if (d_row_counts) {
          CUDA_CHECK(cudaFree(d_row_counts));
        }
        if (d_overflow) {
          CUDA_CHECK(cudaFree(d_overflow));
        }
        if (d_out_total) {
          CUDA_CHECK(cudaFree(d_out_total));
        }
        if (d_next_count) {
          CUDA_CHECK(cudaFree(d_next_count));
        }
        if (d_factor_next) {
          CUDA_CHECK(cudaFree(d_factor_next));
        }
        if (d_col_next) {
          CUDA_CHECK(cudaFree(d_col_next));
        }
        if (d_row_next) {
          CUDA_CHECK(cudaFree(d_row_next));
        }
        if (d_edge_next) {
          CUDA_CHECK(cudaFree(d_edge_next));
        }
        if (d_factor_curr) {
          CUDA_CHECK(cudaFree(d_factor_curr));
        }
        if (d_col_curr) {
          CUDA_CHECK(cudaFree(d_col_curr));
        }
        if (d_row_curr) {
          CUDA_CHECK(cudaFree(d_row_curr));
        }
        if (d_edge_curr) {
          CUDA_CHECK(cudaFree(d_edge_curr));
        }
        if (d_node_arr) {
          CUDA_CHECK(cudaFree(d_node_arr));
        }
        if (d_edge_arr) {
          CUDA_CHECK(cudaFree(d_edge_arr));
        }
        if (h_node_arr) {
          CUDA_CHECK(cudaFreeHost(h_node_arr));
        }
        if (h_edge_arr) {
          CUDA_CHECK(cudaFreeHost(h_edge_arr));
        }
      };

      try {
        const size_t max_out = static_cast<size_t>(gate.num_mac) * nDim;
        if (max_out == 0) {
          return false;
        }
        const int threads = 256;

        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_edge_arr), gate.num_edges * sizeof(dd::GPU_DD_edge)));
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_node_arr), gate.num_nodes * sizeof(dd::GPU_DD_node)));
        dd->DFS_fill_gpu_structure(gate.fused_edge, h_edge_arr, h_node_arr);

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_edge_arr), gate.num_edges * sizeof(dd::GPU_DD_edge)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_node_arr), gate.num_nodes * sizeof(dd::GPU_DD_node)));
        CUDA_CHECK(cudaMemcpy(d_edge_arr, h_edge_arr, gate.num_edges * sizeof(dd::GPU_DD_edge), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_node_arr, h_node_arr, gate.num_nodes * sizeof(dd::GPU_DD_node), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->state.d_ray_rows), max_out * sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->state.d_ray_cols), max_out * sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->state.d_ray_vals), max_out * sizeof(cuComplex)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->state.sphereValues), max_out * sizeof(cuComplex)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->state.aabbs), max_out * sizeof(OptixAabb)));

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_edge_curr), max_out * sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_row_curr), max_out * sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_col_curr), max_out * sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_factor_curr), max_out * sizeof(cuComplex)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_edge_next), max_out * sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_row_next), max_out * sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_col_next), max_out * sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_factor_next), max_out * sizeof(cuComplex)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_next_count), sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out_total), sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_overflow), sizeof(int)));

        const int root_edge = 0;
        const int root_row = 0;
        const int root_col = 0;
        const cuComplex root_factor = make_cuComplex(1.0, 0.0);

        if (impl->use_symbolic) {
          CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_row_counts), nDim * sizeof(int)));
          CUDA_CHECK(cudaMemset(d_row_counts, 0, nDim * sizeof(int)));
          CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_total_nnz), sizeof(unsigned long long)));
          CUDA_CHECK(cudaMemset(d_total_nnz, 0, sizeof(unsigned long long)));

          CUDA_CHECK(cudaMemcpy(d_edge_curr, &root_edge, sizeof(int), cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(d_row_curr, &root_row, sizeof(int), cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(d_col_curr, &root_col, sizeof(int), cudaMemcpyHostToDevice));

          int curr_count_sym = 1;
          while (curr_count_sym > 0) {
            CUDA_CHECK(cudaMemset(d_next_count, 0, sizeof(int)));
            CUDA_CHECK(cudaMemset(d_overflow, 0, sizeof(int)));
            const int blocks_sym = static_cast<int>((curr_count_sym + threads - 1) / threads);
            bfs_symbolic_kernel<<<blocks_sym, threads>>>(
                d_edge_arr,
                d_node_arr,
                d_edge_curr,
                d_row_curr,
                d_col_curr,
                curr_count_sym,
                d_edge_next,
                d_row_next,
                d_col_next,
                d_next_count,
                d_row_counts,
                d_total_nnz,
                static_cast<int>(max_out),
                d_overflow);
            CUDA_CHECK(cudaGetLastError());

            int overflow_sym = 0;
            CUDA_CHECK(cudaMemcpy(&overflow_sym, d_overflow, sizeof(int), cudaMemcpyDeviceToHost));
            if (overflow_sym != 0) {
              impl->use_symbolic = false;
              break;
            }

            int next_count_sym = 0;
            CUDA_CHECK(cudaMemcpy(&next_count_sym, d_next_count, sizeof(int), cudaMemcpyDeviceToHost));
            std::swap(d_edge_curr, d_edge_next);
            std::swap(d_row_curr, d_row_next);
            std::swap(d_col_curr, d_col_next);
            curr_count_sym = next_count_sym;
          }

          if (impl->use_symbolic) {
            unsigned long long total_nnz = 0;
            CUDA_CHECK(cudaMemcpy(&total_nnz, d_total_nnz, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            impl->density_est = static_cast<double>(total_nnz) /
                                (static_cast<double>(nDim) * static_cast<double>(nDim));

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
            const int warp = 32;
            const int base = (impl->max_row_nnz > 0) ? impl->max_row_nnz : gate.num_mac;
            int padded = ((base + warp - 1) / warp) * warp;
            const int max_width = static_cast<int>(nDim);
            if (padded > max_width) {
              padded = max_width;
            }
            impl->ell_width = padded;
            impl->use_dense_mv = (impl->density_est >= impl->density_threshold);
          }
        }

        CUDA_CHECK(cudaMemcpy(d_edge_curr, &root_edge, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_row_curr, &root_row, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_col_curr, &root_col, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_factor_curr, &root_factor, sizeof(cuComplex), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemset(d_out_total, 0, sizeof(int)));
        int curr_count = 1;
        while (curr_count > 0) {
          CUDA_CHECK(cudaMemset(d_next_count, 0, sizeof(int)));
          CUDA_CHECK(cudaMemset(d_overflow, 0, sizeof(int)));
          const int blocks = static_cast<int>((curr_count + threads - 1) / threads);
          bfs_expand_kernel<<<blocks, threads>>>(
              d_edge_arr,
              d_node_arr,
              d_edge_curr,
              d_row_curr,
              d_col_curr,
              d_factor_curr,
              curr_count,
              d_edge_next,
              d_row_next,
              d_col_next,
              d_factor_next,
              d_next_count,
              impl->state.d_ray_rows,
              impl->state.d_ray_cols,
              impl->state.d_ray_vals,
              impl->state.sphereValues,
              impl->state.aabbs,
              d_out_total,
              static_cast<int>(max_out),
              static_cast<int>(max_out),
              d_overflow);
          CUDA_CHECK(cudaGetLastError());

          int overflow = 0;
          CUDA_CHECK(cudaMemcpy(&overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost));
          if (overflow != 0) {
            gpu_success = false;
            break;
          }

          int next_count = 0;
          CUDA_CHECK(cudaMemcpy(&next_count, d_next_count, sizeof(int), cudaMemcpyDeviceToHost));
          std::swap(d_edge_curr, d_edge_next);
          std::swap(d_row_curr, d_row_next);
          std::swap(d_col_curr, d_col_next);
          std::swap(d_factor_curr, d_factor_next);
          curr_count = next_count;
        }

        int nnz = 0;
        CUDA_CHECK(cudaMemcpy(&nnz, d_out_total, sizeof(int), cudaMemcpyDeviceToHost));
        if (nnz > 0 && nnz <= static_cast<int>(max_out)) {
          impl->num_rays = static_cast<size_t>(nnz);
          impl->state.d_size = impl->num_rays;
          impl->state.sphere_size = impl->num_rays;
          impl->state.aabb_size = impl->num_rays;
          impl->state.m_result_dim = make_int2(static_cast<int>(nDim), static_cast<int>(nDim));

          impl->state.d_result_buf_size = impl->num_rays * sizeof(cuComplex);
          CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->state.d_result), impl->state.d_result_buf_size));
          CUDA_CHECK(cudaMemset(impl->state.d_result, 0, impl->state.d_result_buf_size));
          CUDA_CHECK(cudaDeviceSynchronize());
          gpu_success = true;
        }
      } catch (const std::exception&) {
        gpu_success = false;
      }

      if (!gpu_success) {
        impl->cleanupGeometry();
      }
      free_temp();
    }

    if (gpu_success) {
      auto dd_stop = std::chrono::high_resolution_clock::now();
      last_stats.dd_ms = std::chrono::duration<double, std::milli>(dd_stop - dd_start).count();
      last_stats.h2d_ms = 0.0;
      return true;
    }

    last_stats = {};
    dd_start = std::chrono::high_resolution_clock::now();
    std::vector<DDNnz> entries;
    entries.reserve(static_cast<size_t>(gate.num_mac) * nDim);
    collectFromDD(gate.fused_edge, 0, 0, make_cuComplex(1.0, 0.0), entries);

    const size_t nnz = entries.size();
    if (nnz == 0) {
      return false;
    }

    std::vector<int> ray_rows;
    std::vector<int> ray_cols;
    std::vector<cuComplex> ray_vals;
    ray_rows.reserve(nnz);
    ray_cols.reserve(nnz);
    ray_vals.reserve(nnz);

    std::vector<OptixAabb> aabbs;
    std::vector<cuComplex> aabb_vals;
    aabbs.reserve(nnz);
    aabb_vals.reserve(nnz);

    for (const auto& entry : entries) {
      ray_rows.push_back(entry.row);
      ray_cols.push_back(entry.col);
      ray_vals.push_back(make_cuComplex(1.0, 0.0));

      OptixAabb aabb{};
      aabb.minX = static_cast<float>(entry.col);
      aabb.maxX = static_cast<float>(entry.col + 1);
      aabb.minY = static_cast<float>(entry.row);
      aabb.maxY = static_cast<float>(entry.row + 1);
      aabb.minZ = 0.0f;
      aabb.maxZ = 1.0f;
      aabbs.push_back(aabb);
      aabb_vals.push_back(entry.val);
    }
    auto dd_stop = std::chrono::high_resolution_clock::now();
    last_stats.dd_ms = std::chrono::duration<double, std::milli>(dd_stop - dd_start).count();

    impl->num_rays = nnz;

    auto h2d_start = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->state.d_ray_rows), nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->state.d_ray_cols), nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->state.d_ray_vals), nnz * sizeof(cuComplex)));
    CUDA_CHECK(cudaMemcpy(impl->state.d_ray_rows, ray_rows.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(impl->state.d_ray_cols, ray_cols.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(impl->state.d_ray_vals, ray_vals.data(), nnz * sizeof(cuComplex), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->state.sphereValues), nnz * sizeof(cuComplex)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->state.aabbs), nnz * sizeof(OptixAabb)));
    CUDA_CHECK(cudaMemcpy(impl->state.sphereValues, aabb_vals.data(), nnz * sizeof(cuComplex), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(impl->state.aabbs, aabbs.data(), nnz * sizeof(OptixAabb), cudaMemcpyHostToDevice));

    impl->state.d_result_buf_size = nnz * sizeof(cuComplex);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->state.d_result), impl->state.d_result_buf_size));
    CUDA_CHECK(cudaMemset(impl->state.d_result, 0, impl->state.d_result_buf_size));

    CUDA_CHECK(cudaDeviceSynchronize());
    auto h2d_stop = std::chrono::high_resolution_clock::now();
    last_stats.h2d_ms = std::chrono::duration<double, std::milli>(h2d_stop - h2d_start).count();

    impl->state.d_size = nnz;
    impl->state.sphere_size = nnz;
    impl->state.aabb_size = nnz;
    impl->state.m_result_dim = make_int2(static_cast<int>(nDim), static_cast<int>(nDim));
    return true;
  } catch (const std::exception&) {
    impl->cleanupGeometry();
    return false;
  }
}

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
    impl->cleanupGeometry();
    impl->resetState();
    impl->last_fused_gates = 0;
    impl->last_reached_density = false;

    impl->use_cuda_merge = true;
    impl->use_symbolic = true;
    impl->density_threshold = envDouble("BQSIM_RT_DENSITY_TARGET", 0.01);
    const uint64_t max_gates_env = envUInt64("BQSIM_RT_SPM_MAX_GATES", static_cast<uint64_t>(gate_count));
    const size_t max_gates = std::min(static_cast<size_t>(max_gates_env), gate_count);
    const bool verbose = envFlag("BQSIM_RT_SPM_VERBOSE");
    impl->num_qubits = num_qubits;
    impl->nDim = nDim;
    qc::GatePrimitive* d_gates = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gates), gate_count * sizeof(qc::GatePrimitive)));
    CUDA_CHECK(cudaMemcpy(d_gates, gates, gate_count * sizeof(qc::GatePrimitive), cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks_n = static_cast<int>((nDim + threads - 1) / threads);

    int* d_M_rows = nullptr;
    int* d_M_cols = nullptr;
    cuComplex* d_M_vals = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_M_rows), nDim * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_M_cols), nDim * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_M_vals), nDim * sizeof(cuComplex)));
    init_identity_kernel<<<blocks_n, threads>>>(static_cast<int>(nDim), d_M_rows, d_M_cols, d_M_vals);
    CUDA_CHECK(cudaGetLastError());
    size_t M_nnz = static_cast<size_t>(nDim);

    const size_t G_nnz = static_cast<size_t>(nDim) * 2;
    int* d_G_rows = nullptr;
    int* d_G_cols = nullptr;
    cuComplex* d_G_vals = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_G_rows), G_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_G_cols), G_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_G_vals), G_nnz * sizeof(cuComplex)));

    auto free_counts = [&]() {
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
    };

    auto free_out = [&]() {
      if (impl->state.d_out_rows) {
        safeCudaFree(impl->state.d_out_rows, "cudaFree(state.d_out_rows)");
      }
      if (impl->state.d_out_cols) {
        safeCudaFree(impl->state.d_out_cols, "cudaFree(state.d_out_cols)");
      }
      if (impl->state.d_out_vals) {
        safeCudaFree(impl->state.d_out_vals, "cudaFree(state.d_out_vals)");
      }
    impl->state.out_capacity = 0;
    };

    auto launch_optix = [&](size_t rays) {
      impl->ensurePipeline();
      impl->buildGas();
      impl->buildSbt();
      if (!impl->stream) {
        CUDA_CHECK(cudaStreamCreate(&impl->stream));
      }
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
      OPTIX_CHECK(optixLaunch(impl->state.pipeline,
                              impl->stream,
                              impl->d_param,
                              sizeof(Params),
                              &impl->state.sbt,
                              static_cast<unsigned int>(rays),
                              1,
                              1));
      CUDA_CHECK(cudaStreamSynchronize(impl->stream));
    };

    for (size_t g = 0; g < max_gates; ++g) {
      const auto gate_start = std::chrono::high_resolution_clock::now();
      build_gate_coo_kernel<<<blocks_n, threads>>>(d_gates,
                                                   static_cast<int>(g),
                                                   static_cast<int>(nDim),
                                                   d_G_rows,
                                                   d_G_cols,
                                                   d_G_vals);
      CUDA_CHECK(cudaGetLastError());
      if (verbose) {
        CUDA_CHECK(cudaDeviceSynchronize());
      }

      if (impl->state.aabbs) {
        CUDA_CHECK(cudaFree(impl->state.aabbs));
        impl->state.aabbs = nullptr;
      }
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->state.aabbs), M_nnz * sizeof(OptixAabb)));
      const int blocks_aabb = static_cast<int>((M_nnz + threads - 1) / threads);
      coo_to_aabb_kernel<<<blocks_aabb, threads>>>(d_M_rows, d_M_cols, impl->state.aabbs, static_cast<int>(M_nnz));
      CUDA_CHECK(cudaGetLastError());
      if (verbose) {
        CUDA_CHECK(cudaDeviceSynchronize());
      }

      impl->num_rays = G_nnz;
      impl->state.d_size = G_nnz;
      impl->state.sphere_size = M_nnz;
      impl->state.aabb_size = M_nnz;
      impl->state.m_result_dim = make_int2(static_cast<int>(nDim), static_cast<int>(nDim));
      impl->state.d_ray_rows = d_G_rows;
      impl->state.d_ray_cols = d_G_cols;
      impl->state.d_ray_vals = d_G_vals;
      impl->state.sphereValues = d_M_vals;

      free_counts();
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->state.d_ray_counts), G_nnz * sizeof(int)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->state.d_row_counts), nDim * sizeof(int)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->state.d_ray_offsets), G_nnz * sizeof(int)));
      CUDA_CHECK(cudaMemset(impl->state.d_ray_counts, 0, G_nnz * sizeof(int)));
      CUDA_CHECK(cudaMemset(impl->state.d_row_counts, 0, nDim * sizeof(int)));

      impl->state.rt_mode = 0;
      const auto sym_start = std::chrono::high_resolution_clock::now();
      launch_optix(G_nnz);
      const auto sym_stop = std::chrono::high_resolution_clock::now();

      auto ray_counts_ptr = thrust::device_pointer_cast(impl->state.d_ray_counts);
      const int total_hits = static_cast<int>(thrust::reduce(thrust::device,
                                                             ray_counts_ptr,
                                                             ray_counts_ptr + G_nnz,
                                                             0));
      if (total_hits <= 0) {
        std::cerr << "[SPMSPM] total_hits <= 0 at gate " << g << ", stopping fusion." << std::endl;
        break;
      }
      thrust::exclusive_scan(thrust::device,
                             ray_counts_ptr,
                             ray_counts_ptr + G_nnz,
                             thrust::device_pointer_cast(impl->state.d_ray_offsets));

      free_out();
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->state.d_out_rows), total_hits * sizeof(int)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->state.d_out_cols), total_hits * sizeof(int)));
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->state.d_out_vals), total_hits * sizeof(cuComplex)));
      impl->state.out_capacity = static_cast<uint64_t>(total_hits);
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->state.d_ray_write_pos), G_nnz * sizeof(int)));
      CUDA_CHECK(cudaMemset(impl->state.d_ray_write_pos, 0, G_nnz * sizeof(int)));

      impl->state.rt_mode = 1;
      const auto num_start = std::chrono::high_resolution_clock::now();
      launch_optix(G_nnz);
      const auto num_stop = std::chrono::high_resolution_clock::now();

      if (impl->state.d_result) {
        CUDA_CHECK(cudaFree(impl->state.d_result));
        impl->state.d_result = nullptr;
      }
      impl->state.d_result_buf_size = static_cast<size_t>(total_hits) * sizeof(cuComplex);
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->state.d_result), impl->state.d_result_buf_size));
      CUDA_CHECK(cudaMemset(impl->state.d_result, 0, impl->state.d_result_buf_size));

      impl->state.d_ray_rows = impl->state.d_out_rows;
      impl->state.d_ray_cols = impl->state.d_out_cols;
      impl->state.sphereValues = impl->state.d_out_vals;
      impl->num_rays = static_cast<size_t>(total_hits);
      impl->state.d_size = impl->num_rays;

      const auto merge_start = std::chrono::high_resolution_clock::now();
      if (!impl->runCudaMerge()) {
        std::cerr << "[SPMSPM] CUDA merge failed at gate " << g << std::endl;
        return false;
      }
      const auto merge_stop = std::chrono::high_resolution_clock::now();
      impl->state.d_out_rows = nullptr;
      impl->state.d_out_cols = nullptr;
      impl->state.d_out_vals = nullptr;
      impl->state.out_capacity = 0;
      if (d_M_rows && d_M_rows != impl->state.d_ray_rows) {
        safeCudaFree(d_M_rows, "cudaFree(d_M_rows)");
      }
      if (d_M_cols && d_M_cols != impl->state.d_ray_cols) {
        safeCudaFree(d_M_cols, "cudaFree(d_M_cols)");
      }
      if (d_M_vals && d_M_vals != impl->state.d_result) {
        safeCudaFree(d_M_vals, "cudaFree(d_M_vals)");
      }

      d_M_rows = impl->state.d_ray_rows;
      d_M_cols = impl->state.d_ray_cols;
      d_M_vals = impl->state.d_result;
      M_nnz = impl->num_rays;
      impl->state.sphereValues = d_M_vals;
      impl->state.sphere_size = M_nnz;

      free_counts();
      free_out();
      impl->state.out_capacity = 0;

      impl->density_est =
          static_cast<double>(M_nnz) / (static_cast<double>(nDim) * static_cast<double>(nDim));
#if defined(BQSIM_DEBUG)
      if (verbose || (g % 4 == 0) || g + 1 == max_gates) {
        const auto gate_stop = std::chrono::high_resolution_clock::now();
        const auto gate_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(gate_stop - gate_start).count();
        const auto sym_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(sym_stop - sym_start).count();
        const auto num_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(num_stop - num_start).count();
        const auto merge_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(merge_stop - merge_start).count();
        std::cerr << "[SPMSPM] gate " << (g + 1) << "/" << max_gates
                  << " M_nnz=" << M_nnz
                  << " density=" << impl->density_est
                  << " gate_ms=" << gate_ms
                  << " sym_ms=" << sym_ms
                  << " num_ms=" << num_ms
                  << " merge_ms=" << merge_ms
                  << std::endl;
      }
#endif
      if (!force_full && impl->density_est >= impl->density_threshold) {
        impl->last_reached_density = true;
        impl->last_fused_gates = g + 1;
        break;
      }
      impl->last_fused_gates = g + 1;
    }
    if (impl->last_fused_gates == 0) {
      impl->last_fused_gates = max_gates;
    }

    int* d_row_counts = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_row_counts), nDim * sizeof(int)));
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
    CUDA_CHECK(cudaFree(d_row_counts));

    int max_row = 0;
    for (size_t i = 0; i < host_counts.size(); ++i) {
      if (host_counts[i] > max_row) {
        max_row = host_counts[i];
      }
    }
    impl->max_row_nnz = max_row;
    const int warp = 32;
    const int base = (impl->max_row_nnz > 0) ? impl->max_row_nnz : 1;
    int padded = ((base + warp - 1) / warp) * warp;
    const int max_width = static_cast<int>(nDim);
    if (padded > max_width) {
      padded = max_width;
    }
    impl->ell_width = padded;
    impl->use_dense_mv = (impl->density_est >= impl->density_threshold);

    impl->num_rays = M_nnz;
    impl->state.d_ray_rows = d_M_rows;
    impl->state.d_ray_cols = d_M_cols;
    impl->state.d_result = d_M_vals;
    impl->state.d_size = impl->num_rays;
    impl->state.d_result_buf_size = impl->num_rays * sizeof(cuComplex);
    impl->state.sphereValues = d_M_vals;
    impl->state.sphere_size = M_nnz;
    impl->state.aabb_size = M_nnz;
    impl->state.m_result_dim = make_int2(static_cast<int>(nDim), static_cast<int>(nDim));
    impl->precomputed_result = true;

    impl->state.d_ray_vals = nullptr;
    CUDA_CHECK(cudaFree(d_gates));
    CUDA_CHECK(cudaFree(d_G_rows));
    CUDA_CHECK(cudaFree(d_G_cols));
    CUDA_CHECK(cudaFree(d_G_vals));
    free_counts();
    free_out();
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[SPMSPM] Exception: " << e.what() << std::endl;
    impl->cleanupGeometry();
    return false;
  }
}

bool RTSpMSpMEngine::launchRTMultiply() {
  if (!available) {
    return false;
  }
  if (impl->num_rays == 0) {
    return false;
  }
  if (impl->precomputed_result) {
    return true;
  }

  if (impl->use_cuda_merge) {
    auto merge_start = std::chrono::high_resolution_clock::now();
    bool ok = impl->runCudaMerge();
    auto merge_stop = std::chrono::high_resolution_clock::now();
    last_stats.gas_ms = 0.0;
    last_stats.launch_ms =
        std::chrono::duration<double, std::milli>(merge_stop - merge_start).count();
    last_stats.compute_ms = last_stats.launch_ms;
    return ok;
  }

  try {
    impl->ensurePipeline();
    auto gas_start = std::chrono::high_resolution_clock::now();
    impl->buildGas();
    auto gas_stop = std::chrono::high_resolution_clock::now();
    last_stats.gas_ms = std::chrono::duration<double, std::milli>(gas_stop - gas_start).count();
    impl->buildSbt();

    if (!impl->stream) {
      CUDA_CHECK(cudaStreamCreate(&impl->stream));
    }

    if (!impl->d_param) {
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&impl->d_param), sizeof(Params)));
    }

    CUDA_CHECK(cudaMemsetAsync(impl->state.d_result, 0, impl->state.d_result_buf_size, impl->stream));

    impl->state.params = {};
    impl->state.params.handle = impl->state.gas_handle;
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(impl->d_param),
                               &impl->state.params,
                               sizeof(Params),
                               cudaMemcpyHostToDevice,
                               impl->stream));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, impl->stream));
    OPTIX_CHECK(optixLaunch(impl->state.pipeline,
                            impl->stream,
                            impl->d_param,
                            sizeof(Params),
                            &impl->state.sbt,
                            static_cast<unsigned int>(impl->num_rays),
                            1,
                            1));
    CUDA_CHECK(cudaEventRecord(stop, impl->stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    last_stats.compute_ms = ms;
    last_stats.launch_ms = ms;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return true;
  } catch (const std::exception&) {
    return false;
  }
}

bool RTSpMSpMEngine::collectResultToELL(cuComplex* values,
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
    CUDA_CHECK(cudaMemset(values, 0, sizeof(cuComplex) * nDim * num_non_zeros));
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

double RTSpMSpMEngine::densityEstimate() const {
  return impl ? impl->density_est : 0.0;
}

int RTSpMSpMEngine::maxRowNNZ() const {
  return impl ? impl->max_row_nnz : 0;
}

int RTSpMSpMEngine::ellWidthHint(int fallback) const {
  if (!impl) {
    return fallback;
  }
  if (impl->ell_width <= 0) {
    return fallback;
  }
  return (impl->ell_width > fallback) ? impl->ell_width : fallback;
}

bool RTSpMSpMEngine::useDenseMV() const {
  return impl ? impl->use_dense_mv : false;
}

std::size_t RTSpMSpMEngine::lastFusedGateCount() const {
  return impl ? impl->last_fused_gates : 0;
}

bool RTSpMSpMEngine::lastReachedDensity() const {
  return impl ? impl->last_reached_density : false;
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

#endif  // BQSIM_USE_RTSPMSPM
