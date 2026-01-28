#ifndef QBATCH_SIMULATOR_H
#define QBATCH_SIMULATOR_H



#include "QuantumComputation.hpp"
#include "Definitions.hpp"
#include "dd/Package.hpp"
#include "operations/OpType.hpp"
#include "dd/Export.hpp"
#include "dd/Operations.hpp"
#include "CircuitOptimizer.hpp"
#include "RTSpMSpMEngine.hpp"
#include "GatePrimitive.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <chrono>
#include <thread>
#include <string>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <cublas_v2.h>
#include <cooperative_groups.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/cuda/cudaflow.hpp>

enum DDELLConversion
{   
  DDELL_GPU = 0, 
  DDELL_CPU = 1, 
  DDELL_Mixed = 2
};


// #define ONE_GB 1*1024*1024

__global__ void replicate(cuDoubleComplex *input_arr_d, int N) {
  input_arr_d[threadIdx.x+blockIdx.x*N] = input_arr_d[blockIdx.x*N];
}

__global__ void initial_check(cuDoubleComplex *input_arr_d, bool *identical, int N) {
  extern __shared__ bool s[];
  __shared__ int res[1];
  if (threadIdx.x == 0) {
    res[0] = true;
  }
  __syncthreads();
  s[threadIdx.x] = ((input_arr_d[threadIdx.x+blockIdx.x*N].x == input_arr_d[blockIdx.x*N].x) && 
    (input_arr_d[threadIdx.x+blockIdx.x*N].y == input_arr_d[blockIdx.x*N].y));
  __syncthreads();
  atomicAnd(res, (int)s[threadIdx.x]);
  __syncthreads();
  if (threadIdx.x == 0) {
    identical[blockIdx.x] = res[0];
  }
}


__global__ void dd_extract_matrix(
  dd::GPU_DD_edge* dd_edges,
  dd::GPU_DD_node* dd_nodes,
  cuDoubleComplex *fused_gate_val,
  int *fused_gate_indices,
  int num_nodes,
  int num_edges,
  int num_non_zeros,
  int num_qubits
) {
  __shared__ int decoded_locs[MAX_DECODED_MACS];
  __shared__ cuDoubleComplex decoded_factors[MAX_DECODED_MACS];
  // recording the recursive state of a certain node
  __shared__ uint8_t left_or_right[MAX_LEV]; // left: F right: T
  __shared__ bool up_or_down[MAX_LEV]; // up: F down: T
  __shared__ int decode_ptr[1];
  __shared__ int edge_stack[MAX_LEV];

  int bid = blockIdx.x;
  int tid = threadIdx.x;
  
  if (tid < num_qubits) {
    left_or_right[tid] = 0;
    up_or_down[num_qubits-1-tid] = bid & (1 << tid);
  }
  __syncthreads();

  // every block decodes the DDNode struct and list the necessary MACs (weights & location) in shared mem
  if (tid == 0) {
    int edge_ptr = 0;
    int node_ptr = 0;
    int stack_ptr = 0;
    decode_ptr[0] = 0;
    
    edge_stack[stack_ptr] = 0;
    cuDoubleComplex rec_factor = {1, 0};
    int rec_loc = 0; // recursive location
    // DFS
    while (stack_ptr >= 0) {
      if (decode_ptr[0] == num_non_zeros) break;
      // fetch node
      edge_ptr = edge_stack[stack_ptr];
      if (edge_ptr == dd::const_zero_edge) {
        stack_ptr--;
        continue;
      }
      node_ptr = dd_edges[edge_ptr].DD_node_ptr;
      if (node_ptr == dd::const_one_node) {
        decoded_locs[decode_ptr[0]] = rec_loc;
        decoded_factors[decode_ptr[0]] = cuCmul(rec_factor, dd_edges[edge_ptr].w);
        stack_ptr--; decode_ptr[0]++;
        continue;
      }

      int child_idx = (int)(left_or_right[stack_ptr]) + (int)(up_or_down[stack_ptr]) * 2;
      // return or move forward
      if (left_or_right[stack_ptr] == 2) {
        left_or_right[stack_ptr] = 0;
        rec_factor = cuCdiv(rec_factor, dd_edges[edge_ptr].w);
        rec_loc -= (1 << dd_nodes[node_ptr].qubit);
        stack_ptr--;
      }
      else {
        left_or_right[stack_ptr]++;
        rec_factor = (left_or_right[stack_ptr] == 1)? cuCmul(rec_factor, dd_edges[edge_ptr].w) : rec_factor;
        rec_loc += (1 << dd_nodes[node_ptr].qubit) * (int)(left_or_right[stack_ptr] -1);
        stack_ptr++;
        edge_stack[stack_ptr] = dd_nodes[node_ptr].outgoing_DD_edge_ptr[child_idx];
      }
    }
  }

  __syncthreads();
  if (tid < num_non_zeros) {
    fused_gate_val[bid * num_non_zeros + tid] = {0, 0};
    fused_gate_indices[bid * num_non_zeros + tid] = 0;
  }
  __syncthreads();

  if (tid < decode_ptr[0]) {
    fused_gate_val[bid * num_non_zeros + tid] = decoded_factors[tid];
    fused_gate_indices[bid * num_non_zeros + tid] = decoded_locs[tid];
  }
  __syncthreads();

}

__global__ void dd_extract_matrix_warp(
  dd::GPU_DD_edge* dd_edges,
  dd::GPU_DD_node* dd_nodes,
  cuDoubleComplex *fused_gate_val,
  int *fused_gate_indices,
  int num_nodes,
  int num_edges,
  int num_non_zeros,
  int num_qubits
) {
  __shared__ int decoded_locs[MAX_DECODED_MACS*WARPS_PER_BLOCK];
  __shared__ cuDoubleComplex decoded_factors[MAX_DECODED_MACS*WARPS_PER_BLOCK];
  // recording the recursive state of a certain node
  __shared__ uint8_t left_or_right[MAX_LEV*WARPS_PER_BLOCK]; // left: F right: T
  __shared__ bool up_or_down[MAX_LEV*WARPS_PER_BLOCK]; // up: F down: T
  __shared__ int decode_ptr[WARPS_PER_BLOCK];
  __shared__ int edge_stack[MAX_LEV*WARPS_PER_BLOCK];

  int bid = blockIdx.x;
  int tid = threadIdx.x;
  
  if (tid%WARP_SIZE < num_qubits) {
    left_or_right[MAX_LEV*(tid/WARP_SIZE) + tid%WARP_SIZE] = 0;
    up_or_down[MAX_LEV*(tid/WARP_SIZE) + num_qubits-1-tid%WARP_SIZE] = (bid*WARPS_PER_BLOCK+tid/WARP_SIZE) & (1 << tid%WARP_SIZE);
  }
  __syncwarp();

  // every block decodes the DDNode struct and list the necessary MACs (weights & location) in shared mem
  if (tid%WARP_SIZE == 0) {
    int edge_ptr = 0;
    int node_ptr = 0;
    int stack_ptr = 0;
    decode_ptr[tid/WARP_SIZE] = 0;
    
    edge_stack[MAX_LEV*(tid/WARP_SIZE)+stack_ptr] = 0;
    cuDoubleComplex rec_factor = {1, 0};
    int rec_loc = 0; // recursive location
    // DFS
    while (stack_ptr >= 0) {
      if (decode_ptr[tid/WARP_SIZE] == num_non_zeros) break;
      // fetch node
      edge_ptr = edge_stack[MAX_LEV*(tid/WARP_SIZE)+stack_ptr];
      if (edge_ptr == dd::const_zero_edge) {
        stack_ptr--;
        continue;
      }
      node_ptr = dd_edges[edge_ptr].DD_node_ptr;
      if (node_ptr == dd::const_one_node) {
        decoded_locs[MAX_DECODED_MACS*(tid/WARP_SIZE)+decode_ptr[tid/WARP_SIZE]] = rec_loc;
        decoded_factors[MAX_DECODED_MACS*(tid/WARP_SIZE)+decode_ptr[tid/WARP_SIZE]] = cuCmul(rec_factor, dd_edges[edge_ptr].w);
        stack_ptr--; decode_ptr[tid/WARP_SIZE]++;
        continue;
      }

      int child_idx = (int)(left_or_right[MAX_LEV*(tid/WARP_SIZE) + stack_ptr]) + (int)(up_or_down[MAX_LEV*(tid/WARP_SIZE) + stack_ptr]) * 2;
      // return or move forward
      if (left_or_right[MAX_LEV*(tid/WARP_SIZE) + stack_ptr] == 2) {
        left_or_right[MAX_LEV*(tid/WARP_SIZE) + stack_ptr] = 0;
        rec_factor = cuCdiv(rec_factor, dd_edges[edge_ptr].w);
        rec_loc -= (1 << dd_nodes[node_ptr].qubit);
        stack_ptr--;
      }
      else {
        left_or_right[MAX_LEV*(tid/WARP_SIZE) + stack_ptr]++;
        rec_factor = (left_or_right[MAX_LEV*(tid/WARP_SIZE) + stack_ptr] == 1)? cuCmul(rec_factor, dd_edges[edge_ptr].w) : rec_factor;
        rec_loc += (1 << dd_nodes[node_ptr].qubit) * (int)(left_or_right[MAX_LEV*(tid/WARP_SIZE) + stack_ptr] -1);
        stack_ptr++;
        edge_stack[MAX_LEV*(tid/WARP_SIZE)+stack_ptr] = dd_nodes[node_ptr].outgoing_DD_edge_ptr[child_idx];
      }
    }
  }

  __syncwarp();
  if (tid%WARP_SIZE < num_non_zeros) {
    fused_gate_val[(bid*WARPS_PER_BLOCK+tid/WARP_SIZE) * num_non_zeros + tid%WARP_SIZE] = {0, 0};
    fused_gate_indices[(bid*WARPS_PER_BLOCK+tid/WARP_SIZE) * num_non_zeros + tid%WARP_SIZE] = 0;
  }
  __syncwarp();

  if (tid%WARP_SIZE < decode_ptr[tid/WARP_SIZE]) {
    fused_gate_val[(bid*WARPS_PER_BLOCK+tid/WARP_SIZE) * num_non_zeros + tid%WARP_SIZE] = decoded_factors[MAX_DECODED_MACS*(tid/WARP_SIZE)+tid%WARP_SIZE];
    fused_gate_indices[(bid*WARPS_PER_BLOCK+tid/WARP_SIZE) * num_non_zeros + tid%WARP_SIZE] = decoded_locs[MAX_DECODED_MACS*(tid/WARP_SIZE)+tid%WARP_SIZE];
  }
  __syncwarp();

}



// __global__ void run_fused_gate(
//   cuDoubleComplex *gates_val,
//   int *gates_indices,
//   int num_non_zero,
//   cuDoubleComplex *input_state,
//   cuDoubleComplex *output_state,
//   int batch_size, 
//   int nDim
// ) {
  
//   int tidx = threadIdx.x;
//   int tidy = threadIdx.y;
//   int tid = tidx+tidy*blockDim.y;
//   int rounds = nDim / (gridDim.x*blockDim.y);
//   int bid = blockIdx.x;
//   __shared__ int share_indices[MAX_VAL];
//   __shared__ cuDoubleComplex shared_val[MAX_VAL];


//   for (int i = 0; i < rounds; i++) {
//     if (tid < num_non_zero * blockDim.y) {
//       share_indices[tid] = gates_indices[((rounds * bid+i)*blockDim.y) * num_non_zero + tid];
//       shared_val[tid] = gates_val[((rounds * bid+i)*blockDim.y) * num_non_zero + tid];
//     }
//     __syncthreads();
//     cuDoubleComplex result_value = {0, 0};
//     for (int j = 0; j < num_non_zero; j++) {
//       cuDoubleComplex temp_value = cuCmul(input_state[share_indices[tidy*num_non_zero+j]*batch_size+tidx], shared_val[tidy*num_non_zero+j]);
//       result_value = cuCadd(result_value, temp_value);
//     }
//     __syncthreads();
//     output_state[((rounds * bid+i)*blockDim.y+tidy)*batch_size +tidx] = result_value;
//   }
//   __syncthreads();
// }

__global__ void run_fused_gate_warp(
  const cuDoubleComplex* __restrict__ gates_val,
  const int* __restrict__ gates_indices,
  int num_non_zero,
  const cuDoubleComplex* __restrict__ input_state,
  cuDoubleComplex* __restrict__ output_state,
  int batch_size,
  int nDim
) {
  const int lane = threadIdx.x & 31;
  const int warps_per_block = blockDim.x >> 5;
  const int warp_in_block = threadIdx.x >> 5;
  const int warp_id = blockIdx.x * warps_per_block + warp_in_block;
  const int batch_chunks = (batch_size + 31) >> 5;
  const int row = warp_id / batch_chunks;
  if (row >= nDim) {
    return;
  }
  const int batch = (warp_id % batch_chunks) * 32 + lane;
  if (batch >= batch_size) {
    return;
  }
  const size_t row_off = static_cast<size_t>(row) * static_cast<size_t>(num_non_zero);
  cuDoubleComplex acc = {0, 0};
  for (int j = 0; j < num_non_zero; ++j) {
    int col = 0;
    double vx = 0.0;
    double vy = 0.0;
    if (lane == 0) {
      col = __ldg(gates_indices + row_off + j);
      const double2 v = __ldg(reinterpret_cast<const double2*>(gates_val + row_off + j));
      vx = v.x;
      vy = v.y;
    }
    col = __shfl_sync(0xFFFFFFFF, col, 0);
    vx = __shfl_sync(0xFFFFFFFF, vx, 0);
    vy = __shfl_sync(0xFFFFFFFF, vy, 0);
    if (vx != 0.0 || vy != 0.0) {
      const size_t b_idx = static_cast<size_t>(col) * batch_size + batch;
      const double2 b2 = __ldg(reinterpret_cast<const double2*>(input_state + b_idx));
      const cuDoubleComplex b = make_cuDoubleComplex(b2.x, b2.y);
      const cuDoubleComplex a = make_cuDoubleComplex(vx, vy);
      acc = cuCadd(acc, cuCmul(a, b));
    }
  }
  output_state[static_cast<size_t>(row) * batch_size + batch] = acc;
}

__global__ void run_fused_gate_warp_mega(
  const cuDoubleComplex* const* __restrict__ gates_val_list,
  const int* const* __restrict__ gates_idx_list,
  const int* __restrict__ nnz_list,
  int gate_count,
  const cuDoubleComplex* __restrict__ state_a,
  cuDoubleComplex* __restrict__ state_b,
  int batch_size,
  int nDim
) {
  namespace cg = cooperative_groups;
  cg::grid_group grid = cg::this_grid();
  const int lane = threadIdx.x & 31;
  const int warps_per_block = blockDim.x >> 5;
  const int warp_in_block = threadIdx.x >> 5;
  const int warp_id = blockIdx.x * warps_per_block + warp_in_block;
  const int batch_chunks = (batch_size + 31) >> 5;
  const int row = warp_id / batch_chunks;
  if (row >= nDim) {
    return;
  }
  const int batch = (warp_id % batch_chunks) * 32 + lane;
  if (batch >= batch_size) {
    return;
  }

  for (int g = 0; g < gate_count; ++g) {
    const cuDoubleComplex* gates_val = gates_val_list[g];
    const int* gates_indices = gates_idx_list[g];
    const int num_non_zero = nnz_list[g];
    const cuDoubleComplex* input_state = (g & 1) ? state_b : state_a;
    cuDoubleComplex* output_state = (g & 1) ? const_cast<cuDoubleComplex*>(state_a) : state_b;
    const size_t row_off = static_cast<size_t>(row) * static_cast<size_t>(num_non_zero);
    cuDoubleComplex acc = {0, 0};
    for (int j = 0; j < num_non_zero; ++j) {
      int col = 0;
      double vx = 0.0;
      double vy = 0.0;
      if (lane == 0) {
        col = __ldg(gates_indices + row_off + j);
        const double2 v = __ldg(reinterpret_cast<const double2*>(gates_val + row_off + j));
        vx = v.x;
        vy = v.y;
      }
      col = __shfl_sync(0xFFFFFFFF, col, 0);
      vx = __shfl_sync(0xFFFFFFFF, vx, 0);
      vy = __shfl_sync(0xFFFFFFFF, vy, 0);
      if (vx != 0.0 || vy != 0.0) {
        const size_t b_idx = static_cast<size_t>(col) * batch_size + batch;
        const double2 b2 = __ldg(reinterpret_cast<const double2*>(input_state + b_idx));
        const cuDoubleComplex b = make_cuDoubleComplex(b2.x, b2.y);
        const cuDoubleComplex a = make_cuDoubleComplex(vx, vy);
        acc = cuCadd(acc, cuCmul(a, b));
      }
    }
    output_state[static_cast<size_t>(row) * batch_size + batch] = acc;
    grid.sync();
  }
}

__global__ void ell_to_dense(
  const cuDoubleComplex* ell_vals,
  const int* ell_indices,
  cuDoubleComplex* dense,
  int ell_width,
  int nDim
) {
  const int row = blockIdx.y;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= nDim || idx >= ell_width) {
    return;
  }
  const int col = ell_indices[row * ell_width + idx];
  const cuDoubleComplex val = ell_vals[row * ell_width + idx];
  if (val.x != 0.0 || val.y != 0.0) {
    dense[static_cast<size_t>(row) * static_cast<size_t>(nDim) + col] = val;
  }
}

__global__ void run_dense_gate_tiled(
  const cuDoubleComplex* dense,
  const cuDoubleComplex* input_state,
  cuDoubleComplex* output_state,
  int batch_size,
  int nDim,
  int tile_k,
  bool assume_dense
) {
  const int tid = threadIdx.x;
  if (tid >= batch_size) {
    return;
  }
  extern __shared__ cuDoubleComplex shared_a[];
  int row = blockIdx.x;
  while (row < nDim) {
    cuDoubleComplex acc = {0, 0};
    const size_t row_off = static_cast<size_t>(row) * static_cast<size_t>(nDim);
    for (int tile = 0; tile < nDim; tile += tile_k) {
      const int tile_end = min(tile + tile_k, nDim);
      const int tile_len = tile_end - tile;
      for (int t = tid; t < tile_len; t += blockDim.x) {
        shared_a[t] = dense[row_off + static_cast<size_t>(tile + t)];
      }
      __syncthreads();
      if (assume_dense) {
        for (int t = 0; t < tile_len; ++t) {
          const int col = tile + t;
          const cuDoubleComplex b = input_state[static_cast<size_t>(col) * batch_size + tid];
          acc = cuCadd(acc, cuCmul(shared_a[t], b));
        }
      } else {
        for (int t = 0; t < tile_len; ++t) {
          const cuDoubleComplex a = shared_a[t];
          if (a.x != 0.0 || a.y != 0.0) {
            const int col = tile + t;
            const cuDoubleComplex b = input_state[static_cast<size_t>(col) * batch_size + tid];
            acc = cuCadd(acc, cuCmul(a, b));
          }
        }
      }
      __syncthreads();
    }
    output_state[static_cast<size_t>(row) * batch_size + tid] = acc;
    row += gridDim.x;
  }
}


template<class Config = dd::DDPackageConfig>
class QBatchSimulator {
public:
    explicit QBatchSimulator(std::unique_ptr<qc::QuantumComputation>&& qc_, int batch_size_, int num_batch_) : 
    qc(std::move(qc_)), batch_size(batch_size_), num_batch(num_batch_), rtEngine(std::make_unique<RTSpMSpMEngine>())
    {
        #if defined(BQSIM_USE_RTSPMSPM)
        rtEngine->setAvailable(true);
        #endif
        // Force CUDA runtime initialization early to surface driver/device issues.
        int device_count = 0;
        const cudaError_t init_err = cudaGetDeviceCount(&device_count);
        if (init_err != cudaSuccess || device_count <= 0) {
          throw std::runtime_error(std::string("CUDA init failed: ") +
                                   cudaGetErrorString(init_err));
        }
        checkCudaErrors(cudaSetDevice(0));
        checkCudaErrors(cudaFree(0));
        QBatchSimulator<Config>::dd->resize(qc->getNqubits());
        const auto nQubits = qc->getNqubits();
        nDim    = std::pow(2, nQubits);
        
        cuDoubleComplex *h_batch0;
        cuDoubleComplex *h_batch1;
        const size_t host_bytes = nDim * batch_size_ * sizeof(cuDoubleComplex);
        bool pinned0 = false;
        bool pinned1 = false;
        if (cudaMallocHost((void**)&h_batch0, host_bytes) == cudaSuccess) {
          pinned0 = true;
        } else {
          h_batch0 = static_cast<cuDoubleComplex*>(std::malloc(host_bytes));
          if (!h_batch0) {
            throw std::bad_alloc();
          }
        }
        if (cudaMallocHost((void**)&h_batch1, host_bytes) == cudaSuccess) {
          pinned1 = true;
        } else {
          h_batch1 = static_cast<cuDoubleComplex*>(std::malloc(host_bytes));
          if (!h_batch1) {
            if (pinned0) {
              checkCudaErrors(cudaFreeHost(h_batch0));
            } else {
              std::free(h_batch0);
            }
            throw std::bad_alloc();
          }
        }

        std::string filename = "../../input_batch/n"+std::to_string(nQubits)+".txt";
        std::ifstream file;
        file.open((filename).c_str());

        if (!file.is_open()) {
            std::cerr << "Failed to open file." << std::endl;
            exit(-1);
        }
        std::string line;
        while (getline(file, line)) {
            std::istringstream iss(line);
            double real, imag;
            int amp_id = 0;
            while (iss >> real >> imag) {
            h_batch0[amp_id*batch_size_] = {real, imag};
            amp_id++;
            }
        }
        file.close();

        cuDoubleComplex *input_d;
        checkCudaErrors(cudaMalloc((void**)&input_d, nDim * batch_size_ * sizeof(cuDoubleComplex)));
        checkCudaErrors(cudaMemcpy(input_d, h_batch0, nDim * batch_size_ * sizeof(cuDoubleComplex),
                cudaMemcpyHostToDevice));
        replicate<<<nDim, batch_size>>>(input_d, batch_size_);
        checkCudaErrors(cudaMemcpy(h_batch0, input_d, nDim * batch_size_ * sizeof(cuDoubleComplex),
                cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(input_d));
        
        memset(h_batch1, 0, nDim * batch_size_ * sizeof(cuDoubleComplex));
        h_batch.push_back(h_batch0);
        h_batch_pinned.push_back(static_cast<uint8_t>(pinned0));
        h_batch.push_back(h_batch1);
        h_batch_pinned.push_back(static_cast<uint8_t>(pinned1));

        for (int buf = 0; buf < 4; buf++) {
          cuDoubleComplex *d_batch_buf;
          checkCudaErrors(cudaMalloc((void**)&d_batch_buf, nDim * batch_size_ * sizeof(cuDoubleComplex)));
          d_batch.push_back(d_batch_buf);
        }
        
    };

    ~QBatchSimulator() {
      for (size_t i = 0; i < h_batch.size(); i++)
      {
        if (i < h_batch_pinned.size() && h_batch_pinned[i]) {
          checkCudaErrors(cudaFreeHost(h_batch[i]));
        } else {
          std::free(h_batch[i]);
        }
      }
      for (int i = 0; i < d_batch.size(); i++) {
        checkCudaErrors(cudaFree(d_batch[i]));
      }
      for (int i = 0; i < fused_gates_val_d.size(); i++) {
        checkCudaErrors(cudaFree(fused_gates_val_d[i]));
        checkCudaErrors(cudaFree(fused_gates_indices_d[i]));
      }
      for (int i = 0; i < fused_gates_dense_d.size(); i++) {
        if (fused_gates_dense_d[i]) {
          checkCudaErrors(cudaFree(fused_gates_dense_d[i]));
        }
      }
      // for (int i = 0; i < fused_gates_val_mored.size(); i++) {
      //   checkCudaErrors(cudaFree(fused_gates_val_mored[i]));
      //   checkCudaErrors(cudaFree(fused_gates_indices_mored[i]));
      // }
    }

    void simulate() {
        bool hasNonmeasurementNonUnitary = false;
        bool hasMeasurements             = false;
        bool measurementsLast            = true;


        for (auto& op: *qc) {
            if (op->isClassicControlledOperation() || (op->isNonUnitaryOperation() && op->getType() != qc::Measure && op->getType() != qc::Barrier)) {
                hasNonmeasurementNonUnitary = true;
            }
            if (op->getType() == qc::Measure) {
                auto* nonUnitaryOp = dynamic_cast<qc::NonUnitaryOperation*>(op.get());
                if (nonUnitaryOp == nullptr) {
                    throw std::runtime_error("Op with type Measurement could not be casted to NonUnitaryOperation");
                }
                hasMeasurements = true;

                const auto& quantum = nonUnitaryOp->getTargets();
                const auto& classic = nonUnitaryOp->getClassics();

                if (quantum.size() != classic.size()) {
                    throw std::runtime_error("Measurement: Sizes of quantum and classic register mismatch.");
                }

            }

            if (hasMeasurements && op->isUnitary()) {
                measurementsLast = false;
            }
        }

        // easiest case: all gates are unitary --> simulate once and sample away on all qubits
        if (!hasNonmeasurementNonUnitary && !hasMeasurements) {
            singleShot(false);
            return;
        }

        // single shot is enough, but the sampling should only return actually measured qubits
        if (!hasNonmeasurementNonUnitary && measurementsLast) {
            singleShot(true);
            const auto                         qubits = qc->getNqubits();
            const auto                         cbits  = qc->getNcbits();

            return;
        }
        return;
    }


    void singleShot(bool ignoreNonUnitaries) {
        std::size_t                 opNum = 0;
        std::vector<int> fused_num_nonzero;
        std::vector<qc::FusedGate> fused_gates;
        int current_buffer_idx = 0;

        auto envFlag = [](const char* name) {
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
        };
        auto envDouble = [](const char* name, double fallback) {
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
        };
        auto envUInt64 = [](const char* name, uint64_t fallback) {
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
        };
        int dense_tile_k = static_cast<int>(envUInt64("BQSIM_RT_DENSE_TILE", 256));
        const bool dense_assume_dense = envFlag("BQSIM_RT_DENSE_ASSUME_DENSE");
        if (dense_tile_k <= 0) {
          dense_tile_k = 256;
        }
        if (dense_tile_k > static_cast<int>(nDim)) {
          dense_tile_k = static_cast<int>(nDim);
        }
        if (dense_tile_k > 2048) {
          dense_tile_k = 2048;
        }

        bool used_spm_pipeline = false;
        const char* pipeline_mode = std::getenv("BQSIM_RT_PIPELINE_MODE");
        const bool use_spm_pipeline = pipeline_mode && std::strcmp(pipeline_mode, "SPMSPM") == 0 &&
                                      rtEngine && rtEngine->isAvailable() && qc->getNqubits() >= 12;
        if (use_spm_pipeline) {
          std::vector<qc::GatePrimitive> primitives;
          if (!buildGatePrimitives(primitives)) {
            std::cerr << "[SPMSPM] GatePrimitive build failed; aborting SPMSPM pipeline." << std::endl;
            return;
          }
          auto begin_convert = std::chrono::high_resolution_clock::now();
          const bool hybrid_enabled = envFlag("BQSIM_RT_HYBRID_DENSE");
          const uint64_t block_gates_env = envUInt64("BQSIM_RT_SPM_BLOCK_GATES", 0);
          const size_t total_gates = primitives.size();
          const size_t block_gates = (block_gates_env == 0)
                                         ? total_gates
                                         : std::min(static_cast<size_t>(block_gates_env), total_gates);
          size_t target_fused_count = static_cast<size_t>(envUInt64("BQSIM_RT_TARGET_FUSED_COUNT", 4));
          if (target_fused_count == 0) {
            target_fused_count = 1;
          }
          target_fused_count = std::min(target_fused_count, total_gates);
          if (block_gates_env > 0 && block_gates > 0) {
            const size_t min_blocks_for_cap = (total_gates + block_gates - 1) / block_gates;
            target_fused_count = std::max(target_fused_count, min_blocks_for_cap);
          }
          bool dense_enabled = envFlag("BQSIM_RT_DENSE_GEMV");
          const double dense_threshold = envDouble("BQSIM_RT_DENSE_THRESHOLD", 0.01);
          const uint64_t dense_max_bytes =
              envUInt64("BQSIM_RT_DENSE_MAX_BYTES", 512ULL * 1024ULL * 1024ULL);
          if (dense_enabled && (batch_size % 32 != 0)) {
            std::cerr << "[SPMSPM] Dense GEMV: batch_size not multiple of 32; expect lower memory coalescing." << std::endl;
          }

          auto cleanup_spm = [&]() {
            for (size_t i = 0; i < fused_gates_val_d.size(); ++i) {
              if (fused_gates_val_d[i]) {
                cudaFree(fused_gates_val_d[i]);
              }
              if (i < fused_gates_indices_d.size() && fused_gates_indices_d[i]) {
                cudaFree(fused_gates_indices_d[i]);
              }
              if (i < fused_gates_dense_d.size() && fused_gates_dense_d[i]) {
                cudaFree(fused_gates_dense_d[i]);
              }
            }
            fused_gates_val_d.clear();
            fused_gates_indices_d.clear();
            fused_gates_dense_d.clear();
            fused_gates_use_dense.clear();
          };

          std::vector<size_t> block_sizes;
          block_sizes.reserve(target_fused_count);
          const size_t base_count = total_gates / target_fused_count;
          const size_t remainder = total_gates % target_fused_count;
          for (size_t i = 0; i < target_fused_count; ++i) {
            const size_t extra = (i >= target_fused_count - remainder) ? 1 : 0;
            size_t count = base_count + extra;
            if (count == 0) {
              count = 1;
            }
            if (block_gates_env > 0 && count > block_gates) {
              count = block_gates;
            }
            block_sizes.push_back(count);
          }

          size_t cursor = 0;
          size_t block_id = 0;
          while (cursor < total_gates) {
            const size_t remaining = total_gates - cursor;
            const size_t plan_idx = std::min(block_id, block_sizes.size() - 1);
            const size_t planned = std::min(block_sizes[plan_idx], remaining);
            if (planned == 0) {
              break;
            }
            std::cout << "[SPMSPM] Fusing block " << (block_id + 1)
                      << " (plan " << (plan_idx + 1) << "/" << block_sizes.size() << ")"
                      << " starting at gate " << cursor
                      << " with up to " << planned << " gates" << std::endl;

            rtEngine->resetStats();
            if (!(rtEngine->prepareGeometryFromGates(primitives.data() + cursor,
                                                     planned,
                                                     static_cast<int>(qc->getNqubits()),
                                                     nDim,
                                                     !hybrid_enabled) &&
                  rtEngine->launchRTMultiply())) {
              std::cerr << "[SPMSPM] prepareGeometryFromGates/launchRTMultiply failed; aborting SPMSPM pipeline."
                        << std::endl;
              cleanup_spm();
              return;
            }

            const int ell_width = rtEngine->ellWidthHint(1);
            cuDoubleComplex* fused_gate_val = nullptr;
            int* fused_gate_indices = nullptr;
            if (cudaMalloc((void**)&fused_gate_val, ell_width * nDim * sizeof(cuDoubleComplex)) != cudaSuccess ||
                cudaMalloc((void**)&fused_gate_indices, ell_width * nDim * sizeof(int)) != cudaSuccess) {
              if (fused_gate_val) {
                cudaFree(fused_gate_val);
              }
              if (fused_gate_indices) {
                cudaFree(fused_gate_indices);
              }
              std::cerr << "[SPMSPM] cudaMalloc failed during ELL allocation; aborting SPMSPM pipeline."
                        << std::endl;
              cleanup_spm();
              return;
            }
            if (rtEngine->collectResultToELL(fused_gate_val, fused_gate_indices, ell_width, nDim)) {
              fused_gates_val_d.push_back(fused_gate_val);
              fused_gates_indices_d.push_back(fused_gate_indices);
              fused_num_nonzero.push_back(ell_width);
              fused_gates_use_dense.push_back(0);
              fused_gates_dense_d.push_back(nullptr);
              if (dense_enabled && rtEngine->densityEstimate() >= dense_threshold) {
                const uint64_t nDim64 = static_cast<uint64_t>(nDim);
                const uint64_t dense_elems = nDim64 * nDim64;
                const uint64_t dense_bytes = dense_elems * sizeof(cuDoubleComplex);
                if ((nDim64 == 0) || (dense_elems / nDim64 != nDim64)) {
                  std::cerr << "[SPMSPM] dense size overflow; skipping dense GEMV" << std::endl;
                } else if (dense_max_bytes == 0 || dense_bytes <= dense_max_bytes) {
                  cuDoubleComplex* dense = nullptr;
                  if (cudaMalloc((void**)&dense, dense_bytes) == cudaSuccess) {
                    checkCudaErrors(cudaMemset(dense, 0, dense_bytes));
                    dim3 block(256, 1, 1);
                    dim3 grid((ell_width + block.x - 1) / block.x, nDim, 1);
                    ell_to_dense<<<grid, block>>>(fused_gate_val, fused_gate_indices, dense, ell_width, static_cast<int>(nDim));
                    checkCudaErrors(cudaGetLastError());
                    fused_gates_use_dense.back() = 1;
                    fused_gates_dense_d.back() = dense;
                  } else {
                    cudaGetLastError();
                    std::cerr << "[SPMSPM] dense cudaMalloc failed; fallback to ELL" << std::endl;
                  }
                }
              }
            } else {
              checkCudaErrors(cudaFree(fused_gate_val));
              checkCudaErrors(cudaFree(fused_gate_indices));
              std::cerr << "[SPMSPM] collectResultToELL failed; aborting SPMSPM pipeline." << std::endl;
              cleanup_spm();
              return;
            }

            size_t actual = rtEngine->lastFusedGateCount();
            if (actual == 0) {
              actual = planned; // Avoid infinite loops if the engine does not report progress.
              std::cerr << "[SPMSPM] lastFusedGateCount returned 0; assuming " << actual
                        << " gates were fused." << std::endl;
            }
            std::cout << "[SPMSPM]   fused " << actual << " gate(s)" << std::endl;
            cursor += std::min(actual, remaining);
            ++block_id;
          }
          used_spm_pipeline = true;
          auto end_convert = std::chrono::high_resolution_clock::now();
          std::cout << "[Stage 1: Gate Fusion] time: 0" << std::endl;
          std::cout << "[Stage 2: DD-to-ELL Conversion] time: "
                    << std::chrono::duration_cast<std::chrono::milliseconds>(end_convert - begin_convert).count()
                    << std::endl;
        } else {
        auto begin_fusion = std::chrono::high_resolution_clock::now();
        // GateFusion takes ownership of the DD package; use a local instance.
        auto dd_local = std::make_unique<dd::Package<Config>>();
        dd_local->resize(qc->getNqubits());
        qc::CircuitOptimizer::GateFusion(std::move(qc), fused_gates, std::move(dd_local), nDim, true);
        auto end_fusion = std::chrono::high_resolution_clock::now();
        std::cout << "[Stage 1: Gate Fusion] time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_fusion - begin_fusion).count() << std::endl;

        auto begin_convert = std::chrono::high_resolution_clock::now();
        int total_macs = 0;
        double rt_h2d_ms = 0.0;
        double rt_dd_ms = 0.0;
        double rt_gas_ms = 0.0;
        double rt_launch_ms = 0.0;
        double rt_ell_ms = 0.0;
        int rt_count = 0;
        int *sparse_idx_x;
        checkCudaErrors(cudaMallocHost((void**)&sparse_idx_x, nDim* sizeof(int)));
        cuDoubleComplex *fused_gate_val_h;
        int * fused_gate_indices_h;
        for (int idx = 0; idx < fused_gates.size(); idx++) {
          qc::FusedGate fused_gate = fused_gates[idx];
          fused_num_nonzero.push_back(fused_gate.num_mac );
          fused_gates_use_dense.push_back(0);
          fused_gates_dense_d.push_back(nullptr);
          total_macs += fused_gate.num_mac;

          std::cout << "Converting fused gate #" << idx << " using ";
          auto begin_gate_convert = std::chrono::high_resolution_clock::now();
          bool rt_done = false;
          const bool use_rt = rtEngine && rtEngine->isAvailable() &&
                              (fused_gate.num_edges > rt_threshold || qc->getNqubits() > rt_qubit_threshold);
          if (use_rt) {
            std::cout << "RT Core" << std::endl;
            rtEngine->resetStats();
            if (rtEngine->prepareGeometry(fused_gate, QBatchSimulator<Config>::dd.get(),
                                          qc->getNqubits(), nDim) &&
                rtEngine->launchRTMultiply()) {
              const int ell_width = rtEngine->ellWidthHint(fused_gate.num_mac);
              cuDoubleComplex* fused_gate_val;
              int* fused_gate_indices;
              checkCudaErrors(cudaMalloc((void**)&fused_gate_val, ell_width * nDim* sizeof(cuDoubleComplex)));
              checkCudaErrors(cudaMalloc((void**)&fused_gate_indices, ell_width * nDim* sizeof(int)));
              if (rtEngine->collectResultToELL(fused_gate_val, fused_gate_indices, ell_width, nDim)) {
                fused_gates_val_d.push_back(fused_gate_val);
                fused_gates_indices_d.push_back(fused_gate_indices);
                if (ell_width != fused_gate.num_mac) {
                  total_macs += (ell_width - fused_gate.num_mac);
                  fused_num_nonzero.back() = ell_width;
                }
                const auto& stats = rtEngine->lastStats();
                rt_h2d_ms += stats.h2d_ms;
                rt_dd_ms += stats.dd_ms;
                rt_gas_ms += stats.gas_ms;
                rt_launch_ms += stats.launch_ms;
                rt_ell_ms += stats.ell_ms;
                rt_count++;
                rt_done = true;
              } else {
                checkCudaErrors(cudaFree(fused_gate_val));
                checkCudaErrors(cudaFree(fused_gate_indices));
              }
            }
            if (!rt_done) {
              std::cout << "[WARN] RT Core path unavailable, falling back to DD-ELL" << std::endl;
            }
          }
          if (rt_done) {
            auto end_gate_convert = std::chrono::high_resolution_clock::now();
            std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end_gate_convert - begin_gate_convert).count()
            << " " << fused_gate.num_nodes << " " << fused_gate.num_edges << " " << qc->getNqubits() <<std::endl;
            continue;
          }
          const bool force_cpu = fused_gate.num_mac > MAX_DECODED_MACS;
          if (force_cpu) {
            std::cerr << "[WARN] Num of decoded MACs " << fused_gate.num_mac
                      << " exceeded limit " << MAX_DECODED_MACS
                      << ", using CPU path\n";
          }
          if (!force_cpu &&
              ((ddell_conversion == DDELL_Mixed && fused_gate.num_edges < conversion_edge_thresh) ||
               (ddell_conversion == DDELL_GPU))) {
            std::cout << "GPU" << std::endl;
            cuDoubleComplex *fused_gate_val;
            int *fused_gate_indices;
            dd::GPU_DD_edge* d_edge_arr;
            dd::GPU_DD_node* d_node_arr; 
            int num_edges = fused_gate.num_edges;
            int num_nodes = fused_gate.num_nodes;
            dd::GPU_DD_edge* h_edge_arr;
            dd::GPU_DD_node* h_node_arr;
            checkCudaErrors(cudaMallocHost((void**)&h_edge_arr, num_edges* sizeof(dd::GPU_DD_edge)));
            checkCudaErrors(cudaMallocHost((void**)&h_node_arr, num_nodes* sizeof(dd::GPU_DD_node)));
            // DFS GPU struct. construction
            QBatchSimulator<Config>::dd->DFS_fill_gpu_structure(fused_gate.fused_edge, h_edge_arr, h_node_arr);

            checkCudaErrors(cudaMalloc((void**)&d_edge_arr, num_edges* sizeof(dd::GPU_DD_edge)));
            checkCudaErrors(cudaMalloc((void**)&d_node_arr, num_nodes* sizeof(dd::GPU_DD_node)));
            checkCudaErrors(cudaMemcpy(d_edge_arr, h_edge_arr, num_edges* sizeof(dd::GPU_DD_edge), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_node_arr, h_node_arr, num_nodes* sizeof(dd::GPU_DD_node), cudaMemcpyHostToDevice));

            checkCudaErrors(cudaMalloc((void**)&fused_gate_val, fused_gate.num_mac * nDim* sizeof(cuDoubleComplex)));
            checkCudaErrors(cudaMalloc((void**)&fused_gate_indices, fused_gate.num_mac  *nDim* sizeof(int)));

            // kernel
            dd_extract_matrix<<<nDim, MAX_LEV>>>(
              d_edge_arr, d_node_arr, fused_gate_val, fused_gate_indices,
              num_nodes, num_edges,  fused_gate.num_mac, qc->getNqubits()
            );
            // dd_extract_matrix_warp<<<nDim/WARPS_PER_BLOCK, WARP_SIZE*WARPS_PER_BLOCK>>>(
            //   d_edge_arr, d_node_arr, fused_gate_val, fused_gate_indices,
            //   num_nodes, num_edges,  fused_gate.num_mac, qc->getNqubits()
            // );
            checkCudaErrors( cudaDeviceSynchronize() );
            fused_gates_val_d.push_back(fused_gate_val);
            fused_gates_indices_d.push_back(fused_gate_indices);
            checkCudaErrors(cudaFreeHost(h_edge_arr));
            checkCudaErrors(cudaFreeHost(h_node_arr));
            checkCudaErrors(cudaFree(d_edge_arr));
            checkCudaErrors(cudaFree(d_node_arr));
          }
          else {
            std::cout << "CPU" << std::endl;
            checkCudaErrors(cudaMallocHost((void**)&fused_gate_val_h, fused_gate.num_mac * nDim* sizeof(cuDoubleComplex)));
            checkCudaErrors(cudaMallocHost((void**)&fused_gate_indices_h, fused_gate.num_mac  *nDim* sizeof(int)));
            memset(sparse_idx_x, 0, nDim * sizeof(int));
            QBatchSimulator<Config>::dd->dd_extract_matrix_cpu(fused_gate.fused_edge, fused_gate_val_h, 
              fused_gate_indices_h, 0, 0, sparse_idx_x, fused_gate.num_mac, {1, 0});
            
            ////////////////////////////////////////////
            // new experiment: nzr uniform distribution
            // std::unordered_map<int, int> nzr_map;
            // for (size_t row_itr = 0; row_itr < nDim; row_itr++)
            // {
            //   int nzr = 0;
            //   for (size_t col_iter = 0; col_iter < fused_gate.num_mac; col_iter++)
            //   {
            //     if (fused_gate_val_h[row_itr * fused_gate.num_mac + col_iter].x != 0 || 
            //         fused_gate_val_h[row_itr * fused_gate.num_mac + col_iter].y != 0) {
            //       nzr++;
            //     }
            //   }
            //   if (nzr_map.find(nzr) != nzr_map.end()) {
            //     nzr_map[nzr] = nzr_map[nzr]+1;
            //   }
            //   else {
            //     nzr_map.insert({nzr, 1});
            //   }
            // }
            // for (auto nzr_pair : nzr_map)
            //   std::cout <<"  NZR Pair: " << nzr_pair.first << ", " << nzr_pair.second << '\n';

            ////////////////////////////////////////////
            cuDoubleComplex *fused_gate_val;
            int *fused_gate_indices;
            checkCudaErrors(cudaMalloc((void**)&fused_gate_val, fused_gate.num_mac * nDim* sizeof(cuDoubleComplex)));
            checkCudaErrors(cudaMalloc((void**)&fused_gate_indices, fused_gate.num_mac  *nDim* sizeof(int)));
            checkCudaErrors(cudaMemcpy(fused_gate_val, fused_gate_val_h, fused_gate.num_mac * nDim* sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(fused_gate_indices, fused_gate_indices_h, fused_gate.num_mac * nDim* sizeof(int), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaFreeHost(fused_gate_val_h));
            checkCudaErrors(cudaFreeHost(fused_gate_indices_h));
            fused_gates_val_d.push_back(fused_gate_val);
            fused_gates_indices_d.push_back(fused_gate_indices);
          }
          auto end_gate_convert = std::chrono::high_resolution_clock::now();
          std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end_gate_convert - begin_gate_convert).count()
          << " " << fused_gate.num_nodes << " " << fused_gate.num_edges << " " << qc->getNqubits() <<std::endl;
        }
        checkCudaErrors(cudaFreeHost(sparse_idx_x));
        auto end_convert = std::chrono::high_resolution_clock::now();
        auto stage2_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_convert - begin_convert).count();
        if (rt_h2d_ms > 0.0) {
          stage2_ms -= static_cast<long long>(rt_h2d_ms);
          if (stage2_ms < 0) {
            stage2_ms = 0;
          }
        }
        std::cout << "[Stage 2: DD-to-ELL Conversion] time: " << stage2_ms << std::endl;
        if (rt_count > 0) {
          std::cout << "  DD traversal: " << rt_dd_ms
                    << " ms, GAS build: " << rt_gas_ms
                    << " ms, optixLaunch: " << rt_launch_ms
                    << " ms, ELL reshape: " << rt_ell_ms
                    << " ms" << std::endl;
        }

        if (export_fused_gates == true)
        {
          std::ofstream outputFile("../../log/fused_gates/"+qc->getName()+".txt");
          std::vector<std::vector<int>> tgt_qubits_gates;
          std::vector<std::vector<dd::ComplexValue>> tensor_gates;
          for (int idx = 0; idx < fused_gates.size(); idx++) {
            printf("exporting fused gate %d to tensor\n", idx);
            qc::FusedGate fused_gate = fused_gates[idx];
            QBatchSimulator<Config>::dd->DD_matrix_extract(fused_gate.fused_edge, tgt_qubits_gates, tensor_gates);
          }
          outputFile << fused_gates.size() << "\n";
          for (int idx = 0; idx < fused_gates.size(); idx++) {
            printf("exporting fused gate tensor %d to file\n", idx);
            outputFile << tgt_qubits_gates[idx].size() << "\n";
            outputFile << tgt_qubits_gates[idx][0];
            for (int tgt_idx = 1; tgt_idx < tgt_qubits_gates[idx].size(); tgt_idx++) {
              outputFile << " " << tgt_qubits_gates[idx][tgt_idx];
            }
            outputFile << "\n";
            outputFile << tensor_gates[idx].size() << "\n";
            outputFile << tensor_gates[idx][0].r << " " << tensor_gates[idx][0].i;
            for (int ten_idx = 1; ten_idx < tensor_gates[idx].size(); ten_idx++) {
              outputFile << " " << tensor_gates[idx][ten_idx].r << " " << tensor_gates[idx][ten_idx].i;
            }
            outputFile << "\n";
          }
          outputFile.close();
        }
        
        ///////////////////////////////////////////
        printf("fused gates num. = %d\n", fused_num_nonzero.size());
        printf("total macs = %d\n", total_macs);

        }

        const bool use_cublas = envFlag("BQSIM_RT_DENSE_CUBLAS");
        bool any_dense = false;
        for (size_t i = 0; i < fused_gates_use_dense.size(); ++i) {
          if (fused_gates_use_dense[i]) {
            any_dense = true;
            break;
          }
        }
        if (use_cublas && any_dense) {
          const size_t bytes = nDim * batch_size * sizeof(cuDoubleComplex);
          cudaStream_t stream{};
          checkCudaErrors(cudaStreamCreate(&stream));
          cublasHandle_t handle{};
          if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
            checkCudaErrors(cudaStreamDestroy(stream));
            throw std::runtime_error("cublasCreate failed");
          }
          cublasSetStream(handle, stream);

          checkCudaErrors(cudaMemcpyAsync(d_batch[0], h_batch[0], bytes, cudaMemcpyHostToDevice, stream));
          checkCudaErrors(cudaStreamSynchronize(stream));
          cudaEvent_t compute_start{};
          cudaEvent_t compute_end{};
          checkCudaErrors(cudaEventCreate(&compute_start));
          checkCudaErrors(cudaEventCreate(&compute_end));
          checkCudaErrors(cudaEventRecord(compute_start, stream));
          int cur = 0;
          int next = 1;
          const int dense_grid = (nDim > 8192) ? 8192 : nDim;
          const int sparse_block = 256;
          const int warps_per_block = sparse_block / 32;
          const int batch_chunks = (batch_size + 31) / 32;
          const int total_warps = static_cast<int>(nDim) * batch_chunks;
          const int sparse_grid = (total_warps + warps_per_block - 1) / warps_per_block;
          dim3 sparse_block_size = dim3(sparse_block, 1, 1);
          const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
          const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);

          for (size_t g = 0; g < fused_num_nonzero.size(); ++g) {
            const bool use_dense =
                (g < fused_gates_use_dense.size() && fused_gates_use_dense[g] &&
                 g < fused_gates_dense_d.size() && fused_gates_dense_d[g] != nullptr);
            if (use_dense) {
              // Row-major C = A * B via column-major GEMM on transposed operands.
              const cublasStatus_t status = cublasZgemm(
                  handle,
                  CUBLAS_OP_N,
                  CUBLAS_OP_N,
                  batch_size,
                  static_cast<int>(nDim),
                  static_cast<int>(nDim),
                  &alpha,
                  d_batch[cur],
                  batch_size,
                  fused_gates_dense_d[g],
                  static_cast<int>(nDim),
                  &beta,
                  d_batch[next],
                  batch_size);
              if (status != CUBLAS_STATUS_SUCCESS) {
                cublasDestroy(handle);
                checkCudaErrors(cudaStreamDestroy(stream));
                throw std::runtime_error("cublasZgemm failed");
              }
            } else {
              run_fused_gate_warp<<<sparse_grid, sparse_block_size, 0, stream>>>(
                  fused_gates_val_d[g], fused_gates_indices_d[g], fused_num_nonzero[g],
                  d_batch[cur], d_batch[next], batch_size, nDim);
              checkCudaErrors(cudaGetLastError());
            }
            const int tmp = cur;
            cur = next;
            next = tmp;
          }

          checkCudaErrors(cudaEventRecord(compute_end, stream));
          checkCudaErrors(cudaEventSynchronize(compute_end));
          float compute_ms = 0.0f;
          checkCudaErrors(cudaEventElapsedTime(&compute_ms, compute_start, compute_end));
          checkCudaErrors(cudaEventDestroy(compute_start));
          checkCudaErrors(cudaEventDestroy(compute_end));
          checkCudaErrors(cudaMemcpyAsync(h_batch[1], d_batch[cur], bytes, cudaMemcpyDeviceToHost, stream));
          checkCudaErrors(cudaStreamSynchronize(stream));
          cublasDestroy(handle);
          checkCudaErrors(cudaStreamDestroy(stream));

          QBatchSimulator<Config>::final_state_idx = 1;
          current_buffer_idx = cur;
          QBatchSimulator<Config>::final_state_idx_gpu = current_buffer_idx;
          std::cout << "[Stage 3: ELL-based batch simulation] time: "
                    << static_cast<long long>(compute_ms)
                    << std::endl;
          return;
        }

        if (envFlag("BQSIM_RT_COMPACT_LAUNCH")) {
          const size_t bytes = nDim * batch_size * sizeof(cuDoubleComplex);
          cudaStream_t streams[2]{};
          checkCudaErrors(cudaStreamCreate(&streams[0]));
          checkCudaErrors(cudaStreamCreate(&streams[1]));
          const int dense_grid = (nDim > 8192)?8192:nDim;
          const int sparse_block = 256;
          const int warps_per_block = sparse_block / 32;
          const int batch_chunks = (batch_size + 31) / 32;
          const int total_warps = static_cast<int>(nDim) * batch_chunks;
          const int sparse_grid = (total_warps + warps_per_block - 1) / warps_per_block;
          dim3 dense_block_size = dim3(batch_size, 1, 1);
          dim3 sparse_block_size = dim3(sparse_block, 1, 1);
          const bool use_cuda_graph = envFlag("BQSIM_RT_USE_CUDA_GRAPH");
          const bool enable_mega = envFlag("BQSIM_RT_MEGA_KERNEL") && !use_cuda_graph;
          bool use_mega = enable_mega && !any_dense;
          cudaDeviceProp props{};
          int device_id = 0;
          checkCudaErrors(cudaGetDevice(&device_id));
          checkCudaErrors(cudaGetDeviceProperties(&props, device_id));
          const int gate_count = static_cast<int>(fused_num_nonzero.size());
          cuDoubleComplex** d_gate_vals = nullptr;
          int** d_gate_idx = nullptr;
          int* d_gate_nnz = nullptr;
          int coop_blocks = 0;
          cudaEvent_t compute_start[2]{};
          cudaEvent_t compute_end[2]{};
          bool compute_started[2]{false, false};
          checkCudaErrors(cudaEventCreate(&compute_start[0]));
          checkCudaErrors(cudaEventCreate(&compute_start[1]));
          checkCudaErrors(cudaEventCreate(&compute_end[0]));
          checkCudaErrors(cudaEventCreate(&compute_end[1]));
          int last_cur_per_stream[2]{0, 2};
          cudaGraph_t graphs[2]{};
          cudaGraphExec_t graph_execs[2]{};
          bool graph_ready[2]{false, false};
          if (use_mega) {
            int max_blocks_per_sm = 0;
            checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &max_blocks_per_sm,
                run_fused_gate_warp_mega,
                sparse_block,
                0));
            const int max_blocks = max_blocks_per_sm * props.multiProcessorCount;
            coop_blocks = sparse_grid;
            if (!props.cooperativeLaunch || coop_blocks > max_blocks) {
              use_mega = false;
              coop_blocks = 0;
              std::cerr << "[SPMSPM] Mega-kernel unavailable; fallback to per-gate launch." << std::endl;
            } else {
              const size_t ptr_bytes = static_cast<size_t>(gate_count) * sizeof(cuDoubleComplex*);
              const size_t idx_bytes = static_cast<size_t>(gate_count) * sizeof(int*);
              const size_t nnz_bytes = static_cast<size_t>(gate_count) * sizeof(int);
              checkCudaErrors(cudaMalloc(&d_gate_vals, ptr_bytes));
              checkCudaErrors(cudaMalloc(&d_gate_idx, idx_bytes));
              checkCudaErrors(cudaMalloc(&d_gate_nnz, nnz_bytes));
              checkCudaErrors(cudaMemcpy(d_gate_vals, fused_gates_val_d.data(), ptr_bytes, cudaMemcpyHostToDevice));
              checkCudaErrors(cudaMemcpy(d_gate_idx, fused_gates_indices_d.data(), idx_bytes, cudaMemcpyHostToDevice));
              checkCudaErrors(cudaMemcpy(d_gate_nnz, fused_num_nonzero.data(), nnz_bytes, cudaMemcpyHostToDevice));
            }
          }
          for (int batch_id = 0; batch_id < num_batch; ++batch_id) {
            const int stream_id = batch_id & 1;
            cudaStream_t stream = streams[stream_id];
            const int base = stream_id * 2;
            int cur = base;
            int next = base + 1;
            const bool last_for_stream = (batch_id + 2 >= num_batch);
            if (use_cuda_graph) {
              if (!graph_ready[stream_id]) {
                checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
                for (size_t g = 0; g < fused_num_nonzero.size(); ++g) {
                  const bool use_dense =
                      (g < fused_gates_use_dense.size() && fused_gates_use_dense[g] &&
                       g < fused_gates_dense_d.size() && fused_gates_dense_d[g] != nullptr);
                  if (use_dense) {
                    const int tile_k = std::max(1, dense_tile_k);
                    const size_t shared_bytes = static_cast<size_t>(tile_k) * sizeof(cuDoubleComplex);
                    run_dense_gate_tiled<<<dense_grid, dense_block_size, shared_bytes, stream>>>(
                        fused_gates_dense_d[g], d_batch[cur], d_batch[next], batch_size, nDim, tile_k, dense_assume_dense);
                  } else {
                    run_fused_gate_warp<<<sparse_grid, sparse_block_size, 0, stream>>>(
                        fused_gates_val_d[g], fused_gates_indices_d[g], fused_num_nonzero[g],
                        d_batch[cur], d_batch[next], batch_size, nDim);
                  }
                  const int tmp = cur;
                  cur = next;
                  next = tmp;
                }
                checkCudaErrors(cudaStreamEndCapture(stream, &graphs[stream_id]));
                checkCudaErrors(cudaGraphInstantiate(&graph_execs[stream_id], graphs[stream_id], nullptr, nullptr, 0));
                graph_ready[stream_id] = true;
                cur = base;
                next = base + 1;
              }
              checkCudaErrors(cudaMemcpyAsync(d_batch[base], h_batch[0], bytes, cudaMemcpyHostToDevice, stream));
              if (!compute_started[stream_id]) {
                checkCudaErrors(cudaEventRecord(compute_start[stream_id], stream));
                compute_started[stream_id] = true;
              }
              checkCudaErrors(cudaGraphLaunch(graph_execs[stream_id], stream));
              if (gate_count & 1) {
                const int tmp = cur;
                cur = next;
                next = tmp;
              }
              if (last_for_stream) {
                checkCudaErrors(cudaEventRecord(compute_end[stream_id], stream));
              }
            } else if (use_mega) {
              checkCudaErrors(cudaMemcpyAsync(d_batch[base], h_batch[0], bytes, cudaMemcpyHostToDevice, stream));
              if (!compute_started[stream_id]) {
                checkCudaErrors(cudaEventRecord(compute_start[stream_id], stream));
                compute_started[stream_id] = true;
              }
              if (coop_blocks > 0) {
                int gate_count_i = gate_count;
                int batch_size_i = batch_size;
                int nDim_i = static_cast<int>(nDim);
                void* args[] = {
                    &d_gate_vals,
                    &d_gate_idx,
                    &d_gate_nnz,
                    &gate_count_i,
                    &d_batch[cur],
                    &d_batch[next],
                    &batch_size_i,
                    &nDim_i};
                checkCudaErrors(cudaLaunchCooperativeKernel(
                    reinterpret_cast<void*>(run_fused_gate_warp_mega),
                    coop_blocks,
                    sparse_block,
                    args,
                    0,
                    stream));
                checkCudaErrors(cudaGetLastError());
                if (gate_count & 1) {
                  const int tmp = cur;
                  cur = next;
                  next = tmp;
                }
              }
              if (coop_blocks == 0) {
                for (size_t g = 0; g < fused_num_nonzero.size(); ++g) {
                  run_fused_gate_warp<<<sparse_grid, sparse_block_size, 0, stream>>>(
                      fused_gates_val_d[g], fused_gates_indices_d[g], fused_num_nonzero[g],
                      d_batch[cur], d_batch[next], batch_size, nDim);
                  checkCudaErrors(cudaGetLastError());
                  const int tmp = cur;
                  cur = next;
                  next = tmp;
                }
              }
              if (last_for_stream) {
                checkCudaErrors(cudaEventRecord(compute_end[stream_id], stream));
              }
            } else {
              checkCudaErrors(cudaMemcpyAsync(d_batch[base], h_batch[0], bytes, cudaMemcpyHostToDevice, stream));
              if (!compute_started[stream_id]) {
                checkCudaErrors(cudaEventRecord(compute_start[stream_id], stream));
                compute_started[stream_id] = true;
              }
              for (size_t g = 0; g < fused_num_nonzero.size(); ++g) {
                const bool use_dense =
                    (g < fused_gates_use_dense.size() && fused_gates_use_dense[g] &&
                     g < fused_gates_dense_d.size() && fused_gates_dense_d[g] != nullptr);
                if (use_dense) {
                  const int tile_k = std::max(1, dense_tile_k);
                  const size_t shared_bytes = static_cast<size_t>(tile_k) * sizeof(cuDoubleComplex);
                  run_dense_gate_tiled<<<dense_grid, dense_block_size, shared_bytes, stream>>>(
                      fused_gates_dense_d[g], d_batch[cur], d_batch[next], batch_size, nDim, tile_k, dense_assume_dense);
                } else {
                  run_fused_gate_warp<<<sparse_grid, sparse_block_size, 0, stream>>>(
                      fused_gates_val_d[g], fused_gates_indices_d[g], fused_num_nonzero[g],
                      d_batch[cur], d_batch[next], batch_size, nDim);
                }
                checkCudaErrors(cudaGetLastError());
                const int tmp = cur;
                cur = next;
                next = tmp;
              }
              if (last_for_stream) {
                checkCudaErrors(cudaEventRecord(compute_end[stream_id], stream));
              }
            }
            last_cur_per_stream[stream_id] = cur;
          }
          const int last_stream = (num_batch - 1) & 1;
          const int last_cur = last_cur_per_stream[last_stream];
          checkCudaErrors(cudaMemcpyAsync(h_batch[1], d_batch[last_cur], bytes, cudaMemcpyDeviceToHost, streams[last_stream]));
          if (d_gate_vals) {
            checkCudaErrors(cudaFree(d_gate_vals));
          }
          if (d_gate_idx) {
            checkCudaErrors(cudaFree(d_gate_idx));
          }
          if (d_gate_nnz) {
            checkCudaErrors(cudaFree(d_gate_nnz));
          }
          checkCudaErrors(cudaStreamSynchronize(streams[0]));
          checkCudaErrors(cudaStreamSynchronize(streams[1]));
          float ms0 = 0.0f;
          float ms1 = 0.0f;
          if (compute_started[0]) {
            checkCudaErrors(cudaEventElapsedTime(&ms0, compute_start[0], compute_end[0]));
          }
          if (compute_started[1]) {
            checkCudaErrors(cudaEventElapsedTime(&ms1, compute_start[1], compute_end[1]));
          }
          checkCudaErrors(cudaEventDestroy(compute_start[0]));
          checkCudaErrors(cudaEventDestroy(compute_start[1]));
          checkCudaErrors(cudaEventDestroy(compute_end[0]));
          checkCudaErrors(cudaEventDestroy(compute_end[1]));
          if (graph_ready[0]) {
            checkCudaErrors(cudaGraphExecDestroy(graph_execs[0]));
            checkCudaErrors(cudaGraphDestroy(graphs[0]));
          }
          if (graph_ready[1]) {
            checkCudaErrors(cudaGraphExecDestroy(graph_execs[1]));
            checkCudaErrors(cudaGraphDestroy(graphs[1]));
          }
          checkCudaErrors(cudaStreamDestroy(streams[0]));
          checkCudaErrors(cudaStreamDestroy(streams[1]));
          QBatchSimulator<Config>::final_state_idx = 1;
          current_buffer_idx = last_cur;
          QBatchSimulator<Config>::final_state_idx_gpu = current_buffer_idx;
          std::cout << "[Stage 3: ELL-based batch simulation] time: "
                    << static_cast<long long>(std::max(ms0, ms1))
                    << std::endl;
          return;
        }


        auto begin_sim = std::chrono::high_resolution_clock::now();
        const size_t host_batch_bytes = nDim * batch_size * sizeof(cuDoubleComplex);
        std::vector<cuDoubleComplex*> host_outputs(num_batch, nullptr);
        std::vector<uint8_t> host_outputs_pinned(num_batch, 0);
        for (int batch_id = 0; batch_id < num_batch; ++batch_id) {
          cuDoubleComplex* ptr = nullptr;
          if (cudaMallocHost((void**)&ptr, host_batch_bytes) == cudaSuccess) {
            host_outputs_pinned[batch_id] = 1;
          } else {
            ptr = static_cast<cuDoubleComplex*>(std::malloc(host_batch_bytes));
            if (!ptr) {
              for (int i = 0; i < batch_id; ++i) {
                if (host_outputs_pinned[i]) {
                  cudaFreeHost(host_outputs[i]);
                } else {
                  std::free(host_outputs[i]);
                }
              }
              throw std::bad_alloc();
            }
          }
          host_outputs[batch_id] = ptr;
        }
        tf::Taskflow taskflow("ELL-sim");
        tf::Executor executor;

        taskflow.emplace([&](){
          tf::cudaFlow cudaflow;
          std::vector<tf::cudaTask> input_copies;
          std::vector<tf::cudaTask> output_copies;
          std::vector<tf::cudaTask> simulate_fused_gate;
          std::vector<tf::cudaTask> gate_val_copies;
          std::vector<tf::cudaTask> gate_indices_copies;
          input_copies.reserve(num_batch);
          output_copies.reserve(num_batch);
          simulate_fused_gate.reserve(num_batch*fused_num_nonzero.size());
          // int grid_size = (nDim / (MAX_CUDA_THREADS_PER_BLOCK/batch_size) > 8192)?8192:(nDim/(MAX_CUDA_THREADS_PER_BLOCK/batch_size));
          // dim3 block_size = dim3(batch_size, MAX_CUDA_THREADS_PER_BLOCK/batch_size, 1);
          const int dense_grid = (nDim > 8192)?8192:nDim;
          const int sparse_block = 256;
          const int warps_per_block = sparse_block / 32;
          const int batch_chunks = (batch_size + 31) / 32;
          const int total_warps = static_cast<int>(nDim) * batch_chunks;
          const int sparse_grid = (total_warps + warps_per_block - 1) / warps_per_block;
          dim3 dense_block_size = dim3(batch_size, 1, 1);
          dim3 sparse_block_size = dim3(sparse_block, 1, 1);

          // Fill the graph nodes
          for (int batch_id = 0; batch_id < num_batch; batch_id++) {
            // input_copies.emplace_back(cudaflow.copy(
            //   d_batch[(batch_id%2)*2+(batch_id*(fused_num_nonzero.size()+1))%2], h_batch[0], nDim * batch_size
            // ).name("input_H2D_Host->"+std::to_string((batch_id*(fused_num_nonzero.size()+1))%2)));
            input_copies.emplace_back(cudaflow.copy(
              d_batch[(batch_id%2)*2+((batch_id/2)*(fused_num_nonzero.size()+1))%2], h_batch[0], nDim * batch_size
            ).name("input_H2D_Host->"+std::to_string((batch_id%2)*2+((batch_id/2)*(fused_num_nonzero.size()+1))%2)));

            for (size_t g = 0; g < fused_num_nonzero.size(); ++g) {
              const bool use_dense =
                  (g < fused_gates_use_dense.size() && fused_gates_use_dense[g] &&
                   g < fused_gates_dense_d.size() && fused_gates_dense_d[g] != nullptr);
              if (use_dense) {
                const int tile_k = std::max(1, dense_tile_k);
                const size_t shared_bytes = static_cast<size_t>(tile_k) * sizeof(cuDoubleComplex);
                simulate_fused_gate.emplace_back(cudaflow.kernel(
                  dense_grid,
                  dense_block_size,
                  shared_bytes,
                  run_dense_gate_tiled,
                  fused_gates_dense_d[g],
                  d_batch[(batch_id%2)*2+((batch_id/2)*(fused_num_nonzero.size()+1)+g)%2],
                  d_batch[(batch_id%2)*2+((batch_id/2)*(fused_num_nonzero.size()+1)+g+1)%2],
                  batch_size, nDim, tile_k, dense_assume_dense
                ).name("dense_gate_"+std::to_string(g)));
              } else {
                simulate_fused_gate.emplace_back(cudaflow.kernel(
                  sparse_grid,
                  sparse_block_size,
                  0,
                  run_fused_gate_warp,
                  fused_gates_val_d[g], fused_gates_indices_d[g], fused_num_nonzero[g],
                  d_batch[(batch_id%2)*2+((batch_id/2)*(fused_num_nonzero.size()+1)+g)%2], 
                  d_batch[(batch_id%2)*2+((batch_id/2)*(fused_num_nonzero.size()+1)+g+1)%2], batch_size, nDim
                ).name("fused_gate_"+std::to_string(g)));
              }
            }

            // output_copies.emplace_back(cudaflow.copy(
            //   h_batch[1], d_batch[(batch_id%2)*2+((batch_id+1)*fused_num_nonzero.size()+batch_id)%2], nDim * batch_size
            // ).name("output_D2H_"+std::to_string(((batch_id+1)*fused_num_nonzero.size()+batch_id)%2)+"->Host"));
            output_copies.emplace_back(cudaflow.copy(
              host_outputs[batch_id], d_batch[(batch_id%2)*2+((batch_id/2)*(fused_num_nonzero.size()+1)+fused_num_nonzero.size())%2], nDim * batch_size
            ).name("output_D2H_"+std::to_string((batch_id%2)*2+((batch_id/2)*(fused_num_nonzero.size()+1)+fused_num_nonzero.size())%2)+"->Host"));
          }

          // Dependencies
          for (int batch_id = 0; batch_id < num_batch; batch_id++) {
            // dependencies between H2D and the kernels
            input_copies[batch_id].precede(simulate_fused_gate[batch_id*fused_num_nonzero.size()]);
            if (batch_id > 1) {
              simulate_fused_gate[(batch_id-1)*fused_num_nonzero.size()-1].precede(input_copies[batch_id]);
            }

            // dependencies within the kernels
            if (batch_id > 0) {
              simulate_fused_gate[batch_id*fused_num_nonzero.size()-1].precede(simulate_fused_gate[batch_id*fused_num_nonzero.size()]);
            }
            for (size_t g = 1; g < fused_num_nonzero.size(); ++g) {
              simulate_fused_gate[batch_id*fused_num_nonzero.size()+g-1].precede(simulate_fused_gate[batch_id*fused_num_nonzero.size()+g]);
            }

            // dependencies between D2H and the kernels
            simulate_fused_gate[(batch_id+1)*fused_num_nonzero.size()-1].precede(output_copies[batch_id]);
            if (batch_id < num_batch-2) {
              output_copies[batch_id].precede(simulate_fused_gate[(batch_id+2)*fused_num_nonzero.size()]);
            }
          }
          
          // else {
          //   // Fill the graph nodes
          //   for (int batch_id = 0; batch_id < num_batch; batch_id++) {
          //     input_copies.emplace_back(cudaflow.copy(
          //       d_batch[(batch_id%2)*2+0], h_batch[0], nDim * batch_size
          //     ).name("input_H2D_0->"+std::to_string((batch_id%2)*2+0)));
          //     for (opNum = 0; opNum < gpu_full_at; opNum++) {
          //       simulate_fused_gate.emplace_back(cudaflow.kernel(
          //         grid_size,
          //         block_size,
          //         0,
          //         run_fused_gate,
          //         fused_gates_val_d[opNum], fused_gates_indices_d[opNum], fused_num_nonzero[opNum],
          //         d_batch[(batch_id%2)*2+opNum%2], d_batch[(batch_id%2)*2+(opNum+1)%2], batch_size, nDim
          //       ).name("fused_gate_"+std::to_string(opNum)));
          //     }
          //     for (opNum = gpu_full_at; opNum < fused_num_nonzero.size(); opNum++) {
          //       gate_val_copies.push_back(cudaflow.copy(
          //         fused_gates_val_mored[(opNum-gpu_full_at)%2], fused_gates_val_moreh[(opNum-gpu_full_at)], nDim * fused_num_nonzero[opNum]
          //       ).name("gate_val_H2D"));
          //       gate_indices_copies.push_back(cudaflow.copy(
          //         fused_gates_indices_mored[(opNum-gpu_full_at)%2], fused_gates_indices_moreh[(opNum-gpu_full_at)], nDim * fused_num_nonzero[opNum]
          //       ).name("gate_indices_H2D"));
          //       simulate_fused_gate.emplace_back(cudaflow.kernel(
          //         grid_size,
          //         block_size,
          //         0,
          //         run_fused_gate,
          //         fused_gates_val_mored[(opNum-gpu_full_at)%2], fused_gates_indices_mored[(opNum-gpu_full_at)%2], fused_num_nonzero[opNum],
          //         d_batch[(batch_id%2)*2+opNum%2], d_batch[(batch_id%2)*2+(opNum+1)%2], batch_size, nDim
          //       ).name("fused_gate_"+std::to_string(opNum)));
          //     }

          //     output_copies.emplace_back(cudaflow.copy(
          //       h_batch[1], d_batch[(batch_id%2)*2+opNum%2], nDim * batch_size
          //     ).name("output_D2H_"+std::to_string((batch_id%2)*2+opNum%2)+"->1"));
          //   }

          //   // Dependencies
          //   for (int batch_id = 0; batch_id < num_batch; batch_id++) {
          //     // dependencies between H2D and the kernels
          //     input_copies[batch_id].precede(simulate_fused_gate[batch_id*fused_num_nonzero.size()]);
          //     if (batch_id > 1) {
          //       simulate_fused_gate[(batch_id-1)*fused_num_nonzero.size()-1].precede(input_copies[batch_id]);
          //     }

          //     // // dependencies within the kernels
          //     if (batch_id > 0) {
          //       simulate_fused_gate[batch_id*fused_num_nonzero.size()-1].precede(simulate_fused_gate[batch_id*fused_num_nonzero.size()]);
          //       simulate_fused_gate[batch_id*fused_num_nonzero.size()-2].precede(gate_val_copies[batch_id*(fused_num_nonzero.size()-gpu_full_at)]);
          //       simulate_fused_gate[batch_id*fused_num_nonzero.size()-1].precede(gate_val_copies[batch_id*(fused_num_nonzero.size()-gpu_full_at)+1]);
          //       simulate_fused_gate[batch_id*fused_num_nonzero.size()-2].precede(gate_indices_copies[batch_id*(fused_num_nonzero.size()-gpu_full_at)]);
          //       simulate_fused_gate[batch_id*fused_num_nonzero.size()-1].precede(gate_indices_copies[batch_id*(fused_num_nonzero.size()-gpu_full_at)+1]);
          //     }

          //     for (opNum = 1; opNum < gpu_full_at; opNum++) {
          //       simulate_fused_gate[batch_id*fused_num_nonzero.size()+opNum-1].precede(simulate_fused_gate[batch_id*fused_num_nonzero.size()+opNum]);
          //     }
              
          //     for (opNum = gpu_full_at; opNum < fused_num_nonzero.size(); opNum++) {
          //       gate_val_copies[batch_id*(fused_num_nonzero.size()-gpu_full_at)+opNum-gpu_full_at].precede(simulate_fused_gate[batch_id*fused_num_nonzero.size()+opNum]);
          //       gate_indices_copies[batch_id*(fused_num_nonzero.size()-gpu_full_at)+opNum-gpu_full_at].precede(simulate_fused_gate[batch_id*fused_num_nonzero.size()+opNum]);
          //       if (opNum - gpu_full_at > 1) {
          //         simulate_fused_gate[batch_id*fused_num_nonzero.size()+opNum-2].precede(gate_val_copies[batch_id*(fused_num_nonzero.size()-gpu_full_at)+opNum-gpu_full_at]);
          //         simulate_fused_gate[batch_id*fused_num_nonzero.size()+opNum-2].precede(gate_indices_copies[batch_id*(fused_num_nonzero.size()-gpu_full_at)+opNum-gpu_full_at]);
          //       }
          //       simulate_fused_gate[batch_id*fused_num_nonzero.size()+opNum-1].precede(simulate_fused_gate[batch_id*fused_num_nonzero.size()+opNum]);
          //     }

          //     // dependencies between D2H and the kernels
          //     simulate_fused_gate[(batch_id+1)*fused_num_nonzero.size()-1].precede(output_copies[batch_id]);
          //     if (batch_id < num_batch-2) {
          //       output_copies[batch_id].precede(simulate_fused_gate[(batch_id+2)*fused_num_nonzero.size()]);
          //     }
          //   }
          // }
 
          tf::cudaStream stream;
          cudaflow.run(stream);
          stream.synchronize(); 
          // cudaflow.dump(std::cout); 
        });

        executor.run(taskflow).wait();

        if (num_batch > 0 && host_outputs[num_batch - 1]) {
          std::memcpy(h_batch[1], host_outputs[num_batch - 1], host_batch_bytes);
        }
        for (int batch_id = 0; batch_id < num_batch; ++batch_id) {
          if (!host_outputs[batch_id]) {
            continue;
          }
          if (host_outputs_pinned[batch_id]) {
            cudaFreeHost(host_outputs[batch_id]);
          } else {
            std::free(host_outputs[batch_id]);
          }
        }
        QBatchSimulator<Config>::final_state_idx = 1;
        const int last_stream = (num_batch - 1) & 1;
        int cur = last_stream * 2;
        int next = cur + 1;
        for (size_t g = 0; g < fused_num_nonzero.size(); ++g) {
          const int tmp = cur;
          cur = next;
          next = tmp;
        }
        current_buffer_idx = cur;
        QBatchSimulator<Config>::final_state_idx_gpu = current_buffer_idx;
        auto end_sim = std::chrono::high_resolution_clock::now();
        std::cout << "[Stage 3: ELL-based batch simulation] time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_sim - begin_sim).count() << std::endl;
    }

    [[nodiscard]]
    cuDoubleComplex* getVector() const {
        if (getNumberOfQubits() >= MAX_LEV) {
            // On 64bit system the vector can hold up to (2^60)-1 elements, if memory permits
            throw std::range_error("getVector only supports less than 60 qubits.");
        }
        return h_batch[final_state_idx];
    }

    [[nodiscard]] std::size_t getNumberOfQubits() const { return qc->getNqubits(); };

    [[nodiscard]] std::size_t getNumberOfOps() const { return qc->getNops(); };

    [[nodiscard]] std::string getName() const { return qc->getName(); };

    std::unique_ptr<dd::Package<Config>>     dd  = std::make_unique<dd::Package<Config>>();
    std::vector<cuDoubleComplex *> h_batch, d_batch;
    std::vector<uint8_t> h_batch_pinned;

    int                                     final_state_idx;
    int        final_state_idx_gpu;

    unsigned int                            fuse = 0;
    size_t nDim = 1;
    bool export_fused_gates = false;
    int ddell_conversion = 2;
    int conversion_edge_thresh = 2000;
    std::unique_ptr<RTSpMSpMEngine> rtEngine;
    int rt_threshold = 2000;
    int rt_qubit_threshold = 12;

    bool buildGatePrimitives(std::vector<qc::GatePrimitive>& out) const {
      out.clear();
      if (!qc) {
        return false;
      }
      auto set_matrix2 = [](qc::GatePrimitive& gp,
                            double a00, double b00,
                            double a01, double b01,
                            double a10, double b10,
                            double a11, double b11) {
        gp.matrix_dim = 2;
        gp.matrix[0] = make_float2(static_cast<float>(a00), static_cast<float>(b00));
        gp.matrix[1] = make_float2(static_cast<float>(a01), static_cast<float>(b01));
        gp.matrix[2] = make_float2(static_cast<float>(a10), static_cast<float>(b10));
        gp.matrix[3] = make_float2(static_cast<float>(a11), static_cast<float>(b11));
      };

      for (const auto& op : *qc) {
        if (!op->isUnitary()) {
          return false;
        }
        const auto type = op->getType();
        if (type == qc::Barrier) {
          continue;
        }

        qc::GatePrimitive gp{};
        gp.gate_type = static_cast<int>(type);
        gp.target_count = static_cast<int>(op->getTargets().size());
        gp.control_count = static_cast<int>(op->getControls().size());
        gp.is_controlled = gp.control_count > 0;
        if (gp.target_count <= 0 || gp.target_count > qc::MAX_TARGETS) {
          return false;
        }
        if (gp.control_count > qc::MAX_CONTROLS) {
          return false;
        }

        int ti = 0;
        for (auto t : op->getTargets()) {
          gp.targets[ti++] = static_cast<int>(t);
        }
        int ci = 0;
        for (const auto& c : op->getControls()) {
          if (c.type != qc::Control::Type::Pos) {
            return false;
          }
          gp.controls[ci++] = static_cast<int>(c.qubit);
        }

        if (gp.control_count > 0) {
          if (gp.target_count != 1) {
            return false;
          }
          if (type != qc::X && type != qc::Z) {
            return false;
          }
          if (type == qc::X) {
            set_matrix2(gp, 0, 0, 1, 0, 1, 0, 0, 0);
          } else {
            set_matrix2(gp, 1, 0, 0, 0, 0, 0, -1, 0);
          }
          out.push_back(gp);
          continue;
        }

        if (gp.target_count != 1) {
          return false;
        }

        const auto& params = op->getParameter();
        switch (type) {
          case qc::X:
            set_matrix2(gp, 0, 0, 1, 0, 1, 0, 0, 0);
            break;
          case qc::H: {
            const double inv = 1.0 / std::sqrt(2.0);
            set_matrix2(gp, inv, 0, inv, 0, inv, 0, -inv, 0);
            break;
          }
          case qc::Z:
            set_matrix2(gp, 1, 0, 0, 0, 0, 0, -1, 0);
            break;
          case qc::S:
            set_matrix2(gp, 1, 0, 0, 0, 0, 0, 0, 1);
            break;
          case qc::T: {
            const double angle = qc::PI_4;
            set_matrix2(gp, 1, 0, 0, 0, 0, 0, std::cos(angle), std::sin(angle));
            break;
          }
          case qc::RX: {
            const double theta = params.empty() ? 0.0 : params[0];
            const double c = std::cos(theta / 2.0);
            const double s = std::sin(theta / 2.0);
            set_matrix2(gp, c, 0, 0, -s, 0, -s, c, 0);
            break;
          }
          case qc::RY: {
            const double theta = params.empty() ? 0.0 : params[0];
            const double c = std::cos(theta / 2.0);
            const double s = std::sin(theta / 2.0);
            set_matrix2(gp, c, 0, -s, 0, s, 0, c, 0);
            break;
          }
          case qc::RZ: {
            const double theta = params.empty() ? 0.0 : params[0];
            const double c = std::cos(theta / 2.0);
            const double s = std::sin(theta / 2.0);
            set_matrix2(gp, c, -s, 0, 0, 0, 0, c, s);
            break;
          }
          default:
            return false;
        }
        out.push_back(gp);
      }

      return !out.empty();
    }

protected:
    dd::fp        epsilon = 0.001;
    std::unique_ptr<qc::QuantumComputation> qc;
    std::size_t                             singleShots{0};
    int batch_size = 1;
    int num_batch = 1;
    int gpu_full_at = -1;
    std::vector<cuDoubleComplex*> fused_gates_val_d;
    std::vector<int*> fused_gates_indices_d;
    std::vector<cuDoubleComplex*> fused_gates_dense_d;
    std::vector<uint8_t> fused_gates_use_dense;


    // std::vector<cuDoubleComplex*> fused_gates_val_moreh;
    // std::vector<int*> fused_gates_indices_moreh;
    // std::vector<cuDoubleComplex*> fused_gates_val_mored;
    // std::vector<int*> fused_gates_indices_mored;

    // 
};

template class QBatchSimulator<dd::DDPackageConfig>;

#endif //QBATCH_SIMULATOR_H
