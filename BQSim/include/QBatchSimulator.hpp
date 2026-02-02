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
#include <limits>
#include <cublas_v2.h>
#include <cusparse.h>
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

__global__ void replicate(cuComplex *input_arr_d, int N) {
  input_arr_d[threadIdx.x+blockIdx.x*N] = input_arr_d[blockIdx.x*N];
}

__global__ void initial_check(cuComplex *input_arr_d, bool *identical, int N) {
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
  cuComplex *fused_gate_val,
  int *fused_gate_indices,
  int num_nodes,
  int num_edges,
  int num_non_zeros,
  int num_qubits
) {
  __shared__ int decoded_locs[MAX_DECODED_MACS];
  __shared__ cuComplex decoded_factors[MAX_DECODED_MACS];
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
    cuComplex rec_factor = {1.0f, 0.0f};
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
        const cuComplex edge_w = dd_edges[edge_ptr].w;
        decoded_factors[decode_ptr[0]] = cuCmulf(rec_factor, edge_w);
        stack_ptr--; decode_ptr[0]++;
        continue;
      }

      int child_idx = (int)(left_or_right[stack_ptr]) + (int)(up_or_down[stack_ptr]) * 2;
      // return or move forward
      if (left_or_right[stack_ptr] == 2) {
        left_or_right[stack_ptr] = 0;
        const cuComplex edge_w = dd_edges[edge_ptr].w;
        rec_factor = cuCdivf(rec_factor, edge_w);
        rec_loc -= (1 << dd_nodes[node_ptr].qubit);
        stack_ptr--;
      }
      else {
        left_or_right[stack_ptr]++;
        if (left_or_right[stack_ptr] == 1) {
          const cuComplex edge_w = dd_edges[edge_ptr].w;
          rec_factor = cuCmulf(rec_factor, edge_w);
        }
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
  cuComplex *fused_gate_val,
  int *fused_gate_indices,
  int num_nodes,
  int num_edges,
  int num_non_zeros,
  int num_qubits
) {
  __shared__ int decoded_locs[MAX_DECODED_MACS*WARPS_PER_BLOCK];
  __shared__ cuComplex decoded_factors[MAX_DECODED_MACS*WARPS_PER_BLOCK];
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
    cuComplex rec_factor = {1.0f, 0.0f};
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
        const cuComplex edge_w = dd_edges[edge_ptr].w;
        decoded_factors[MAX_DECODED_MACS*(tid/WARP_SIZE)+decode_ptr[tid/WARP_SIZE]] = cuCmulf(rec_factor, edge_w);
        stack_ptr--; decode_ptr[tid/WARP_SIZE]++;
        continue;
      }

      int child_idx = (int)(left_or_right[MAX_LEV*(tid/WARP_SIZE) + stack_ptr]) + (int)(up_or_down[MAX_LEV*(tid/WARP_SIZE) + stack_ptr]) * 2;
      // return or move forward
      if (left_or_right[MAX_LEV*(tid/WARP_SIZE) + stack_ptr] == 2) {
        left_or_right[MAX_LEV*(tid/WARP_SIZE) + stack_ptr] = 0;
        const cuComplex edge_w = dd_edges[edge_ptr].w;
        rec_factor = cuCdivf(rec_factor, edge_w);
        rec_loc -= (1 << dd_nodes[node_ptr].qubit);
        stack_ptr--;
      }
      else {
        left_or_right[MAX_LEV*(tid/WARP_SIZE) + stack_ptr]++;
        if (left_or_right[MAX_LEV*(tid/WARP_SIZE) + stack_ptr] == 1) {
          const cuComplex edge_w = dd_edges[edge_ptr].w;
          rec_factor = cuCmulf(rec_factor, edge_w);
        }
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
//   cuComplex *gates_val,
//   int *gates_indices,
//   int num_non_zero,
//   cuComplex *input_state,
//   cuComplex *output_state,
//   int batch_size, 
//   int nDim
// ) {
  
//   int tidx = threadIdx.x;
//   int tidy = threadIdx.y;
//   int tid = tidx+tidy*blockDim.y;
//   int rounds = nDim / (gridDim.x*blockDim.y);
//   int bid = blockIdx.x;
//   __shared__ int share_indices[MAX_VAL];
//   __shared__ cuComplex shared_val[MAX_VAL];


//   for (int i = 0; i < rounds; i++) {
//     if (tid < num_non_zero * blockDim.y) {
//       share_indices[tid] = gates_indices[((rounds * bid+i)*blockDim.y) * num_non_zero + tid];
//       shared_val[tid] = gates_val[((rounds * bid+i)*blockDim.y) * num_non_zero + tid];
//     }
//     __syncthreads();
//     cuComplex result_value = {0, 0};
//     for (int j = 0; j < num_non_zero; j++) {
//       cuComplex temp_value = cuCmulf(input_state[share_indices[tidy*num_non_zero+j]*batch_size+tidx], shared_val[tidy*num_non_zero+j]);
//       result_value = cuCaddf(result_value, temp_value);
//     }
//     __syncthreads();
//     output_state[((rounds * bid+i)*blockDim.y+tidy)*batch_size +tidx] = result_value;
//   }
//   __syncthreads();
// }

__global__ void run_fused_gate_warp(
  const cuComplex* __restrict__ gates_val,
  const int* __restrict__ gates_indices,
  int num_non_zero,
  const cuComplex* __restrict__ input_state,
  cuComplex* __restrict__ output_state,
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
  cuComplex acc = {0.0f, 0.0f};
  for (int j = 0; j < num_non_zero; ++j) {
    int col = 0;
    float vx = 0.0f;
    float vy = 0.0f;
    if (lane == 0) {
      col = __ldg(gates_indices + row_off + j);
      const float2 v = __ldg(reinterpret_cast<const float2*>(gates_val + row_off + j));
      vx = v.x;
      vy = v.y;
    }
    col = __shfl_sync(0xFFFFFFFF, col, 0);
    vx = __shfl_sync(0xFFFFFFFF, vx, 0);
    vy = __shfl_sync(0xFFFFFFFF, vy, 0);
    if (vx != 0.0 || vy != 0.0) {
      const size_t b_idx = static_cast<size_t>(col) * batch_size + batch;
      const float2 b2 = __ldg(reinterpret_cast<const float2*>(input_state + b_idx));
      const cuComplex b = make_cuComplex(b2.x, b2.y);
      const cuComplex a = make_cuComplex(vx, vy);
      acc = cuCaddf(acc, cuCmulf(a, b));
    }
  }
  output_state[static_cast<size_t>(row) * batch_size + batch] = acc;
}

__global__ void run_fused_gate_warp_mega(
  const cuComplex* const* __restrict__ gates_val_list,
  const int* const* __restrict__ gates_idx_list,
  const int* __restrict__ nnz_list,
  int gate_count,
  const cuComplex* __restrict__ state_a,
  cuComplex* __restrict__ state_b,
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
    const cuComplex* gates_val = gates_val_list[g];
    const int* gates_indices = gates_idx_list[g];
    const int num_non_zero = nnz_list[g];
    const cuComplex* input_state = (g & 1) ? state_b : state_a;
    cuComplex* output_state = (g & 1) ? const_cast<cuComplex*>(state_a) : state_b;
    const size_t row_off = static_cast<size_t>(row) * static_cast<size_t>(num_non_zero);
    cuComplex acc = {0.0f, 0.0f};
    for (int j = 0; j < num_non_zero; ++j) {
      int col = 0;
      float vx = 0.0f;
      float vy = 0.0f;
      if (lane == 0) {
        col = __ldg(gates_indices + row_off + j);
        const float2 v = __ldg(reinterpret_cast<const float2*>(gates_val + row_off + j));
        vx = v.x;
        vy = v.y;
      }
      col = __shfl_sync(0xFFFFFFFF, col, 0);
      vx = __shfl_sync(0xFFFFFFFF, vx, 0);
      vy = __shfl_sync(0xFFFFFFFF, vy, 0);
      if (vx != 0.0 || vy != 0.0) {
        const size_t b_idx = static_cast<size_t>(col) * batch_size + batch;
        const float2 b2 = __ldg(reinterpret_cast<const float2*>(input_state + b_idx));
        const cuComplex b = make_cuComplex(b2.x, b2.y);
        const cuComplex a = make_cuComplex(vx, vy);
        acc = cuCaddf(acc, cuCmulf(a, b));
      }
    }
    output_state[static_cast<size_t>(row) * batch_size + batch] = acc;
    grid.sync();
  }
}

__global__ void ell_to_dense(
  const cuComplex* ell_vals,
  const int* ell_indices,
  cuComplex* dense,
  int ell_width,
  int nDim
) {
  const int row = blockIdx.y;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= nDim || idx >= ell_width) {
    return;
  }
  const int col = ell_indices[row * ell_width + idx];
  const cuComplex val = ell_vals[row * ell_width + idx];
  if (val.x != 0.0 || val.y != 0.0) {
    dense[static_cast<size_t>(row) * static_cast<size_t>(nDim) + col] = val;
  }
}


template<class Config = dd::DDPackageConfig>
class QBatchSimulator {
public:
    explicit QBatchSimulator(std::unique_ptr<qc::QuantumComputation>&& qc_, int batch_size_, int num_batch_) : 
    qc(std::move(qc_)), batch_size(batch_size_), num_batch(num_batch_), rtEngine(std::make_unique<RTSpMSpMEngine>())
    {
        checkCudaErrors(cudaFree(0));
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
        QBatchSimulator<Config>::dd->resize(qc->getNqubits());
        const auto nQubits = qc->getNqubits();
        nDim    = std::pow(2, nQubits);
        
        cuComplex *h_batch0;
        cuComplex *h_batch1;
        const size_t host_bytes = nDim * batch_size_ * sizeof(cuComplex);
        checkCudaErrors(cudaMallocHost((void**)&h_batch0, host_bytes));
        checkCudaErrors(cudaMallocHost((void**)&h_batch1, host_bytes));
        const bool pinned0 = true;
        const bool pinned1 = true;

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
            h_batch0[amp_id*batch_size_] = {static_cast<float>(real), static_cast<float>(imag)};
            amp_id++;
            }
        }
        file.close();

        cuComplex *input_d;
        checkCudaErrors(cudaMalloc((void**)&input_d, nDim * batch_size_ * sizeof(cuComplex)));
        checkCudaErrors(cudaMemcpy(input_d, h_batch0, nDim * batch_size_ * sizeof(cuComplex),
                cudaMemcpyHostToDevice));
        replicate<<<nDim, batch_size>>>(input_d, batch_size_);
        checkCudaErrors(cudaMemcpy(h_batch0, input_d, nDim * batch_size_ * sizeof(cuComplex),
                cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(input_d));
        
        memset(h_batch1, 0, nDim * batch_size_ * sizeof(cuComplex));
        h_batch.push_back(h_batch0);
        h_batch_pinned.push_back(static_cast<uint8_t>(pinned0));
        h_batch.push_back(h_batch1);
        h_batch_pinned.push_back(static_cast<uint8_t>(pinned1));

        for (int buf = 0; buf < 4; buf++) {
          cuComplex *d_batch_buf;
          checkCudaErrors(cudaMalloc((void**)&d_batch_buf, nDim * batch_size_ * sizeof(cuComplex)));
          d_batch.push_back(d_batch_buf);
        }

        if (cublasCreate(&cublas_handles[0]) != CUBLAS_STATUS_SUCCESS ||
            cublasCreate(&cublas_handles[1]) != CUBLAS_STATUS_SUCCESS) {
          throw std::runtime_error("cublasCreate failed");
        }
        checkCudaErrors(cudaStreamCreate(&cublas_stream));
        checkCudaErrors(cudaStreamCreate(&cublas_stream_aux));
        cublasSetMathMode(cublas_handles[0], CUBLAS_TF32_TENSOR_OP_MATH);
        cublasSetMathMode(cublas_handles[1], CUBLAS_TF32_TENSOR_OP_MATH);
        cublasSetStream(cublas_handles[0], cublas_stream);
        cublasSetStream(cublas_handles[1], cublas_stream_aux);
        cublas_ready = true;
        if (cusparseCreate(&cusparse_handle) != CUSPARSE_STATUS_SUCCESS) {
          throw std::runtime_error("cusparseCreate failed");
        }
        cusparse_ready = true;
        
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
      for (size_t i = 0; i < fused_gates_csr_row_offsets_d.size(); ++i) {
        if (fused_gates_csr_row_offsets_d[i]) {
          checkCudaErrors(cudaFree(fused_gates_csr_row_offsets_d[i]));
        }
        if (i < fused_gates_csr_col_indices_d.size() && fused_gates_csr_col_indices_d[i]) {
          checkCudaErrors(cudaFree(fused_gates_csr_col_indices_d[i]));
        }
        if (i < fused_gates_csr_values_d.size() && fused_gates_csr_values_d[i]) {
          checkCudaErrors(cudaFree(fused_gates_csr_values_d[i]));
        }
      }
      if (cublas_ready) {
        checkCudaErrors(cudaStreamDestroy(cublas_stream));
        checkCudaErrors(cudaStreamDestroy(cublas_stream_aux));
        cublasDestroy(cublas_handles[0]);
        cublasDestroy(cublas_handles[1]);
        cublas_ready = false;
      }
      if (cusparse_ready) {
        cusparseDestroy(cusparse_handle);
        cusparse_ready = false;
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
          bool enable_cusparse = envFlag("BQSIM_RT_CUSPARSE_TENSOR");
          bool conversion_allowed = enable_cusparse;

          const double dense_threshold = envDouble("BQSIM_RT_DENSE_THRESHOLD", 0.01);
          const uint64_t dense_max_bytes =
              envUInt64("BQSIM_RT_DENSE_MAX_BYTES", 512ULL * 1024ULL * 1024ULL);
          (void)dense_max_bytes;
          if (conversion_allowed && (batch_size % 32 != 0)) {
            std::cerr << "[SPMSPM] Dense path: batch_size not multiple of 32; expect lower memory coalescing." << std::endl;
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
              if (i < fused_gates_csr_row_offsets_d.size() && fused_gates_csr_row_offsets_d[i]) {
                cudaFree(fused_gates_csr_row_offsets_d[i]);
              }
              if (i < fused_gates_csr_col_indices_d.size() && fused_gates_csr_col_indices_d[i]) {
                cudaFree(fused_gates_csr_col_indices_d[i]);
              }
              if (i < fused_gates_csr_values_d.size() && fused_gates_csr_values_d[i]) {
                cudaFree(fused_gates_csr_values_d[i]);
              }
            }
            fused_gates_val_d.clear();
            fused_gates_indices_d.clear();
            fused_gates_dense_d.clear();
            fused_gates_csr_row_offsets_d.clear();
            fused_gates_csr_col_indices_d.clear();
            fused_gates_csr_values_d.clear();
            fused_gates_csr_nnz.clear();
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
            cuComplex* fused_gate_val = nullptr;
            int* fused_gate_indices = nullptr;
            if (cudaMalloc((void**)&fused_gate_val, ell_width * nDim * sizeof(cuComplex)) != cudaSuccess ||
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
              fused_gates_csr_row_offsets_d.push_back(nullptr);
              fused_gates_csr_col_indices_d.push_back(nullptr);
              fused_gates_csr_values_d.push_back(nullptr);
              fused_gates_csr_nnz.push_back(0);
              if (conversion_allowed && rtEngine->densityEstimate() >= dense_threshold) {
                printf("[DEBUG] Gate #%lu: CONVERTING to CSR (Density: %.6f >= %.6f)\n", 
                fused_num_nonzero.size(), rtEngine->densityEstimate(), dense_threshold);
                const size_t ell_elems = static_cast<size_t>(ell_width) * static_cast<size_t>(nDim);
                std::vector<cuComplex> ell_vals(ell_elems);
                std::vector<int> ell_cols(ell_elems);
                checkCudaErrors(cudaMemcpy(ell_vals.data(), fused_gate_val,
                                           ell_elems * sizeof(cuComplex),
                                           cudaMemcpyDeviceToHost));
                checkCudaErrors(cudaMemcpy(ell_cols.data(), fused_gate_indices,
                                           ell_elems * sizeof(int),
                                           cudaMemcpyDeviceToHost));

                std::vector<int> row_offsets(static_cast<size_t>(nDim) + 1, 0);
                size_t nnz = 0;
                for (size_t row = 0; row < static_cast<size_t>(nDim); ++row) {
                  int count = 0;
                  const size_t row_off = row * static_cast<size_t>(ell_width);
                  for (int j = 0; j < ell_width; ++j) {
                    const cuComplex v = ell_vals[row_off + static_cast<size_t>(j)];
                    if (v.x != 0.0f || v.y != 0.0f) {
                      ++count;
                    }
                  }
                  nnz += static_cast<size_t>(count);
                  row_offsets[row + 1] = static_cast<int>(nnz);
                }

                if (nnz > static_cast<size_t>(std::numeric_limits<int>::max())) {
                  std::cerr << "[SPMSPM] CSR nnz overflow; skipping CSR path" << std::endl;
                } else {
                  std::vector<int> csr_cols(nnz);
                  std::vector<cuComplex> csr_vals(nnz);
                  size_t pos = 0;
                  for (size_t row = 0; row < static_cast<size_t>(nDim); ++row) {
                    const size_t row_off = row * static_cast<size_t>(ell_width);
                    for (int j = 0; j < ell_width; ++j) {
                      const size_t idx = row_off + static_cast<size_t>(j);
                      const cuComplex v = ell_vals[idx];
                      if (v.x != 0.0f || v.y != 0.0f) {
                        csr_cols[pos] = ell_cols[idx];
                        csr_vals[pos] = v;
                        ++pos;
                      }
                    }
                  }

                  int* csr_row_offsets_d = nullptr;
                  int* csr_col_indices_d = nullptr;
                  cuComplex* csr_values_d = nullptr;
                  checkCudaErrors(cudaMalloc((void**)&csr_row_offsets_d,
                                             (static_cast<size_t>(nDim) + 1) * sizeof(int)));
                  checkCudaErrors(cudaMalloc((void**)&csr_col_indices_d, nnz * sizeof(int)));
                  checkCudaErrors(cudaMalloc((void**)&csr_values_d, nnz * sizeof(cuComplex)));
                  checkCudaErrors(cudaMemcpy(csr_row_offsets_d, row_offsets.data(),
                                             (static_cast<size_t>(nDim) + 1) * sizeof(int),
                                             cudaMemcpyHostToDevice));
                  checkCudaErrors(cudaMemcpy(csr_col_indices_d, csr_cols.data(),
                                             nnz * sizeof(int), cudaMemcpyHostToDevice));
                  checkCudaErrors(cudaMemcpy(csr_values_d, csr_vals.data(),
                                             nnz * sizeof(cuComplex), cudaMemcpyHostToDevice));

                  fused_gates_use_dense.back() = 1;
                  fused_gates_csr_row_offsets_d.back() = csr_row_offsets_d;
                  fused_gates_csr_col_indices_d.back() = csr_col_indices_d;
                  fused_gates_csr_values_d.back() = csr_values_d;
                  fused_gates_csr_nnz.back() = static_cast<int>(nnz);
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
        cuComplex *fused_gate_val_h;
        int * fused_gate_indices_h;
        for (int idx = 0; idx < fused_gates.size(); idx++) {
          qc::FusedGate fused_gate = fused_gates[idx];
          fused_num_nonzero.push_back(fused_gate.num_mac );
          fused_gates_use_dense.push_back(0);
          fused_gates_dense_d.push_back(nullptr);
          fused_gates_csr_row_offsets_d.push_back(nullptr);
          fused_gates_csr_col_indices_d.push_back(nullptr);
          fused_gates_csr_values_d.push_back(nullptr);
          fused_gates_csr_nnz.push_back(0);
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
              cuComplex* fused_gate_val;
              int* fused_gate_indices;
              checkCudaErrors(cudaMalloc((void**)&fused_gate_val, ell_width * nDim* sizeof(cuComplex)));
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
            cuComplex *fused_gate_val;
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

            checkCudaErrors(cudaMalloc((void**)&fused_gate_val, fused_gate.num_mac * nDim* sizeof(cuComplex)));
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
            checkCudaErrors(cudaMallocHost((void**)&fused_gate_val_h, fused_gate.num_mac * nDim* sizeof(cuComplex)));
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
            cuComplex *fused_gate_val;
            int *fused_gate_indices;
            checkCudaErrors(cudaMalloc((void**)&fused_gate_val, fused_gate.num_mac * nDim* sizeof(cuComplex)));
            checkCudaErrors(cudaMalloc((void**)&fused_gate_indices, fused_gate.num_mac  *nDim* sizeof(int)));
            checkCudaErrors(cudaMemcpy(fused_gate_val, fused_gate_val_h, fused_gate.num_mac * nDim* sizeof(cuComplex), cudaMemcpyHostToDevice));
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

        const bool use_cusparse = envFlag("BQSIM_RT_CUSPARSE_TENSOR");
        bool any_dense = false;
        for (size_t i = 0; i < fused_gates_use_dense.size(); ++i) {
          if (fused_gates_use_dense[i]) {
            any_dense = true;
            break;
          }
        }
        auto run_stage3_graph = [&](bool allow_dense) {
          auto checkCusparse = [](cusparseStatus_t status, const char* msg) {
            if (status != CUSPARSE_STATUS_SUCCESS) {
              throw std::runtime_error(msg);
            }
          };
          cudaStream_t stream{};
          checkCudaErrors(cudaStreamCreate(&stream));
          if (allow_dense) {
            checkCusparse(cusparseSetStream(cusparse_handle, stream), "cusparseSetStream failed");
          }

          const size_t bytes = nDim * batch_size * sizeof(cuComplex);
          const int sparse_block = 256;
          const int warps_per_block = sparse_block / 32;
          const int batch_chunks = (batch_size + 31) / 32;
          const int total_warps = static_cast<int>(nDim) * batch_chunks;
          const int sparse_grid = (total_warps + warps_per_block - 1) / warps_per_block;
          dim3 sparse_block_size = dim3(sparse_block, 1, 1);
          const cuComplex alpha = make_cuComplex(1.0f, 0.0f);
          const cuComplex beta = make_cuComplex(0.0f, 0.0f);

          cudaGraph_t graph{};
          cudaGraphExec_t graph_exec{};
          std::array<cusparseDnMatDescr_t, 4> dn_batch{};
          for (int i = 0; i < 4; ++i) {
            checkCusparse(cusparseCreateDnMat(&dn_batch[i],
                                              static_cast<int64_t>(nDim),
                                              static_cast<int64_t>(batch_size),
                                              static_cast<int64_t>(batch_size),
                                              d_batch[i],
                                              CUDA_C_32F,
                                              CUSPARSE_ORDER_ROW),
                          "cusparseCreateDnMat failed");
          }
          std::vector<cusparseSpMatDescr_t> csr_mats(fused_num_nonzero.size(), nullptr);
          size_t spmm_buffer_size = 0;
          if (allow_dense) {
            for (size_t g = 0; g < fused_num_nonzero.size(); ++g) {
              if (g < fused_gates_use_dense.size() && fused_gates_use_dense[g] &&
                  g < fused_gates_csr_row_offsets_d.size() && fused_gates_csr_row_offsets_d[g] &&
                  g < fused_gates_csr_col_indices_d.size() && fused_gates_csr_col_indices_d[g] &&
                  g < fused_gates_csr_values_d.size() && fused_gates_csr_values_d[g]) {
                const int nnz = fused_gates_csr_nnz[g];
                checkCusparse(cusparseCreateCsr(&csr_mats[g],
                                                static_cast<int64_t>(nDim),
                                                static_cast<int64_t>(nDim),
                                                static_cast<int64_t>(nnz),
                                                fused_gates_csr_row_offsets_d[g],
                                                fused_gates_csr_col_indices_d[g],
                                                fused_gates_csr_values_d[g],
                                                CUSPARSE_INDEX_32I,
                                                CUSPARSE_INDEX_32I,
                                                CUSPARSE_INDEX_BASE_ZERO,
                                                CUDA_C_32F),
                              "cusparseCreateCsr failed");
                size_t buffer_size = 0;
                checkCusparse(cusparseSpMM_bufferSize(cusparse_handle,
                                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      &alpha,
                                                      csr_mats[g],
                                                      dn_batch[0],
                                                      &beta,
                                                      dn_batch[1],
                                                      CUDA_C_32F,
                                                      CUSPARSE_SPMM_ALG_DEFAULT,
                                                      &buffer_size),
                              "cusparseSpMM_bufferSize failed");
                if (buffer_size > spmm_buffer_size) {
                  spmm_buffer_size = buffer_size;
                }
              }
            }
          }
          void* spmm_buffer = nullptr;
          if (spmm_buffer_size > 0) {
            checkCudaErrors(cudaMalloc(&spmm_buffer, spmm_buffer_size));
          }
          checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
          int last_cur = 0;
          for (int batch_id = 0; batch_id < num_batch; ++batch_id) {
            const int base = (batch_id & 1) * 2;
            int cur = base;
            int next = base + 1;
            checkCudaErrors(cudaMemcpyAsync(d_batch[base], h_batch[0], bytes, cudaMemcpyHostToDevice, stream));
            for (size_t g = 0; g < fused_num_nonzero.size(); ++g) {
              if (allow_dense &&
                  g < fused_gates_use_dense.size() && fused_gates_use_dense[g] &&
                  g < csr_mats.size() && csr_mats[g] != nullptr) {
                checkCusparse(cusparseSpMM(cusparse_handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha,
                                           csr_mats[g],
                                           dn_batch[cur],
                                           &beta,
                                           dn_batch[next],
                                           CUDA_C_32F,
                                           CUSPARSE_SPMM_ALG_DEFAULT,
                                           spmm_buffer),
                              "cusparseSpMM failed");
              } else {
                run_fused_gate_warp<<<sparse_grid, sparse_block_size, 0, stream>>>(
                    fused_gates_val_d[g], fused_gates_indices_d[g], fused_num_nonzero[g],
                    d_batch[cur], d_batch[next], batch_size, nDim);
              }
              const int tmp = cur;
              cur = next;
              next = tmp;
            }
            last_cur = cur;
            checkCudaErrors(cudaMemcpyAsync(h_batch[1], d_batch[cur], bytes, cudaMemcpyDeviceToHost, stream));
          }
          checkCudaErrors(cudaStreamEndCapture(stream, &graph));
          checkCudaErrors(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

          auto begin_sim_total = std::chrono::high_resolution_clock::now();
          checkCudaErrors(cudaGraphLaunch(graph_exec, stream));
          checkCudaErrors(cudaStreamSynchronize(stream));
          auto end_sim_total = std::chrono::high_resolution_clock::now();

          checkCudaErrors(cudaGraphExecDestroy(graph_exec));
          checkCudaErrors(cudaGraphDestroy(graph));
          if (spmm_buffer) {
            checkCudaErrors(cudaFree(spmm_buffer));
          }
          for (auto& mat : csr_mats) {
            if (mat) {
              cusparseDestroySpMat(mat);
            }
          }
          for (auto& dn : dn_batch) {
            if (dn) {
              cusparseDestroyDnMat(dn);
            }
          }
          checkCudaErrors(cudaStreamDestroy(stream));

          QBatchSimulator<Config>::final_state_idx = 1;
          QBatchSimulator<Config>::final_state_idx_gpu = last_cur;
          const auto total_ms =
              std::chrono::duration_cast<std::chrono::milliseconds>(end_sim_total - begin_sim_total).count();
          std::cout << "[Stage 3: ELL-based batch simulation] time: "
                    << total_ms
                    << std::endl;
          return;
        };

        if (use_cusparse && any_dense && cusparse_ready) {
          run_stage3_graph(true);
          return;
        }

        if (envFlag("BQSIM_RT_COMPACT_LAUNCH")) {
          run_stage3_graph(false);
          return;
        }


        run_stage3_graph(false);
        }

    [[nodiscard]]
    cuComplex* getVector() const {
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
    std::vector<cuComplex *> h_batch, d_batch;
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
    std::array<cublasHandle_t, 2> cublas_handles{};
    cudaStream_t cublas_stream{};
    cudaStream_t cublas_stream_aux{};
    cusparseHandle_t cusparse_handle{};
    bool cublas_ready = false;
    bool cusparse_ready = false;
    bool buildGatePrimitives(std::vector<qc::GatePrimitive>& out) const {
      out.clear();
      if (!qc) {
        return false;
      }
      auto set_matrix2 = [](qc::GatePrimitive& gp,
                            float a00, float b00,
                            float a01, float b01,
                            float a10, float b10,
                            float a11, float b11) {
        gp.matrix_dim = 2;
        gp.matrix[0] = make_float2(a00, b00);
        gp.matrix[1] = make_float2(a01, b01);
        gp.matrix[2] = make_float2(a10, b10);
        gp.matrix[3] = make_float2(a11, b11);
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
            const float inv = 1.0f / sqrtf(2.0f);
            set_matrix2(gp, inv, 0.0f, inv, 0.0f, inv, 0.0f, -inv, 0.0f);
            break;
          }
          case qc::Z:
            set_matrix2(gp, 1, 0, 0, 0, 0, 0, -1, 0);
            break;
          case qc::S:
            set_matrix2(gp, 1, 0, 0, 0, 0, 0, 0, 1);
            break;
          case qc::T: {
            const float angle = static_cast<float>(qc::PI_4);
            set_matrix2(gp, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, cosf(angle), sinf(angle));
            break;
          }
          case qc::RX: {
            const float theta = params.empty() ? 0.0f : static_cast<float>(params[0]);
            const float c = cosf(theta * 0.5f);
            const float s = sinf(theta * 0.5f);
            set_matrix2(gp, c, 0.0f, 0.0f, -s, 0.0f, -s, c, 0.0f);
            break;
          }
          case qc::RY: {
            const float theta = params.empty() ? 0.0f : static_cast<float>(params[0]);
            const float c = cosf(theta * 0.5f);
            const float s = sinf(theta * 0.5f);
            set_matrix2(gp, c, 0.0f, -s, 0.0f, s, 0.0f, c, 0.0f);
            break;
          }
          case qc::RZ: {
            const float theta = params.empty() ? 0.0f : static_cast<float>(params[0]);
            const float c = cosf(theta * 0.5f);
            const float s = sinf(theta * 0.5f);
            set_matrix2(gp, c, -s, 0.0f, 0.0f, 0.0f, 0.0f, c, s);
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
    dd::fp        epsilon = 1e-5f;
    std::unique_ptr<qc::QuantumComputation> qc;
    std::size_t                             singleShots{0};
    int batch_size = 1;
    int num_batch = 1;
    int gpu_full_at = -1;
    std::vector<cuComplex*> fused_gates_val_d;
    std::vector<int*> fused_gates_indices_d;
    std::vector<cuComplex*> fused_gates_dense_d;
    std::vector<int*> fused_gates_csr_row_offsets_d;
    std::vector<int*> fused_gates_csr_col_indices_d;
    std::vector<cuComplex*> fused_gates_csr_values_d;
    std::vector<int> fused_gates_csr_nnz;
    std::vector<uint8_t> fused_gates_use_dense;


    // std::vector<cuComplex*> fused_gates_val_moreh;
    // std::vector<int*> fused_gates_indices_moreh;
    // std::vector<cuComplex*> fused_gates_val_mored;
    // std::vector<int*> fused_gates_indices_mored;

    // 
};

template class QBatchSimulator<dd::DDPackageConfig>;

#endif //QBATCH_SIMULATOR_H
