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
#include <cuComplex.h>
#include <cooperative_groups.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
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

__global__ void initial_check(cuDoubleComplex *input_arr_d, bool *identical, int N, double tol) {
  extern __shared__ bool s[];
  __shared__ int res[1];
  if (threadIdx.x == 0) {
    res[0] = true;
  }
  __syncthreads();
  const cuDoubleComplex a = input_arr_d[threadIdx.x + blockIdx.x * N];
  const cuDoubleComplex b = input_arr_d[blockIdx.x * N];
  const bool finite = isfinite(a.x) && isfinite(a.y) && isfinite(b.x) && isfinite(b.y);
  s[threadIdx.x] = finite &&
                   (fabs(a.x - b.x) <= tol) &&
                   (fabs(a.y - b.y) <= tol);
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
    cuDoubleComplex rec_factor = make_cuDoubleComplex(1.0, 0.0);
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
        const cuDoubleComplex edge_w = dd_edges[edge_ptr].w;
        decoded_factors[decode_ptr[0]] = cuCmul(rec_factor, edge_w);
        stack_ptr--; decode_ptr[0]++;
        continue;
      }

      int child_idx = (int)(left_or_right[stack_ptr]) + (int)(up_or_down[stack_ptr]) * 2;
      // return or move forward
      if (left_or_right[stack_ptr] == 2) {
        left_or_right[stack_ptr] = 0;
        const cuDoubleComplex edge_w = dd_edges[edge_ptr].w;
        rec_factor = cuCdiv(rec_factor, edge_w);
        rec_loc -= (1 << dd_nodes[node_ptr].qubit);
        stack_ptr--;
      }
      else {
        left_or_right[stack_ptr]++;
        if (left_or_right[stack_ptr] == 1) {
          const cuDoubleComplex edge_w = dd_edges[edge_ptr].w;
          rec_factor = cuCmul(rec_factor, edge_w);
        }
        rec_loc += (1 << dd_nodes[node_ptr].qubit) * (int)(left_or_right[stack_ptr] -1);
        stack_ptr++;
        edge_stack[stack_ptr] = dd_nodes[node_ptr].outgoing_DD_edge_ptr[child_idx];
      }
    }
  }

  __syncthreads();
  if (tid < num_non_zeros) {
    fused_gate_val[bid * num_non_zeros + tid] = make_cuDoubleComplex(0.0, 0.0);
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
    cuDoubleComplex rec_factor = make_cuDoubleComplex(1.0, 0.0);
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
        const cuDoubleComplex edge_w = dd_edges[edge_ptr].w;
        decoded_factors[MAX_DECODED_MACS*(tid/WARP_SIZE)+decode_ptr[tid/WARP_SIZE]] = cuCmul(rec_factor, edge_w);
        stack_ptr--; decode_ptr[tid/WARP_SIZE]++;
        continue;
      }

      int child_idx = (int)(left_or_right[MAX_LEV*(tid/WARP_SIZE) + stack_ptr]) + (int)(up_or_down[MAX_LEV*(tid/WARP_SIZE) + stack_ptr]) * 2;
      // return or move forward
      if (left_or_right[MAX_LEV*(tid/WARP_SIZE) + stack_ptr] == 2) {
        left_or_right[MAX_LEV*(tid/WARP_SIZE) + stack_ptr] = 0;
        const cuDoubleComplex edge_w = dd_edges[edge_ptr].w;
        rec_factor = cuCdiv(rec_factor, edge_w);
        rec_loc -= (1 << dd_nodes[node_ptr].qubit);
        stack_ptr--;
      }
      else {
        left_or_right[MAX_LEV*(tid/WARP_SIZE) + stack_ptr]++;
        if (left_or_right[MAX_LEV*(tid/WARP_SIZE) + stack_ptr] == 1) {
          const cuDoubleComplex edge_w = dd_edges[edge_ptr].w;
          rec_factor = cuCmul(rec_factor, edge_w);
        }
        rec_loc += (1 << dd_nodes[node_ptr].qubit) * (int)(left_or_right[MAX_LEV*(tid/WARP_SIZE) + stack_ptr] -1);
        stack_ptr++;
        edge_stack[MAX_LEV*(tid/WARP_SIZE)+stack_ptr] = dd_nodes[node_ptr].outgoing_DD_edge_ptr[child_idx];
      }
    }
  }

  __syncwarp();
  if (tid%WARP_SIZE < num_non_zeros) {
    fused_gate_val[(bid*WARPS_PER_BLOCK+tid/WARP_SIZE) * num_non_zeros + tid%WARP_SIZE] = make_cuDoubleComplex(0.0, 0.0);
    fused_gate_indices[(bid*WARPS_PER_BLOCK+tid/WARP_SIZE) * num_non_zeros + tid%WARP_SIZE] = 0;
  }
  __syncwarp();

  if (tid%WARP_SIZE < decode_ptr[tid/WARP_SIZE]) {
    fused_gate_val[(bid*WARPS_PER_BLOCK+tid/WARP_SIZE) * num_non_zeros + tid%WARP_SIZE] = decoded_factors[MAX_DECODED_MACS*(tid/WARP_SIZE)+tid%WARP_SIZE];
    fused_gate_indices[(bid*WARPS_PER_BLOCK+tid/WARP_SIZE) * num_non_zeros + tid%WARP_SIZE] = decoded_locs[MAX_DECODED_MACS*(tid/WARP_SIZE)+tid%WARP_SIZE];
  }
  __syncwarp();

}



__global__ void run_fused_gate(
  const cuDoubleComplex* __restrict__ gates_val,
  const int* __restrict__ gates_indices,
  int num_non_zero,
  const cuDoubleComplex* __restrict__ input_state,
  cuDoubleComplex* __restrict__ output_state,
  int batch_size,
  int nDim
) {
  extern __shared__ unsigned char smem[];
  int* share_indices = reinterpret_cast<int*>(smem);
  const size_t idx_bytes = static_cast<size_t>(num_non_zero) * sizeof(int);
  const size_t val_offset =
      (idx_bytes + alignof(cuDoubleComplex) - 1) & ~(alignof(cuDoubleComplex) - 1);
  cuDoubleComplex* shared_val =
      reinterpret_cast<cuDoubleComplex*>(smem + val_offset);

  const int tid = threadIdx.x;
  const int rows = nDim / gridDim.x;
  for (int i = 0; i < rows; ++i) {
    const int row = i * gridDim.x + blockIdx.x;
    for (int idx = tid; idx < num_non_zero; idx += blockDim.x) {
      const size_t offset = static_cast<size_t>(row) * static_cast<size_t>(num_non_zero) + idx;
      share_indices[idx] = gates_indices[offset];
      shared_val[idx] = gates_val[offset];
    }
    __syncthreads();

    cuDoubleComplex result_value = make_cuDoubleComplex(0.0, 0.0);
    for (int j = 0; j < num_non_zero; ++j) {
      const int col = share_indices[j];
      const cuDoubleComplex v = shared_val[j];
      if (v.x != 0.0 || v.y != 0.0) {
        const size_t in_idx = static_cast<size_t>(col) * batch_size + tid;
        const cuDoubleComplex in_val = input_state[in_idx];
        result_value = cuCadd(result_value, cuCmul(v, in_val));
      }
    }
    output_state[static_cast<size_t>(row) * batch_size + tid] = result_value;
    __syncthreads();
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
        const char* pipeline_mode_init = std::getenv("BQSIM_RT_PIPELINE_MODE");
        const bool warmup_spm = pipeline_mode_init && std::strcmp(pipeline_mode_init, "SPMSPM") == 0 &&
                                rtEngine && rtEngine->isAvailable();
        if (warmup_spm) {
          rtEngine->warmup();
        }
        
        cuDoubleComplex *h_batch0;
        cuDoubleComplex *h_batch1;
        const size_t host_bytes = nDim * batch_size_ * sizeof(cuDoubleComplex);
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
            h_batch0[amp_id*batch_size_] = make_cuDoubleComplex(real, imag);
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
                                      rtEngine && rtEngine->isAvailable();
        if (use_spm_pipeline) {
          std::vector<qc::GatePrimitive> primitives;
          if (!buildGatePrimitives(primitives)) {
            std::cerr << "[SPMSPM] GatePrimitive build failed; aborting SPMSPM pipeline." << std::endl;
            return;
          }
          auto begin_convert = std::chrono::high_resolution_clock::now();
          const bool hybrid_enabled = envFlag("BQSIM_RT_HYBRID_DENSE");
          const size_t total_gates = primitives.size();
          (void)envDouble("BQSIM_RT_DENSE_THRESHOLD", 0.01);
          (void)envUInt64("BQSIM_RT_DENSE_MAX_BYTES", 512ULL * 1024ULL * 1024ULL);
          if (batch_size % 32 != 0) {
            std::cerr << "[SPMSPM] Dense path: batch_size not multiple of 32; expect lower memory coalescing." << std::endl;
          }

          double total_cpu_plan_ms = 0.0;
          double total_h2d_ms = 0.0;
          double total_ray_gen_ms = 0.0;
          double total_bvh_ms = 0.0;
          double total_launch_ms = 0.0;
          double total_merge_ms = 0.0;
          double total_overhead_ms = 0.0;
          double total_ell_convert_ms = 0.0;

          auto cleanup_spm = [&]() {
            for (size_t i = 0; i < fused_gates_val_d.size(); ++i) {
              if (fused_gates_val_d[i]) {
                cudaFree(fused_gates_val_d[i]);
              }
              if (i < fused_gates_indices_d.size() && fused_gates_indices_d[i]) {
                cudaFree(fused_gates_indices_d[i]);
              }
            }
            fused_gates_val_d.clear();
            fused_gates_indices_d.clear();
          };

          const size_t max_gates_per_block = 20;
          std::vector<size_t> block_sizes;
          block_sizes.reserve(total_gates);
          auto nnz_multiplier = [](const qc::GatePrimitive& gp) -> size_t {
            switch (gp.gate_type) {
              case qc::H:
              case qc::RX:
              case qc::RY:
              case qc::U2:
              case qc::U3:
                return 2;
              case qc::X:
              case qc::Y:
              case qc::Z:
              case qc::S:
              case qc::T:
              case qc::Phase:
              case qc::SWAP:
                return 1;
              default:
                // Default to sparsity-preserving to avoid premature fusion stops.
                return 1;
            }
          };
          size_t plan_cursor = 0;
          auto plan_start = std::chrono::high_resolution_clock::now();
          while (plan_cursor < total_gates) {
            size_t current_nnz = 1;
            size_t block_count = 0;
            for (size_t gi = plan_cursor; gi < total_gates; ++gi) {
              if (block_count >= max_gates_per_block) {
                break;
              }
              const auto& gp = primitives[gi];
              const size_t factor = nnz_multiplier(gp);
              const size_t next_nnz = current_nnz * factor;
              if (next_nnz > 4) {
                break;
              }
              current_nnz = next_nnz;
              ++block_count;
            }
            if (block_count == 0) {
              block_count = 1;
            }
            block_sizes.push_back(block_count);
            plan_cursor += block_count;
          }
          auto plan_stop = std::chrono::high_resolution_clock::now();
          total_cpu_plan_ms += std::chrono::duration<double, std::milli>(plan_stop - plan_start).count();

          size_t cursor = 0;
          size_t block_id = 0;
          while (cursor < total_gates) {
            const size_t remaining = total_gates - cursor;
            const size_t planned = std::min(block_sizes[block_id], remaining);
            if (planned == 0) {
              break;
            }
            std::cout << "[SPMSPM] Fusing block " << (block_id + 1)
                      << " (plan " << (block_id + 1) << "/" << block_sizes.size() << ")"
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
            const auto& stats = rtEngine->lastStats();
            total_h2d_ms += stats.h2d_ms;
            total_ray_gen_ms += stats.ray_gen_ms;
            total_bvh_ms += stats.gas_ms;
            total_launch_ms += stats.launch_ms;
            total_merge_ms += stats.merge_ms;
            total_overhead_ms += stats.overhead_ms;

            int ell_width = rtEngine->maxRowNNZ();
            if (ell_width <= 0) {
              ell_width = 1;
            }
            auto ell_start = std::chrono::high_resolution_clock::now();
            cuDoubleComplex* fused_gate_val = nullptr;
            int* fused_gate_indices = nullptr;
            if (cudaMalloc((void**)&fused_gate_val, ell_width * nDim * sizeof(cuDoubleComplex)) != cudaSuccess ||
                cudaMalloc((void**)&fused_gate_indices, ell_width * nDim * sizeof(int)) != cudaSuccess) {
              if (fused_gate_indices) {
                cudaFree(fused_gate_indices);
              }
              std::cerr << "[SPMSPM] cudaMalloc failed during ELL allocation; aborting SPMSPM pipeline."
                        << std::endl;
              cleanup_spm();
              return;
            }
            checkCudaErrors(cudaMemset(fused_gate_val, 0, ell_width * nDim * sizeof(cuDoubleComplex)));
            checkCudaErrors(cudaMemset(fused_gate_indices, 0, ell_width * nDim * sizeof(int)));
            if (rtEngine->collectResultToELL(fused_gate_val, fused_gate_indices, ell_width, nDim)) {
              auto ell_stop = std::chrono::high_resolution_clock::now();
              total_ell_convert_ms += std::chrono::duration<double, std::milli>(ell_stop - ell_start).count();
              fused_gates_val_d.push_back(fused_gate_val);
              fused_gates_indices_d.push_back(fused_gate_indices);
              fused_num_nonzero.push_back(ell_width);
            } else {
              auto ell_stop = std::chrono::high_resolution_clock::now();
              total_ell_convert_ms += std::chrono::duration<double, std::milli>(ell_stop - ell_start).count();
              if (fused_gate_val) {
                checkCudaErrors(cudaFree(fused_gate_val));
              }
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
            std::cout << "[SPMSPM]   fused " << actual << " gate(s), ELL width: " << ell_width << std::endl;
            cursor += std::min(actual, remaining);
            ++block_id;
          }
          used_spm_pipeline = true;
          auto end_convert = std::chrono::high_resolution_clock::now();
          std::cout << "[Stage 1: RT Core Gate Fusion] time: "
                    << std::chrono::duration_cast<std::chrono::milliseconds>(end_convert - begin_convert).count()
                    << std::endl;
          std::cout << "  Breakdown:" << std::endl;
          std::cout << "  - CPU Planning (Block Size): " << total_cpu_plan_ms << " ms" << std::endl;
          std::cout << "  - H2D Transfer (Params):     " << total_h2d_ms << " ms" << std::endl;
          std::cout << "  - Ray Generation:            " << total_ray_gen_ms << " ms" << std::endl;
          std::cout << "  - BVH Build (OptiX):         " << total_bvh_ms << " ms" << std::endl;
          std::cout << "  - Ray Tracing (Launch):      " << total_launch_ms << " ms" << std::endl;
          std::cout << "  - Sort & Merge (Thrust):     " << total_merge_ms << " ms" << std::endl;
          std::cout << "  - Memory & Overhead:         " << total_overhead_ms << " ms" << std::endl;
          std::cout << "  - ELL Conversion (Result):   " << total_ell_convert_ms << " ms" << std::endl;
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
          total_macs += fused_gate.num_mac;

          std::cout << "Converting fused gate #" << idx << " using ";
          auto begin_gate_convert = std::chrono::high_resolution_clock::now();
          bool rt_done = false;
          const bool use_rt = rtEngine && rtEngine->isAvailable();
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
              fused_gate_indices_h, 0, 0, sparse_idx_x, fused_gate.num_mac, make_cuDoubleComplex(1.0, 0.0));
            
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
            const size_t total = static_cast<size_t>(fused_gate.num_mac) * static_cast<size_t>(nDim);
            cuDoubleComplex *fused_gate_val;
            int *fused_gate_indices;
            checkCudaErrors(cudaMalloc((void**)&fused_gate_val, total * sizeof(cuDoubleComplex)));
            checkCudaErrors(cudaMalloc((void**)&fused_gate_indices, fused_gate.num_mac  *nDim* sizeof(int)));
            checkCudaErrors(cudaMemcpy(fused_gate_val, fused_gate_val_h, total * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
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

        auto run_stage3_graph = [&]() {
          const size_t bytes = nDim * batch_size * sizeof(cuDoubleComplex);
          const int grid_size = (nDim > 8192) ? 8192 : static_cast<int>(nDim);
          dim3 block_size = dim3(static_cast<unsigned int>(batch_size), 1, 1);

          cudaStream_t stream{};
          checkCudaErrors(cudaStreamCreate(&stream));
          cudaGraph_t graph{};
          cudaGraphExec_t graph_exec{};

          checkCudaErrors(cudaGraphCreate(&graph, 0));
          std::vector<cudaGraphNode_t> h2d_nodes(num_batch);
          std::vector<cudaGraphNode_t> d2h_nodes(num_batch);

          struct KernelArgs {
            const cuDoubleComplex* gates_val;
            const int* gates_indices;
            int num_non_zero;
            const cuDoubleComplex* input_state;
            cuDoubleComplex* output_state;
            int batch_size;
            int nDim;
          };

          const size_t kernel_count = static_cast<size_t>(num_batch) * fused_num_nonzero.size();
          std::vector<KernelArgs> kernel_args;
          std::vector<std::array<void*, 7>> kernel_params;
          kernel_args.reserve(kernel_count);
          kernel_params.reserve(kernel_count);

          int last_cur = 0;
          for (int batch_id = 0; batch_id < num_batch; ++batch_id) {
            const int base = (batch_id & 1) * 2;
            int cur = base;
            int next = base + 1;

            cudaMemcpy3DParms h2d_params{};
            h2d_params.srcPtr = make_cudaPitchedPtr(reinterpret_cast<void*>(h_batch[0]), bytes, bytes, 1);
            h2d_params.dstPtr = make_cudaPitchedPtr(reinterpret_cast<void*>(d_batch[base]), bytes, bytes, 1);
            h2d_params.extent = make_cudaExtent(bytes, 1, 1);
            h2d_params.kind = cudaMemcpyHostToDevice;
            checkCudaErrors(cudaGraphAddMemcpyNode(&h2d_nodes[batch_id], graph, nullptr, 0, &h2d_params));
            if (batch_id >= 2) {
              checkCudaErrors(cudaGraphAddDependencies(graph,
                                                       &d2h_nodes[batch_id - 2],
                                                       &h2d_nodes[batch_id],
                                                       1));
            }

            cudaGraphNode_t prev_node = h2d_nodes[batch_id];
            for (size_t g = 0; g < fused_num_nonzero.size(); ++g) {
              const int nnz = fused_num_nonzero[g];
              const size_t idx_bytes = static_cast<size_t>(nnz) * sizeof(int);
              const size_t val_offset =
                  (idx_bytes + alignof(cuDoubleComplex) - 1) & ~(alignof(cuDoubleComplex) - 1);
              const size_t shared_bytes = val_offset + static_cast<size_t>(nnz) * sizeof(cuDoubleComplex);

              kernel_args.push_back({fused_gates_val_d[g],
                                     fused_gates_indices_d[g],
                                     nnz,
                                     d_batch[cur],
                                     d_batch[next],
                                     batch_size,
                                     static_cast<int>(nDim)});
              KernelArgs& args = kernel_args.back();
              kernel_params.push_back({reinterpret_cast<void*>(&args.gates_val),
                                       reinterpret_cast<void*>(&args.gates_indices),
                                       reinterpret_cast<void*>(&args.num_non_zero),
                                       reinterpret_cast<void*>(&args.input_state),
                                       reinterpret_cast<void*>(&args.output_state),
                                       reinterpret_cast<void*>(&args.batch_size),
                                       reinterpret_cast<void*>(&args.nDim)});

              cudaKernelNodeParams kparams{};
              kparams.func = reinterpret_cast<void*>(run_fused_gate);
              kparams.gridDim = dim3(static_cast<unsigned int>(grid_size), 1, 1);
              kparams.blockDim = block_size;
              kparams.sharedMemBytes = static_cast<unsigned int>(shared_bytes);
              kparams.kernelParams = kernel_params.back().data();
              kparams.extra = nullptr;

              cudaGraphNode_t k_node{};
              checkCudaErrors(cudaGraphAddKernelNode(&k_node, graph, &prev_node, 1, &kparams));
              prev_node = k_node;

              const int tmp = cur;
              cur = next;
              next = tmp;
            }

            last_cur = cur;
            cudaMemcpy3DParms d2h_params{};
            d2h_params.srcPtr = make_cudaPitchedPtr(reinterpret_cast<void*>(d_batch[cur]), bytes, bytes, 1);
            d2h_params.dstPtr = make_cudaPitchedPtr(reinterpret_cast<void*>(h_batch[1]), bytes, bytes, 1);
            d2h_params.extent = make_cudaExtent(bytes, 1, 1);
            d2h_params.kind = cudaMemcpyDeviceToHost;
            checkCudaErrors(cudaGraphAddMemcpyNode(&d2h_nodes[batch_id], graph, &prev_node, 1, &d2h_params));
          }
          checkCudaErrors(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

          auto begin_sim_total = std::chrono::high_resolution_clock::now();
          checkCudaErrors(cudaGraphLaunch(graph_exec, stream));
          checkCudaErrors(cudaStreamSynchronize(stream));
          auto end_sim_total = std::chrono::high_resolution_clock::now();

          checkCudaErrors(cudaGraphExecDestroy(graph_exec));
          checkCudaErrors(cudaGraphDestroy(graph));
          checkCudaErrors(cudaStreamDestroy(stream));

          QBatchSimulator<Config>::final_state_idx = 1;
          QBatchSimulator<Config>::final_state_idx_gpu = last_cur;
          const auto total_ms =
              std::chrono::duration_cast<std::chrono::milliseconds>(end_sim_total - begin_sim_total).count();
          if (used_spm_pipeline) {
            std::cout << "[Stage 2: ELL-based batch simulation] time: "
                      << total_ms
                      << std::endl;
          } else {
            std::cout << "[Stage 3: ELL-based batch simulation] time: "
                      << total_ms
                      << std::endl;
          }
        };

        run_stage3_graph();
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
        gp.matrix[0] = make_double2(a00, b00);
        gp.matrix[1] = make_double2(a01, b01);
        gp.matrix[2] = make_double2(a10, b10);
        gp.matrix[3] = make_double2(a11, b11);
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

        const auto& params = op->getParameter();
        if (gp.control_count > 0) {
          if (gp.target_count != 1) {
            return false;
          }
          switch (type) {
            case qc::X:
              set_matrix2(gp, 0, 0, 1, 0, 1, 0, 0, 0);
              break;
            case qc::Z:
              set_matrix2(gp, 1, 0, 0, 0, 0, 0, -1, 0);
              break;
            case qc::RX: {
              const double theta = params.empty() ? 0.0 : static_cast<double>(params[0]);
              const double c = std::cos(theta * 0.5);
              const double s = std::sin(theta * 0.5);
              set_matrix2(gp, c, 0.0f, 0.0f, -s, 0.0f, -s, c, 0.0f);
              break;
            }
            case qc::RY: {
              const double theta = params.empty() ? 0.0 : static_cast<double>(params[0]);
              const double c = std::cos(theta * 0.5);
              const double s = std::sin(theta * 0.5);
              set_matrix2(gp, c, 0.0f, -s, 0.0f, s, 0.0f, c, 0.0f);
              break;
            }
            case qc::RZ: {
              const double theta = params.empty() ? 0.0 : static_cast<double>(params[0]);
              const double c = std::cos(theta * 0.5);
              const double s = std::sin(theta * 0.5);
              set_matrix2(gp, c, -s, 0.0f, 0.0f, 0.0f, 0.0f, c, s);
              break;
            }
            case qc::Phase: {
              const double theta = params.empty() ? 0.0 : static_cast<double>(params[0]);
              set_matrix2(gp, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, std::cos(theta), std::sin(theta));
              break;
            }
            case qc::S:
              set_matrix2(gp, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
              break;
            case qc::Sdag:
              set_matrix2(gp, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f);
              break;
            case qc::T: {
              const double angle = static_cast<double>(qc::PI_4);
              set_matrix2(gp, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, std::cos(angle), std::sin(angle));
              break;
            }
            case qc::Tdag: {
              const double angle = -static_cast<double>(qc::PI_4);
              set_matrix2(gp, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, std::cos(angle), std::sin(angle));
              break;
            }
            case qc::U2: {
              const double phi = params.size() > 0 ? static_cast<double>(params[0]) : 0.0;
              const double lambda = params.size() > 1 ? static_cast<double>(params[1]) : 0.0;
              const double inv = 1.0 / std::sqrt(2.0);
              const double c0 = std::cos(lambda);
              const double s0 = std::sin(lambda);
              const double c1 = std::cos(phi);
              const double s1 = std::sin(phi);
              const double c2 = std::cos(phi + lambda);
              const double s2 = std::sin(phi + lambda);
              set_matrix2(gp,
                          inv, 0.0f,
                          -inv * c0, -inv * s0,
                          inv * c1, inv * s1,
                          inv * c2, inv * s2);
              break;
            }
            case qc::U3: {
              const double theta = params.size() > 0 ? static_cast<double>(params[0]) : 0.0;
              const double phi = params.size() > 1 ? static_cast<double>(params[1]) : 0.0;
              const double lambda = params.size() > 2 ? static_cast<double>(params[2]) : 0.0;
              const double c = std::cos(theta * 0.5);
              const double s = std::sin(theta * 0.5);
              const double c0 = std::cos(lambda);
              const double s0 = std::sin(lambda);
              const double c1 = std::cos(phi);
              const double s1 = std::sin(phi);
              const double c2 = std::cos(phi + lambda);
              const double s2 = std::sin(phi + lambda);
              set_matrix2(gp,
                          c, 0.0f,
                          -s * c0, -s * s0,
                          s * c1, s * s1,
                          c * c2, c * s2);
              break;
            }
            default:
              return false;
          }
          out.push_back(gp);
          continue;
        }

        if (gp.target_count != 1) {
          return false;
        }

        switch (type) {
          case qc::X:
            set_matrix2(gp, 0, 0, 1, 0, 1, 0, 0, 0);
            break;
          case qc::H: {
            const double inv = 1.0 / std::sqrt(2.0);
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
            const double angle = static_cast<double>(qc::PI_4);
            set_matrix2(gp, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, std::cos(angle), std::sin(angle));
            break;
          }
          case qc::RX: {
            const double theta = params.empty() ? 0.0 : static_cast<double>(params[0]);
            const double c = std::cos(theta * 0.5);
            const double s = std::sin(theta * 0.5);
            set_matrix2(gp, c, 0.0f, 0.0f, -s, 0.0f, -s, c, 0.0f);
            break;
          }
          case qc::RY: {
            const double theta = params.empty() ? 0.0 : static_cast<double>(params[0]);
            const double c = std::cos(theta * 0.5);
            const double s = std::sin(theta * 0.5);
            set_matrix2(gp, c, 0.0f, -s, 0.0f, s, 0.0f, c, 0.0f);
            break;
          }
          case qc::RZ: {
            const double theta = params.empty() ? 0.0 : static_cast<double>(params[0]);
            const double c = std::cos(theta * 0.5);
            const double s = std::sin(theta * 0.5);
            set_matrix2(gp, c, -s, 0.0f, 0.0f, 0.0f, 0.0f, c, s);
            break;
          }
          case qc::Phase: {
            const double theta = params.empty() ? 0.0 : static_cast<double>(params[0]);
            set_matrix2(gp, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, std::cos(theta), std::sin(theta));
            break;
          }
          case qc::Sdag:
            set_matrix2(gp, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f);
            break;
          case qc::Tdag: {
            const double angle = -static_cast<double>(qc::PI_4);
            set_matrix2(gp, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, std::cos(angle), std::sin(angle));
            break;
          }
          case qc::U2: {
            const double phi = params.size() > 0 ? static_cast<double>(params[0]) : 0.0;
            const double lambda = params.size() > 1 ? static_cast<double>(params[1]) : 0.0;
            const double inv = 1.0 / std::sqrt(2.0);
            const double c0 = std::cos(lambda);
            const double s0 = std::sin(lambda);
            const double c1 = std::cos(phi);
            const double s1 = std::sin(phi);
            const double c2 = std::cos(phi + lambda);
            const double s2 = std::sin(phi + lambda);
            set_matrix2(gp,
                        inv, 0.0f,
                        -inv * c0, -inv * s0,
                        inv * c1, inv * s1,
                        inv * c2, inv * s2);
            break;
          }
          case qc::U3: {
            const double theta = params.size() > 0 ? static_cast<double>(params[0]) : 0.0;
            const double phi = params.size() > 1 ? static_cast<double>(params[1]) : 0.0;
            const double lambda = params.size() > 2 ? static_cast<double>(params[2]) : 0.0;
            const double c = std::cos(theta * 0.5);
            const double s = std::sin(theta * 0.5);
            const double c0 = std::cos(lambda);
            const double s0 = std::sin(lambda);
            const double c1 = std::cos(phi);
            const double s1 = std::sin(phi);
            const double c2 = std::cos(phi + lambda);
            const double s2 = std::sin(phi + lambda);
            set_matrix2(gp,
                        c, 0.0f,
                        -s * c0, -s * s0,
                        s * c1, s * s1,
                        c * c2, c * s2);
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
    dd::fp        epsilon = 1e-5;
    std::unique_ptr<qc::QuantumComputation> qc;
    std::size_t                             singleShots{0};
    int batch_size = 1;
    int num_batch = 1;
    int gpu_full_at = -1;
    std::vector<cuDoubleComplex*> fused_gates_val_d;
    std::vector<int*> fused_gates_indices_d;


    // std::vector<cuComplex*> fused_gates_val_moreh;
    // std::vector<int*> fused_gates_indices_moreh;
    // std::vector<cuComplex*> fused_gates_val_mored;
    // std::vector<int*> fused_gates_indices_mored;

    // 
};

template class QBatchSimulator<dd::DDPackageConfig>;

#endif //QBATCH_SIMULATOR_H
