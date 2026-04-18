#ifndef QBATCH_SIMULATOR_H
#define QBATCH_SIMULATOR_H



#include "QuantumComputation.hpp"
#include "Definitions.hpp"
#include "dd/Package.hpp"
#include "operations/OpType.hpp"
#include "CircuitOptimizer.hpp"
#include "RTSpMSpMEngine.hpp"
#include "GatePrimitive.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <cuComplex.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <taskflow/taskflow.hpp>
#include <taskflow/cuda/cudaflow.hpp>

inline void waitForCudaInitializationSuccess() {
  constexpr int kRetryMs = 25;
  int attempt = 0;
  while (true) {
    ++attempt;

    cudaError_t err = cudaFree(0);
    if (err == cudaSuccess) {
      int device_count = 0;
      err = cudaGetDeviceCount(&device_count);
      if (err == cudaSuccess && device_count > 0) {
        err = cudaSetDevice(0);
        if (err == cudaSuccess) {
          return;
        }
      }
    }

    std::cerr << "[CUDA init] attempt " << attempt << " failed: "
              << cudaGetErrorString(err)
              << ", retrying in " << kRetryMs << " ms" << std::endl;
    cudaGetLastError(); // clear sticky runtime error state before retry
    std::this_thread::sleep_for(std::chrono::milliseconds(kRetryMs));
  }
}

__global__ void replicate(bqsim_rt::Complex *input_arr_d, int N) {
  input_arr_d[threadIdx.x+blockIdx.x*N] = input_arr_d[blockIdx.x*N];
}

__global__ void initial_check(bqsim_rt::Complex *input_arr_d, bool *identical, int N, bqsim_rt::Real tol) {
  extern __shared__ bool s[];
  __shared__ int res[1];
  if (threadIdx.x == 0) {
    res[0] = true;
  }
  __syncthreads();
  const bqsim_rt::Complex a = input_arr_d[threadIdx.x + blockIdx.x * N];
  const bqsim_rt::Complex b = input_arr_d[blockIdx.x * N];
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

__global__ void run_fused_gate(
  bqsim_rt::Complex *gates_val,
  int *gates_indices,
  int num_non_zero,
  bqsim_rt::Complex *input_state,
  bqsim_rt::Complex *output_state,
  int batch_size,
  int nDim
) {
  int rows = nDim / gridDim.x;
  const int tid = threadIdx.x;
  int bid = blockIdx.x;
  __shared__ int share_indices[MAX_DECODED_MACS];
  __shared__ bqsim_rt::Complex shared_val[MAX_DECODED_MACS];

  for (int i = 0; i < rows; i++) {
    for (int idx = tid; idx < num_non_zero; idx += blockDim.x) {
      share_indices[idx] = gates_indices[rows * bid * num_non_zero + i * num_non_zero + idx];
      shared_val[idx] = gates_val[rows * bid * num_non_zero + i * num_non_zero + idx];
    }
    __syncthreads();

    bqsim_rt::Complex result_value = bqsim_rt::make_complex(0.0f, 0.0f);
    for (int j = 0; j < num_non_zero; j++) {
      const bqsim_rt::Complex in32 = input_state[share_indices[j] * batch_size + tid];
      const bqsim_rt::Complex temp_value = bqsim_rt::cmul(in32, shared_val[j]);
      result_value = bqsim_rt::cadd(result_value, temp_value);
    }
    __syncthreads();
    output_state[(rows * bid + i) * batch_size + tid] = result_value;
  }
  __syncthreads();
}

__global__ void run_fused_gate_reordered(
  bqsim_rt::Complex *gates_val,
  int *gates_indices,
  int num_non_zero,
  const int *row_order,
  bqsim_rt::Complex *input_state,
  bqsim_rt::Complex *output_state,
  int batch_size,
  int nDim
) {
  int rows = nDim / gridDim.x;
  const int tid = threadIdx.x;
  int bid = blockIdx.x;
  __shared__ int shared_indices[MAX_DECODED_MACS];
  __shared__ bqsim_rt::Complex shared_val[MAX_DECODED_MACS];
  __shared__ int row_idx;

  for (int i = 0; i < rows; i++) {
    if (tid == 0) {
      row_idx = row_order[rows * bid + i];
    }
    __syncthreads();

    for (int idx = tid; idx < num_non_zero; idx += blockDim.x) {
      const int offset = row_idx * num_non_zero + idx;
      shared_indices[idx] = gates_indices[offset];
      shared_val[idx] = gates_val[offset];
    }
    __syncthreads();

    bqsim_rt::Complex result_value = bqsim_rt::make_complex(0.0f, 0.0f);
    for (int j = 0; j < num_non_zero; j++) {
      const bqsim_rt::Complex in32 = input_state[shared_indices[j] * batch_size + tid];
      const bqsim_rt::Complex temp_value = bqsim_rt::cmul(in32, shared_val[j]);
      result_value = bqsim_rt::cadd(result_value, temp_value);
    }
    __syncthreads();
    output_state[row_idx * batch_size + tid] = result_value;
  }
  __syncthreads();
}

struct RowSortKey {
  int v0;
  int v1;
  int v2;
  int v3;

  __host__ __device__ bool operator<(const RowSortKey& other) const {
    if (v0 != other.v0) return v0 < other.v0;
    if (v1 != other.v1) return v1 < other.v1;
    if (v2 != other.v2) return v2 < other.v2;
    return v3 < other.v3;
  }
};

__global__ void build_row_order_keys_w4(const int* gates_indices,
                                        RowSortKey* row_keys,
                                        int* row_order,
                                        int nDim) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= nDim) {
    return;
  }

  int a = gates_indices[row * 4 + 0];
  int b = gates_indices[row * 4 + 1];
  int c = gates_indices[row * 4 + 2];
  int d = gates_indices[row * 4 + 3];

  if (b < a) { const int t = a; a = b; b = t; }
  if (d < c) { const int t = c; c = d; d = t; }
  if (c < a) { const int t = a; a = c; c = t; }
  if (d < b) { const int t = b; b = d; d = t; }
  if (c < b) { const int t = b; b = c; c = t; }

  row_keys[row] = RowSortKey{a, b, c, d};
  row_order[row] = row;
}


template<class Config = dd::DDPackageConfig>
class QBatchSimulator {
public:
    explicit QBatchSimulator(std::unique_ptr<qc::QuantumComputation>&& qc_, int batch_size_, int num_batch_) : 
    qc(std::move(qc_)), batch_size(batch_size_), num_batch(num_batch_), rtEngine(std::make_unique<RTSpMSpMEngine>())
    {
        waitForCudaInitializationSuccess();
        #if defined(BQSIM_USE_RTSPMSPM)
        rtEngine->setAvailable(true);
        #endif
        const auto nQubits = qc->getNqubits();
        nDim    = std::pow(2, nQubits);
        const char* pipeline_mode_init = std::getenv("BQSIM_RT_PIPELINE_MODE");
        const bool warmup_spm = pipeline_mode_init && std::strcmp(pipeline_mode_init, "SPMSPM") == 0 &&
                                rtEngine && rtEngine->isAvailable();
        if (warmup_spm) {
          rtEngine->warmup();
        }
        
        bqsim_rt::Complex *h_batch0;
        bqsim_rt::Complex *h_batch1;
        const size_t host_bytes = nDim * batch_size_ * sizeof(bqsim_rt::Complex);
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
            h_batch0[amp_id*batch_size_] = bqsim_rt::make_complex(
                static_cast<bqsim_rt::Real>(real),
                static_cast<bqsim_rt::Real>(imag));
            amp_id++;
            }
        }
        file.close();

        bqsim_rt::Complex *input_d;
        checkCudaErrors(cudaMalloc((void**)&input_d, nDim * batch_size_ * sizeof(bqsim_rt::Complex)));
        checkCudaErrors(cudaMemcpy(input_d, h_batch0, nDim * batch_size_ * sizeof(bqsim_rt::Complex),
                cudaMemcpyHostToDevice));
        replicate<<<nDim, batch_size>>>(input_d, batch_size_);
        checkCudaErrors(cudaMemcpy(h_batch0, input_d, nDim * batch_size_ * sizeof(bqsim_rt::Complex),
                cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(input_d));
        
        memset(h_batch1, 0, nDim * batch_size_ * sizeof(bqsim_rt::Complex));
        h_batch.push_back(h_batch0);
        h_batch_pinned.push_back(static_cast<uint8_t>(pinned0));
        h_batch.push_back(h_batch1);
        h_batch_pinned.push_back(static_cast<uint8_t>(pinned1));

        for (int buf = 0; buf < 4; buf++) {
          bqsim_rt::Complex *d_batch_buf;
          checkCudaErrors(cudaMalloc((void**)&d_batch_buf, nDim * batch_size_ * sizeof(bqsim_rt::Complex)));
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
        if (i < fused_gates_row_order_d.size() && fused_gates_row_order_d[i]) {
          checkCudaErrors(cudaFree(fused_gates_row_order_d[i]));
        }
      }
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
            singleShot();
            return;
        }

        // single shot is enough, but the sampling should only return actually measured qubits
        if (!hasNonmeasurementNonUnitary && measurementsLast) {
            singleShot();
            return;
        }
        return;
    }


    void singleShot() {
        std::size_t                 opNum = 0;
        std::vector<int> fused_num_nonzero;

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
          const bool force_full_fusion = envFlag("BQSIM_RT_FORCE_FULL_FUSION");
          const char* row_reorder_env = std::getenv("BQSIM_RT_ROW_REORDER");
          const bool use_row_reorder = !row_reorder_env || envFlag("BQSIM_RT_ROW_REORDER");
          const size_t total_gates = primitives.size();
          if (batch_size % 32 != 0) {
            std::cerr << "[SPMSPM] Dense path: batch_size not multiple of 32; expect lower memory coalescing." << std::endl;
          }

          double total_h2d_ms = 0.0;
          double total_ray_gen_ms = 0.0;
          double total_geom_ms = 0.0;
          double total_bvh_ms = 0.0;
          double total_launch_ms = 0.0;
          double total_merge_ms = 0.0;
          double total_overhead_ms = 0.0;
          double total_ell_convert_ms = 0.0;
          std::size_t total_bvh_update_count = 0;
          std::size_t total_bvh_rebuild_count = 0;
          std::size_t total_bvh_skip_count = 0;
          double total_refit_shift_sum = 0.0;
          std::size_t total_refit_shift_samples = 0;

          auto cleanup_spm = [&]() {
            for (size_t i = 0; i < fused_gates_val_d.size(); ++i) {
              if (fused_gates_val_d[i]) {
                cudaFree(fused_gates_val_d[i]);
              }
              if (i < fused_gates_indices_d.size() && fused_gates_indices_d[i]) {
                cudaFree(fused_gates_indices_d[i]);
              }
              if (i < fused_gates_row_order_d.size() && fused_gates_row_order_d[i]) {
                cudaFree(fused_gates_row_order_d[i]);
              }
            }
            fused_gates_val_d.clear();
            fused_gates_indices_d.clear();
            fused_gates_row_order_d.clear();
          };

          fused_gates_val_d.reserve(total_gates);
          fused_gates_indices_d.reserve(total_gates);
          fused_gates_row_order_d.reserve(total_gates);
          fused_num_nonzero.reserve(total_gates);

          size_t cursor = 0;
          size_t block_id = 0;
          const bool dump_build_gates = envFlag("BQSIM_RT_DUMP_BUILD_GATES");
          const auto csv_escape = [](const std::string& s) {
            std::string out;
            out.reserve(s.size() + 2);
            out.push_back('"');
            for (char ch : s) {
              if (ch == '"') {
                out.push_back('"');
              }
              out.push_back(ch);
            }
            out.push_back('"');
            return out;
          };
          const auto join_qubits = [](const int* data, int count) {
            std::ostringstream os;
            for (int i = 0; i < count; ++i) {
              if (i > 0) {
                os << ' ';
              }
              os << data[i];
            }
            return os.str();
          };
          const auto join_all_qubits = [&](const qc::GatePrimitive& gp) {
            std::ostringstream os;
            for (int i = 0; i < gp.control_count; ++i) {
              if (os.tellp() > 0) {
                os << ' ';
              }
              os << "c" << gp.controls[i];
            }
            for (int i = 0; i < gp.target_count; ++i) {
              if (os.tellp() > 0) {
                os << ' ';
              }
              os << "t" << gp.targets[i];
            }
            return os.str();
          };
          if (dump_build_gates) {
            std::filesystem::create_directories("../../log/build_gate");
            const std::string csv_path =
                "../../log/build_gate/" + qc->getName() + "_primitive_gates.csv";
            std::ofstream gate_csv(csv_path, std::ios::trunc);
            if (!gate_csv.is_open()) {
              std::cerr << "[SPMSPM] Failed to open build-gate CSV: " << csv_path << std::endl;
            } else {
              gate_csv << "block_id,block_start_gate,local_gate_idx,global_gate_idx,gate_name,"
                          "acting_qubits,target_qubits,control_qubits,target_count,control_count,"
                          "is_controlled\n";
            }
          }
          while (cursor < total_gates) {
            const size_t remaining = total_gates - cursor;
            const size_t planned = remaining;
            if (planned == 0) {
              break;
            }
            std::cout << "[SPMSPM] Fusing block " << (block_id + 1)
                      << " starting at gate " << cursor
                      << " with up to " << planned << " gates" << std::endl;

            rtEngine->resetStats();
            if (!(rtEngine->prepareGeometryFromGates(primitives.data() + cursor,
                                                     planned,
                                                     static_cast<int>(qc->getNqubits()),
                                                     nDim,
                                                     force_full_fusion) &&
                  rtEngine->launchRTMultiply())) {
              std::cerr << "[SPMSPM] prepareGeometryFromGates/launchRTMultiply failed; aborting SPMSPM pipeline."
                        << std::endl;
              cleanup_spm();
              return;
            }
            const auto& stats = rtEngine->lastStats();
            total_h2d_ms += stats.h2d_ms;
            total_ray_gen_ms += stats.ray_gen_ms;
            total_geom_ms += stats.geom_ms;
            total_bvh_ms += stats.gas_ms;
            total_launch_ms += stats.launch_ms;
            total_merge_ms += stats.merge_ms;
            total_overhead_ms += stats.overhead_ms;
            total_bvh_rebuild_count += stats.bvh_rebuild_count;
            total_bvh_update_count += stats.bvh_update_count;
            total_bvh_skip_count += stats.bvh_skip_count;
            total_refit_shift_sum += stats.bvh_refit_shift_sum;
            total_refit_shift_samples += stats.bvh_refit_shift_samples;
            if (dump_build_gates && !stats.build_gate_events.empty()) {
              try {
                const std::string csv_path =
                    "../../log/build_gate/" + qc->getName() + "_primitive_gates.csv";
                std::ofstream gate_csv(csv_path, std::ios::app);
                if (!gate_csv.is_open()) {
                  std::cerr << "[SPMSPM] Failed to append build-gate CSV: " << csv_path << std::endl;
                } else {
                  for (const auto& event : stats.build_gate_events) {
                    const auto& gp = event.gate;
                    const auto gate_type = static_cast<qc::OpType>(gp.gate_type);
                    gate_csv << block_id << ','
                             << cursor << ','
                             << event.gate_idx << ','
                             << (cursor + event.gate_idx) << ','
                             << csv_escape(qc::toString(gate_type)) << ','
                             << csv_escape(join_all_qubits(gp)) << ','
                             << csv_escape(join_qubits(gp.targets, gp.target_count)) << ','
                             << csv_escape(join_qubits(gp.controls, gp.control_count)) << ','
                             << gp.target_count << ','
                             << gp.control_count << ','
                             << (gp.is_controlled ? 1 : 0) << '\n';
                  }
                }
              } catch (const std::exception& e) {
                std::cerr << "[SPMSPM] Failed to append build-gate CSV: "
                          << e.what() << std::endl;
              }
            }

            int ell_width = rtEngine->maxRowNNZ();
            if (ell_width <= 0) {
              ell_width = 1;
            }
            auto ell_start = std::chrono::high_resolution_clock::now();
            bqsim_rt::Complex* fused_gate_val = nullptr;
            int* fused_gate_indices = nullptr;
            if (cudaMalloc((void**)&fused_gate_val, ell_width * nDim * sizeof(bqsim_rt::Complex)) != cudaSuccess ||
                cudaMalloc((void**)&fused_gate_indices, ell_width * nDim * sizeof(int)) != cudaSuccess) {
              if (fused_gate_indices) {
                cudaFree(fused_gate_indices);
              }
              std::cerr << "[SPMSPM] cudaMalloc failed during ELL allocation; aborting SPMSPM pipeline."
                        << std::endl;
              cleanup_spm();
              return;
            }
            checkCudaErrors(cudaMemset(fused_gate_val, 0, ell_width * nDim * sizeof(bqsim_rt::Complex)));
            checkCudaErrors(cudaMemset(fused_gate_indices, 0, ell_width * nDim * sizeof(int)));
            if (rtEngine->collectResultToELL(fused_gate_val, fused_gate_indices, ell_width, nDim)) {
              auto ell_stop = std::chrono::high_resolution_clock::now();
              total_ell_convert_ms += std::chrono::duration<double, std::milli>(ell_stop - ell_start).count();
              int* fused_gate_row_order = nullptr;
              if (use_row_reorder && ell_width == 4) {
                RowSortKey* fused_gate_row_keys = nullptr;
                if (cudaMalloc((void**)&fused_gate_row_order, nDim * sizeof(int)) == cudaSuccess &&
                    cudaMalloc((void**)&fused_gate_row_keys, nDim * sizeof(RowSortKey)) == cudaSuccess) {
                  constexpr int kThreadsPerBlock = 256;
                  const int blocks = static_cast<int>((nDim + kThreadsPerBlock - 1) / kThreadsPerBlock);
                  build_row_order_keys_w4<<<blocks, kThreadsPerBlock>>>(
                      fused_gate_indices, fused_gate_row_keys, fused_gate_row_order, static_cast<int>(nDim));
                  checkCudaErrors(cudaGetLastError());
                  thrust::stable_sort_by_key(thrust::device,
                                             thrust::device_pointer_cast(fused_gate_row_keys),
                                             thrust::device_pointer_cast(fused_gate_row_keys + nDim),
                                             thrust::device_pointer_cast(fused_gate_row_order));
                } else {
                  if (fused_gate_row_keys) {
                    checkCudaErrors(cudaFree(fused_gate_row_keys));
                  }
                  if (fused_gate_row_order) {
                    checkCudaErrors(cudaFree(fused_gate_row_order));
                    fused_gate_row_order = nullptr;
                  }
                  fused_gate_row_order = nullptr;
                  std::cerr << "[SPMSPM] row-order allocation failed; fallback to original row order." << std::endl;
                }
                if (fused_gate_row_keys) {
                  checkCudaErrors(cudaFree(fused_gate_row_keys));
                }
              }
              fused_gates_val_d.push_back(fused_gate_val);
              fused_gates_indices_d.push_back(fused_gate_indices);
              fused_gates_row_order_d.push_back(fused_gate_row_order);
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
              std::cerr << "[SPMSPM] lastFusedGateCount returned 0; aborting SPMSPM pipeline."
                        << std::endl;
              cleanup_spm();
              return;
            }
            std::cout << "[SPMSPM]   fused " << actual << " gate(s), ELL width: " << ell_width << std::endl;
            cursor += std::min(actual, remaining);
            ++block_id;
          }
          auto end_convert = std::chrono::high_resolution_clock::now();
          std::cout << "[Stage 1: RT Core Gate Fusion] time: "
                    << std::chrono::duration_cast<std::chrono::milliseconds>(end_convert - begin_convert).count()
                    << std::endl;
          std::cout << "  Breakdown:" << std::endl;
          std::cout << "  - H2D Transfer (Params):     " << total_h2d_ms << " ms" << std::endl;
          std::cout << "  - Ray Generation:            " << total_ray_gen_ms << " ms" << std::endl;
          if (total_geom_ms > 0.0) {
            std::cout << "  - COO -> Sphere:             " << total_geom_ms << " ms" << std::endl;
          }
          std::cout << "  - BVH Build (OptiX):         " << total_bvh_ms << " ms" << std::endl;
          std::cout << "  - bvh build update time :    " << total_bvh_update_count << " times" << std::endl;
          std::cout << "  - bvh build rebuild time :   " << total_bvh_rebuild_count << " times" << std::endl;
          std::cout << "  - bvh build skip time :      " << total_bvh_skip_count << " times" << std::endl;
          const double avg_refit_shift =
              (total_refit_shift_samples > 0)
                  ? (total_refit_shift_sum / static_cast<double>(total_refit_shift_samples))
                  : 0.0;
          std::cout << "  - bvh avg shift vs rebuild:  " << avg_refit_shift
                    << " (samples: " << total_refit_shift_samples << ")" << std::endl;
          std::cout << "  - Ray Tracing (Launch):      " << total_launch_ms << " ms" << std::endl;
          std::cout << "  - Sort & Merge (GPU):        " << total_merge_ms << " ms" << std::endl;
          std::cout << "  - Memory & Overhead:         " << total_overhead_ms << " ms" << std::endl;
          std::cout << "  - ELL Conversion (Result):   " << total_ell_convert_ms << " ms" << std::endl;
        } else {
          std::cerr << "[SPMSPM] Legacy DD fusion path removed. "
                    << "Please use BQSIM_RT_PIPELINE_MODE=SPMSPM with RT enabled." << std::endl;
          return;
        }

        auto run_stage3_graph = [&]() {
          tf::Taskflow taskflow("ELL-sim");
          tf::Executor executor;

          taskflow.emplace([&](){
            tf::cudaFlow cudaflow;
            std::vector<tf::cudaTask> input_copies;
            std::vector<tf::cudaTask> output_copies;
            std::vector<tf::cudaTask> simulate_fused_gate;
            input_copies.reserve(num_batch);
            output_copies.reserve(num_batch);
            simulate_fused_gate.reserve(num_batch * fused_num_nonzero.size());
            int grid_size = (nDim > 8192) ? 8192 : static_cast<int>(nDim);
            dim3 block_size = dim3(batch_size, 1, 1);

            for (int batch_id = 0; batch_id < num_batch; batch_id++) {
              input_copies.emplace_back(cudaflow.copy(
                d_batch[(batch_id % 2) * 2 + ((batch_id / 2) * (fused_num_nonzero.size() + 1)) % 2], h_batch[0], nDim * batch_size
              ).name("input_H2D_Host->" + std::to_string((batch_id % 2) * 2 + ((batch_id / 2) * (fused_num_nonzero.size() + 1)) % 2)));

              for (opNum = 0; opNum < fused_num_nonzero.size(); opNum++) {
                const int input_buffer_idx =
                    (batch_id % 2) * 2 + ((batch_id / 2) * (fused_num_nonzero.size() + 1) + opNum) % 2;
                const int output_buffer_idx =
                    (batch_id % 2) * 2 + ((batch_id / 2) * (fused_num_nonzero.size() + 1) + opNum + 1) % 2;

                if (opNum < fused_gates_row_order_d.size() &&
                    fused_gates_row_order_d[opNum] != nullptr) {
                  simulate_fused_gate.emplace_back(cudaflow.kernel(
                    grid_size,
                    block_size,
                    0,
                    run_fused_gate_reordered,
                    fused_gates_val_d[opNum], fused_gates_indices_d[opNum], fused_num_nonzero[opNum],
                    fused_gates_row_order_d[opNum],
                    d_batch[input_buffer_idx],
                    d_batch[output_buffer_idx], batch_size, nDim
                  ).name("fused_gate_reordered_" + std::to_string(opNum)));
                } else {
                  simulate_fused_gate.emplace_back(cudaflow.kernel(
                    grid_size,
                    block_size,
                    0,
                    run_fused_gate,
                    fused_gates_val_d[opNum], fused_gates_indices_d[opNum], fused_num_nonzero[opNum],
                    d_batch[input_buffer_idx],
                    d_batch[output_buffer_idx], batch_size, nDim
                  ).name("fused_gate_" + std::to_string(opNum)));
                }
              }

              output_copies.emplace_back(cudaflow.copy(
                h_batch[1], d_batch[(batch_id % 2) * 2 + ((batch_id / 2) * (fused_num_nonzero.size() + 1) + fused_num_nonzero.size()) % 2], nDim * batch_size
              ).name("output_D2H_" + std::to_string((batch_id % 2) * 2 + ((batch_id / 2) * (fused_num_nonzero.size() + 1) + fused_num_nonzero.size()) % 2) + "->Host"));
            }

            for (int batch_id = 0; batch_id < num_batch; batch_id++) {
              input_copies[batch_id].precede(simulate_fused_gate[batch_id * fused_num_nonzero.size()]);
              if (batch_id > 1) {
                simulate_fused_gate[(batch_id - 1) * fused_num_nonzero.size() - 1].precede(input_copies[batch_id]);
              }

              if (batch_id > 0) {
                simulate_fused_gate[batch_id * fused_num_nonzero.size() - 1].precede(simulate_fused_gate[batch_id * fused_num_nonzero.size()]);
              }
              for (opNum = 1; opNum < fused_num_nonzero.size(); opNum++) {
                simulate_fused_gate[batch_id * fused_num_nonzero.size() + opNum - 1].precede(simulate_fused_gate[batch_id * fused_num_nonzero.size() + opNum]);
              }

              simulate_fused_gate[(batch_id + 1) * fused_num_nonzero.size() - 1].precede(output_copies[batch_id]);
              if (batch_id < num_batch - 2) {
                output_copies[batch_id].precede(simulate_fused_gate[(batch_id + 2) * fused_num_nonzero.size()]);
              }
            }

            tf::cudaStream stream;
            cudaflow.run(stream);
            stream.synchronize();
          });

          auto begin_sim = std::chrono::high_resolution_clock::now();
          executor.run(taskflow).wait();
          auto end_sim = std::chrono::high_resolution_clock::now();

          QBatchSimulator<Config>::final_state_idx = 1;
          QBatchSimulator<Config>::final_state_idx_gpu = ((num_batch - 1) % 2) * 2 +
              (((num_batch - 1) / 2) * (fused_num_nonzero.size() + 1) + fused_num_nonzero.size()) % 2;
          std::cout << "[Stage 2: ELL-based batch simulation] time: "
                    << std::chrono::duration_cast<std::chrono::milliseconds>(end_sim - begin_sim).count()
                    << std::endl;
        };

        run_stage3_graph();
        }

    [[nodiscard]]
    bqsim_rt::Complex* getVector() const {
        if (getNumberOfQubits() >= MAX_LEV) {
            // On 64bit system the vector can hold up to (2^60)-1 elements, if memory permits
            throw std::range_error("getVector only supports less than 60 qubits.");
        }
        return h_batch[final_state_idx];
    }

    [[nodiscard]] std::size_t getNumberOfQubits() const { return qc->getNqubits(); };

    [[nodiscard]] std::size_t getNumberOfOps() const { return qc->getNops(); };

    [[nodiscard]] std::string getName() const { return qc->getName(); };

    std::vector<bqsim_rt::Complex *> h_batch, d_batch;
    std::vector<uint8_t> h_batch_pinned;

    int                                     final_state_idx;
    int        final_state_idx_gpu;

    size_t nDim = 1;
    std::unique_ptr<RTSpMSpMEngine> rtEngine;
    bool buildGatePrimitives(std::vector<qc::GatePrimitive>& out) const {
      out.clear();
      if (!qc) {
        return false;
      }
      auto set_matrix2 = [](qc::GatePrimitive& gp,
                            bqsim_rt::Real a00, bqsim_rt::Real b00,
                            bqsim_rt::Real a01, bqsim_rt::Real b01,
                            bqsim_rt::Real a10, bqsim_rt::Real b10,
                            bqsim_rt::Real a11, bqsim_rt::Real b11) {
        gp.matrix_dim = 2;
        gp.matrix[0] = bqsim_rt::make_matrix_elem(a00, b00);
        gp.matrix[1] = bqsim_rt::make_matrix_elem(a01, b01);
        gp.matrix[2] = bqsim_rt::make_matrix_elem(a10, b10);
        gp.matrix[3] = bqsim_rt::make_matrix_elem(a11, b11);
      };
      auto push_matrix2_gate = [&](qc::OpType gate_type,
                                   int target,
                                   std::initializer_list<int> controls,
                                   bqsim_rt::Real a00, bqsim_rt::Real b00,
                                   bqsim_rt::Real a01, bqsim_rt::Real b01,
                                   bqsim_rt::Real a10, bqsim_rt::Real b10,
                                   bqsim_rt::Real a11, bqsim_rt::Real b11) {
        if (controls.size() > static_cast<size_t>(qc::MAX_CONTROLS)) {
          return false;
        }
        qc::GatePrimitive gp{};
        gp.gate_type = static_cast<int>(gate_type);
        gp.target_count = 1;
        gp.control_count = static_cast<int>(controls.size());
        gp.is_controlled = gp.control_count > 0;
        gp.targets[0] = target;
        int ci = 0;
        for (int c : controls) {
          gp.controls[ci++] = c;
        }
        set_matrix2(gp, a00, b00, a01, b01, a10, b10, a11, b11);
        out.push_back(gp);
        return true;
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
        auto fail_gate = [&](const char* reason) {
          std::cerr << "[SPMSPM] Unsupported gate in buildGatePrimitives: "
                    << qc::toString(type)
                    << " targets=" << gp.target_count
                    << " controls=" << gp.control_count
                    << " reason=" << reason << std::endl;
          return false;
        };
        if (gp.target_count <= 0 || gp.target_count > qc::MAX_TARGETS) {
          return fail_gate("target_count_out_of_range");
        }
        if (gp.control_count > qc::MAX_CONTROLS) {
          return fail_gate("control_count_out_of_range");
        }

        int ti = 0;
        for (auto t : op->getTargets()) {
          gp.targets[ti++] = static_cast<int>(t);
        }
        int ci = 0;
        for (const auto& c : op->getControls()) {
          if (c.type != qc::Control::Type::Pos) {
            return fail_gate("non_positive_control");
          }
          gp.controls[ci++] = static_cast<int>(c.qubit);
        }

        const auto& params = op->getParameter();
        if (type == qc::SWAP && gp.target_count == 2) {
          const int a = gp.targets[0];
          const int b = gp.targets[1];
          if (gp.control_count == 0) {
            if (!push_matrix2_gate(qc::X, b, {a}, 0, 0, 1, 0, 1, 0, 0, 0) ||
                !push_matrix2_gate(qc::X, a, {b}, 0, 0, 1, 0, 1, 0, 0, 0) ||
                !push_matrix2_gate(qc::X, b, {a}, 0, 0, 1, 0, 1, 0, 0, 0)) {
              return false;
            }
            continue;
          }
          if (gp.control_count == 1) {
            const int c = gp.controls[0];
            if (!push_matrix2_gate(qc::X, b, {c, a}, 0, 0, 1, 0, 1, 0, 0, 0) ||
                !push_matrix2_gate(qc::X, a, {c, b}, 0, 0, 1, 0, 1, 0, 0, 0) ||
                !push_matrix2_gate(qc::X, b, {c, a}, 0, 0, 1, 0, 1, 0, 0, 0)) {
              return false;
            }
            continue;
          }
          return fail_gate("unsupported_controlled_swap_arity");
        }
        if (type == qc::RZZ && gp.control_count == 0 && gp.target_count == 2) {
          const int a = gp.targets[0];
          const int b = gp.targets[1];
          const bqsim_rt::Real theta = params.empty() ? 0.0 : static_cast<bqsim_rt::Real>(params[0]);
          const bqsim_rt::Real c = std::cos(theta * 0.5);
          const bqsim_rt::Real s = std::sin(theta * 0.5);
          if (!push_matrix2_gate(qc::X, b, {a}, 0, 0, 1, 0, 1, 0, 0, 0) ||
              !push_matrix2_gate(qc::RZ, b, {}, c, -s, 0.0f, 0.0f, 0.0f, 0.0f, c, s) ||
              !push_matrix2_gate(qc::X, b, {a}, 0, 0, 1, 0, 1, 0, 0, 0)) {
            return false;
          }
          continue;
        }
        if (type == qc::RXX && gp.control_count == 0 && gp.target_count == 2) {
          const int a = gp.targets[0];
          const int b = gp.targets[1];
          const bqsim_rt::Real theta = params.empty() ? 0.0 : static_cast<bqsim_rt::Real>(params[0]);
          const bqsim_rt::Real c = std::cos(theta * 0.5);
          const bqsim_rt::Real s = std::sin(theta * 0.5);
          const bqsim_rt::Real inv = 1.0 / std::sqrt(2.0);
          if (!push_matrix2_gate(qc::H, a, {}, inv, 0.0f, inv, 0.0f, inv, 0.0f, -inv, 0.0f) ||
              !push_matrix2_gate(qc::H, b, {}, inv, 0.0f, inv, 0.0f, inv, 0.0f, -inv, 0.0f) ||
              !push_matrix2_gate(qc::X, b, {a}, 0, 0, 1, 0, 1, 0, 0, 0) ||
              !push_matrix2_gate(qc::RZ, b, {}, c, -s, 0.0f, 0.0f, 0.0f, 0.0f, c, s) ||
              !push_matrix2_gate(qc::X, b, {a}, 0, 0, 1, 0, 1, 0, 0, 0) ||
              !push_matrix2_gate(qc::H, a, {}, inv, 0.0f, inv, 0.0f, inv, 0.0f, -inv, 0.0f) ||
              !push_matrix2_gate(qc::H, b, {}, inv, 0.0f, inv, 0.0f, inv, 0.0f, -inv, 0.0f)) {
            return false;
          }
          continue;
        }
        if (gp.control_count > 0) {
          if (gp.target_count != 1) {
            return fail_gate("controlled_gate_requires_single_target");
          }
          switch (type) {
            case qc::H: {
              const bqsim_rt::Real inv = 1.0 / std::sqrt(2.0);
              set_matrix2(gp, inv, 0.0f, inv, 0.0f, inv, 0.0f, -inv, 0.0f);
              break;
            }
            case qc::X:
              set_matrix2(gp, 0, 0, 1, 0, 1, 0, 0, 0);
              break;
            case qc::Y:
              set_matrix2(gp, 0, 0, 0, -1, 0, 1, 0, 0);
              break;
            case qc::Z:
              set_matrix2(gp, 1, 0, 0, 0, 0, 0, -1, 0);
              break;
            case qc::RX: {
              const bqsim_rt::Real theta = params.empty() ? 0.0 : static_cast<bqsim_rt::Real>(params[0]);
              const bqsim_rt::Real c = std::cos(theta * 0.5);
              const bqsim_rt::Real s = std::sin(theta * 0.5);
              set_matrix2(gp, c, 0.0f, 0.0f, -s, 0.0f, -s, c, 0.0f);
              break;
            }
            case qc::RY: {
              const bqsim_rt::Real theta = params.empty() ? 0.0 : static_cast<bqsim_rt::Real>(params[0]);
              const bqsim_rt::Real c = std::cos(theta * 0.5);
              const bqsim_rt::Real s = std::sin(theta * 0.5);
              set_matrix2(gp, c, 0.0f, -s, 0.0f, s, 0.0f, c, 0.0f);
              break;
            }
            case qc::RZ: {
              const bqsim_rt::Real theta = params.empty() ? 0.0 : static_cast<bqsim_rt::Real>(params[0]);
              const bqsim_rt::Real c = std::cos(theta * 0.5);
              const bqsim_rt::Real s = std::sin(theta * 0.5);
              set_matrix2(gp, c, -s, 0.0f, 0.0f, 0.0f, 0.0f, c, s);
              break;
            }
            case qc::Phase: {
              const bqsim_rt::Real theta = params.empty() ? 0.0 : static_cast<bqsim_rt::Real>(params[0]);
              set_matrix2(gp, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, std::cos(theta), std::sin(theta));
              break;
            }
            case qc::S:
              set_matrix2(gp, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
              break;
            case qc::Sdag:
              set_matrix2(gp, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f);
              break;
            case qc::SX:
            case qc::V:
              set_matrix2(gp, 0.5f, 0.5f, 0.5f, -0.5f, 0.5f, -0.5f, 0.5f, 0.5f);
              break;
            case qc::SXdag:
            case qc::Vdag:
              set_matrix2(gp, 0.5f, -0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, -0.5f);
              break;
            case qc::T: {
              const bqsim_rt::Real angle = static_cast<bqsim_rt::Real>(qc::PI_4);
              set_matrix2(gp, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, std::cos(angle), std::sin(angle));
              break;
            }
            case qc::Tdag: {
              const bqsim_rt::Real angle = -static_cast<bqsim_rt::Real>(qc::PI_4);
              set_matrix2(gp, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, std::cos(angle), std::sin(angle));
              break;
            }
            case qc::U2: {
              const bqsim_rt::Real phi = params.size() > 0 ? static_cast<bqsim_rt::Real>(params[0]) : 0.0;
              const bqsim_rt::Real lambda = params.size() > 1 ? static_cast<bqsim_rt::Real>(params[1]) : 0.0;
              const bqsim_rt::Real inv = 1.0 / std::sqrt(2.0);
              const bqsim_rt::Real c0 = std::cos(lambda);
              const bqsim_rt::Real s0 = std::sin(lambda);
              const bqsim_rt::Real c1 = std::cos(phi);
              const bqsim_rt::Real s1 = std::sin(phi);
              const bqsim_rt::Real c2 = std::cos(phi + lambda);
              const bqsim_rt::Real s2 = std::sin(phi + lambda);
              set_matrix2(gp,
                          inv, 0.0f,
                          -inv * c0, -inv * s0,
                          inv * c1, inv * s1,
                          inv * c2, inv * s2);
              break;
            }
            case qc::U3: {
              const bqsim_rt::Real theta = params.size() > 0 ? static_cast<bqsim_rt::Real>(params[0]) : 0.0;
              const bqsim_rt::Real phi = params.size() > 1 ? static_cast<bqsim_rt::Real>(params[1]) : 0.0;
              const bqsim_rt::Real lambda = params.size() > 2 ? static_cast<bqsim_rt::Real>(params[2]) : 0.0;
              const bqsim_rt::Real c = std::cos(theta * 0.5);
              const bqsim_rt::Real s = std::sin(theta * 0.5);
              const bqsim_rt::Real c0 = std::cos(lambda);
              const bqsim_rt::Real s0 = std::sin(lambda);
              const bqsim_rt::Real c1 = std::cos(phi);
              const bqsim_rt::Real s1 = std::sin(phi);
              const bqsim_rt::Real c2 = std::cos(phi + lambda);
              const bqsim_rt::Real s2 = std::sin(phi + lambda);
              set_matrix2(gp,
                          c, 0.0f,
                          -s * c0, -s * s0,
                          s * c1, s * s1,
                          c * c2, c * s2);
              break;
            }
            default:
              return fail_gate("unsupported_controlled_gate_type");
          }
          out.push_back(gp);
          continue;
        }

        if (gp.target_count != 1) {
          return fail_gate("uncontrolled_gate_requires_single_target");
        }

        switch (type) {
          case qc::X:
            set_matrix2(gp, 0, 0, 1, 0, 1, 0, 0, 0);
            break;
          case qc::Y:
            set_matrix2(gp, 0, 0, 0, -1, 0, 1, 0, 0);
            break;
          case qc::H: {
            const bqsim_rt::Real inv = 1.0 / std::sqrt(2.0);
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
            const bqsim_rt::Real angle = static_cast<bqsim_rt::Real>(qc::PI_4);
            set_matrix2(gp, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, std::cos(angle), std::sin(angle));
            break;
          }
          case qc::RX: {
            const bqsim_rt::Real theta = params.empty() ? 0.0 : static_cast<bqsim_rt::Real>(params[0]);
            const bqsim_rt::Real c = std::cos(theta * 0.5);
            const bqsim_rt::Real s = std::sin(theta * 0.5);
            set_matrix2(gp, c, 0.0f, 0.0f, -s, 0.0f, -s, c, 0.0f);
            break;
          }
          case qc::RY: {
            const bqsim_rt::Real theta = params.empty() ? 0.0 : static_cast<bqsim_rt::Real>(params[0]);
            const bqsim_rt::Real c = std::cos(theta * 0.5);
            const bqsim_rt::Real s = std::sin(theta * 0.5);
            set_matrix2(gp, c, 0.0f, -s, 0.0f, s, 0.0f, c, 0.0f);
            break;
          }
          case qc::RZ: {
            const bqsim_rt::Real theta = params.empty() ? 0.0 : static_cast<bqsim_rt::Real>(params[0]);
            const bqsim_rt::Real c = std::cos(theta * 0.5);
            const bqsim_rt::Real s = std::sin(theta * 0.5);
            set_matrix2(gp, c, -s, 0.0f, 0.0f, 0.0f, 0.0f, c, s);
            break;
          }
          case qc::Phase: {
            const bqsim_rt::Real theta = params.empty() ? 0.0 : static_cast<bqsim_rt::Real>(params[0]);
            set_matrix2(gp, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, std::cos(theta), std::sin(theta));
            break;
          }
          case qc::Sdag:
            set_matrix2(gp, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f);
            break;
          case qc::SX:
          case qc::V:
            set_matrix2(gp, 0.5f, 0.5f, 0.5f, -0.5f, 0.5f, -0.5f, 0.5f, 0.5f);
            break;
          case qc::SXdag:
          case qc::Vdag:
            set_matrix2(gp, 0.5f, -0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, -0.5f);
            break;
          case qc::Tdag: {
            const bqsim_rt::Real angle = -static_cast<bqsim_rt::Real>(qc::PI_4);
            set_matrix2(gp, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, std::cos(angle), std::sin(angle));
            break;
          }
          case qc::U2: {
            const bqsim_rt::Real phi = params.size() > 0 ? static_cast<bqsim_rt::Real>(params[0]) : 0.0;
            const bqsim_rt::Real lambda = params.size() > 1 ? static_cast<bqsim_rt::Real>(params[1]) : 0.0;
            const bqsim_rt::Real inv = 1.0 / std::sqrt(2.0);
            const bqsim_rt::Real c0 = std::cos(lambda);
            const bqsim_rt::Real s0 = std::sin(lambda);
            const bqsim_rt::Real c1 = std::cos(phi);
            const bqsim_rt::Real s1 = std::sin(phi);
            const bqsim_rt::Real c2 = std::cos(phi + lambda);
            const bqsim_rt::Real s2 = std::sin(phi + lambda);
            set_matrix2(gp,
                        inv, 0.0f,
                        -inv * c0, -inv * s0,
                        inv * c1, inv * s1,
                        inv * c2, inv * s2);
            break;
          }
          case qc::U3: {
            const bqsim_rt::Real theta = params.size() > 0 ? static_cast<bqsim_rt::Real>(params[0]) : 0.0;
            const bqsim_rt::Real phi = params.size() > 1 ? static_cast<bqsim_rt::Real>(params[1]) : 0.0;
            const bqsim_rt::Real lambda = params.size() > 2 ? static_cast<bqsim_rt::Real>(params[2]) : 0.0;
            const bqsim_rt::Real c = std::cos(theta * 0.5);
            const bqsim_rt::Real s = std::sin(theta * 0.5);
            const bqsim_rt::Real c0 = std::cos(lambda);
            const bqsim_rt::Real s0 = std::sin(lambda);
            const bqsim_rt::Real c1 = std::cos(phi);
            const bqsim_rt::Real s1 = std::sin(phi);
            const bqsim_rt::Real c2 = std::cos(phi + lambda);
            const bqsim_rt::Real s2 = std::sin(phi + lambda);
            set_matrix2(gp,
                        c, 0.0f,
                        -s * c0, -s * s0,
                        s * c1, s * s1,
                        c * c2, c * s2);
            break;
          }
          default:
            return fail_gate("unsupported_uncontrolled_gate_type");
        }
        out.push_back(gp);
      }

      return !out.empty();
    }

protected:
    std::unique_ptr<qc::QuantumComputation> qc;
    int batch_size = 1;
    int num_batch = 1;
    std::vector<bqsim_rt::Complex*> fused_gates_val_d;
    std::vector<int*> fused_gates_indices_d;
    std::vector<int*> fused_gates_row_order_d;


};

template class QBatchSimulator<dd::DDPackageConfig>;

#endif //QBATCH_SIMULATOR_H
