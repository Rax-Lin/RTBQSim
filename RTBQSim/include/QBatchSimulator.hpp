#ifndef QBATCH_SIMULATOR_H
#define QBATCH_SIMULATOR_H



#include "QuantumComputation.hpp"
#include "Definitions.hpp"
#include "CudaUtils.hpp"
#include "operations/OpType.hpp"
#include "CuSparseSpGEMMEngine.hpp"
#include "GatePrimitiveBuilder.hpp"
#include "GateFusionPlanner.hpp"
#include "RTSpMSpMEngine.hpp"
#include "GatePrimitive.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <deque>
#include <memory>
#include <random>
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
#include <thrust/system/cuda/execution_policy.h>
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

inline std::string cudaMemInfoString() {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  const cudaError_t rc = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (rc != cudaSuccess) {
    cudaGetLastError();
    return std::string("cudaMemGetInfo failed: ") + cudaGetErrorString(rc);
  }
  std::ostringstream oss;
  oss << "free=" << free_bytes << " bytes, total=" << total_bytes << " bytes";
  return oss.str();
}

inline std::filesystem::path projectRootPath() {
  return std::filesystem::path(__FILE__).parent_path().parent_path();
}

inline bool envFlag(const char* name) {
  const char* value = std::getenv(name);
  if (!value) {
    return false;
  }
  return std::strcmp(value, "1") == 0 ||
         std::strcmp(value, "true") == 0 ||
         std::strcmp(value, "TRUE") == 0 ||
         std::strcmp(value, "on") == 0 ||
         std::strcmp(value, "ON") == 0;
}

inline bool envFlagDefaultTrue(const char* name) {
  const char* value = std::getenv(name);
  if (!value) {
    return true;
  }
  return envFlag(name);
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
  const int *row_order,
  bqsim_rt::Complex *input_state,
  bqsim_rt::Complex *output_state,
  int batch_size,
  int nDim
) {
  int rows = nDim / gridDim.x;
  const int tid = threadIdx.x;
  int bid = blockIdx.x;
  extern __shared__ unsigned char shared_storage[];
  int* shared_indices = reinterpret_cast<int*>(shared_storage);
  const std::size_t index_bytes =
      static_cast<std::size_t>(num_non_zero) * sizeof(int);
  const std::size_t value_offset =
      ((index_bytes + alignof(bqsim_rt::Complex) - 1) / alignof(bqsim_rt::Complex)) *
      alignof(bqsim_rt::Complex);
  bqsim_rt::Complex* shared_val =
      reinterpret_cast<bqsim_rt::Complex*>(shared_storage + value_offset);
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

inline std::size_t fusedGateSharedBytes(int num_non_zero) {
  const std::size_t index_bytes =
      static_cast<std::size_t>(std::max(num_non_zero, 0)) * sizeof(int);
  const std::size_t value_offset =
      ((index_bytes + alignof(bqsim_rt::Complex) - 1) / alignof(bqsim_rt::Complex)) *
      alignof(bqsim_rt::Complex);
  return value_offset +
         static_cast<std::size_t>(std::max(num_non_zero, 0)) * sizeof(bqsim_rt::Complex);
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

__global__ void init_identity_row_order(int* row_order, int nDim) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= nDim) {
    return;
  }
  row_order[row] = row;
}

inline int directPrimitiveELLWidth(const qc::GatePrimitive& gate) {
  if (gate.matrix_dim <= 0) {
    return 0;
  }
  int max_row_nnz = 0;
  for (int row = 0; row < gate.matrix_dim; ++row) {
    int row_nnz = 0;
    for (int col = 0; col < gate.matrix_dim; ++col) {
      const auto entry = gate.matrix[row * gate.matrix_dim + col];
      if (entry.x != 0.0 || entry.y != 0.0) {
        ++row_nnz;
      }
    }
    max_row_nnz = std::max(max_row_nnz, row_nnz);
  }
  return std::max(1, max_row_nnz);
}

__global__ void pack_gate_primitive_to_ell(const qc::GatePrimitive gate,
                                           bqsim_rt::Complex* gate_values,
                                           int* gate_indices,
                                           int ell_width,
                                           int nDim) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= nDim) {
    return;
  }

  const int offset = row * ell_width;
  for (int i = 0; i < ell_width; ++i) {
    gate_values[offset + i] = bqsim_rt::make_complex(0.0, 0.0);
    gate_indices[offset + i] = row;
  }

  bool controls_active = true;
  for (int i = 0; i < gate.control_count; ++i) {
    if (((row >> gate.controls[i]) & 1) == 0) {
      controls_active = false;
      break;
    }
  }
  if (!controls_active) {
    gate_values[offset] = bqsim_rt::make_complex(1.0, 0.0);
    gate_indices[offset] = row;
    return;
  }

  const int dim = gate.matrix_dim;
  int local_row = 0;
  int base_col = row;
  for (int i = 0; i < gate.target_count; ++i) {
    const int target = gate.targets[i];
    const int bit = (row >> target) & 1;
    local_row |= (bit << i);
    base_col &= ~(1 << target);
  }

  int write_idx = 0;
  for (int local_col = 0; local_col < dim; ++local_col) {
    const auto entry = gate.matrix[local_row * dim + local_col];
    if (entry.x == 0.0 && entry.y == 0.0) {
      continue;
    }
    int global_col = base_col;
    for (int i = 0; i < gate.target_count; ++i) {
      if ((local_col >> i) & 1) {
        global_col |= (1 << gate.targets[i]);
      }
    }
    gate_values[offset + write_idx] = bqsim_rt::make_complex(entry.x, entry.y);
    gate_indices[offset + write_idx] = global_col;
    ++write_idx;
  }
}


class QBatchSimulator {
public:
    explicit QBatchSimulator(std::unique_ptr<qc::QuantumComputation>&& qc_, int batch_size_, int num_batch_) : 
    qc(std::move(qc_)),
    batch_size(batch_size_),
    num_batch(num_batch_),
    rtEngine(std::make_unique<RTSpMSpMEngine>()),
    cuSparseEngine(std::make_unique<CuSparseSpGEMMEngine>())
    {
        waitForCudaInitializationSuccess();
        rtEngine->setAvailable(true);
        cuSparseEngine->setAvailable(true);
        const auto nQubits = qc->getNqubits();
        nDim    = std::pow(2, nQubits);
        const bool warmup_rt = rtEngine && rtEngine->isAvailable();
        const bool warmup_cusparse = cuSparseEngine && cuSparseEngine->isAvailable();
        if (warmup_rt) {
          rtEngine->warmup();
        }
        if (warmup_cusparse) {
          cuSparseEngine->warmup();
        }
        
        bqsim_rt::Complex *h_batch0;
        bqsim_rt::Complex *h_batch1;
        const size_t host_bytes = nDim * batch_size_ * sizeof(bqsim_rt::Complex);
        checkCudaErrors(cudaMallocHost((void**)&h_batch0, host_bytes));
        checkCudaErrors(cudaMallocHost((void**)&h_batch1, host_bytes));
        const bool pinned0 = true;
        const bool pinned1 = true;

        auto load_input_batch = [&](const std::filesystem::path& path) -> bool {
            std::ifstream file(path);
            if (!file.is_open()) {
                return false;
            }

            std::string line;
            std::size_t amp_id = 0;
            while (std::getline(file, line)) {
                std::istringstream iss(line);
                double real = 0.0;
                double imag = 0.0;
                while (iss >> real >> imag) {
                    if (amp_id >= static_cast<std::size_t>(nDim)) {
                        break;
                    }
                    h_batch0[amp_id * batch_size_] = bqsim_rt::make_complex(
                        static_cast<bqsim_rt::Real>(real),
                        static_cast<bqsim_rt::Real>(imag));
                    ++amp_id;
                }
            }

            if (amp_id != static_cast<std::size_t>(nDim)) {
                std::cerr << "[input_batch] File " << path
                          << " contains " << amp_id
                          << " amplitudes, expected " << nDim << std::endl;
                return false;
            }
            return true;
        };

        auto generate_input_batch = [&](const std::filesystem::path& out_path) -> bool {
            std::vector<bqsim_rt::Complex> state(static_cast<std::size_t>(nDim));
            std::mt19937_64 rng(0xB051B051ULL + static_cast<unsigned long long>(nQubits));
            std::normal_distribution<double> dist(0.0, 1.0);

            long double norm2 = 0.0L;
            for (int i = 0; i < nDim; ++i) {
                const double real = dist(rng);
                const double imag = dist(rng);
                state[static_cast<std::size_t>(i)] = bqsim_rt::make_complex(
                    static_cast<bqsim_rt::Real>(real),
                    static_cast<bqsim_rt::Real>(imag));
                norm2 += static_cast<long double>(real) * real +
                         static_cast<long double>(imag) * imag;
            }

            if (norm2 <= 0.0L) {
                std::cerr << "[input_batch] Failed to generate a normalized input state for n="
                          << nQubits << std::endl;
                return false;
            }

            const long double inv_norm = 1.0L / std::sqrt(norm2);
            for (int i = 0; i < nDim; ++i) {
                auto& value = state[static_cast<std::size_t>(i)];
                value.x = static_cast<bqsim_rt::Real>(
                    static_cast<long double>(value.x) * inv_norm);
                value.y = static_cast<bqsim_rt::Real>(
                    static_cast<long double>(value.y) * inv_norm);
                h_batch0[static_cast<std::size_t>(i) * batch_size_] = value;
            }

            std::error_code ec;
            std::filesystem::create_directories(out_path.parent_path(), ec);
            std::ofstream out(out_path);
            if (!out.is_open()) {
                std::cerr << "[input_batch] Generated input state in memory but failed to write "
                          << out_path << std::endl;
                return true;
            }

            for (int i = 0; i < nDim; ++i) {
                const auto& value = state[static_cast<std::size_t>(i)];
                if (i > 0) {
                    out << ' ';
                }
                out << value.x << ' ' << value.y;
            }
            out << '\n';
            std::cout << "[input_batch] Generated missing batch file: " << out_path << std::endl;
            return true;
        };

        const std::string batch_filename = "n" + std::to_string(nQubits) + ".txt";
        const auto project_root = projectRootPath();
        const std::vector<std::filesystem::path> candidate_paths = {
            std::filesystem::path("../../input_batch") / batch_filename,
            std::filesystem::current_path() / "../../input_batch" / batch_filename,
            project_root / "input_batch" / batch_filename
        };

        bool loaded_batch = false;
        std::filesystem::path loaded_path;
        for (const auto& candidate : candidate_paths) {
            std::error_code ec;
            const auto normalized = std::filesystem::weakly_canonical(candidate, ec);
            const auto& path_to_try = ec ? candidate : normalized;
            if (load_input_batch(path_to_try)) {
                loaded_batch = true;
                loaded_path = path_to_try;
                break;
            }
        }

        if (!loaded_batch) {
            loaded_path = project_root / "input_batch" / batch_filename;
            loaded_batch = generate_input_batch(loaded_path);
        }

        if (!loaded_batch) {
            std::cerr << "[input_batch] Failed to open or generate input batch file for n="
                      << nQubits << " at " << loaded_path << std::endl;
            exit(-1);
        }

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
        std::vector<qc::GatePrimitive> direct_primitives;
        std::vector<int> direct_num_nonzero;
        bool use_direct_primitive_path = false;

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
        auto envInt = [](const char* name, int fallback) {
          const char* value = std::getenv(name);
          if (!value) {
            return fallback;
          }
          char* end = nullptr;
          const long parsed = std::strtol(value, &end, 10);
          if (end == value) {
            return fallback;
          }
          return static_cast<int>(parsed);
        };
        const bool enable_gate_fusion = envFlagDefaultTrue("RT_ENABLE_GATE_FUSION");
        const bool enable_breakdown = envFlagDefaultTrue("BQSIM_ENABLE_BREAKDOWN");
        const bool rt_available = rtEngine && rtEngine->isAvailable();
        const bool cusparse_available = cuSparseEngine && cuSparseEngine->isAvailable();
        const bool use_spm_pipeline = rt_available || cusparse_available;
        const char* backend_override_env = std::getenv("RT_GATE_FUSION_BACKEND");
        std::string backend_override = backend_override_env ? backend_override_env : "";
        std::transform(backend_override.begin(), backend_override.end(), backend_override.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        const int gate_fusion_threshold = envInt("RT_GATE_FUSION_THRESHOLD", std::numeric_limits<int>::max());
        bool use_cusparse_backend = false;
        if (backend_override == "cusparse") {
          use_cusparse_backend = cusparse_available;
        } else if (backend_override == "rt" || backend_override == "rtspmspm") {
          use_cusparse_backend = false;
        } else if (!rt_available && cusparse_available) {
          use_cusparse_backend = true;
        } else if (cusparse_available && static_cast<int>(qc->getNqubits()) > gate_fusion_threshold) {
          use_cusparse_backend = true;
        }
        if (!use_cusparse_backend && !rt_available && cusparse_available) {
          use_cusparse_backend = true;
        }
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
        if (!enable_gate_fusion) {
          auto begin_convert = std::chrono::high_resolution_clock::now();
          std::vector<qc::GatePrimitive> primitives;
          if (!bqsim_rt::buildGatePrimitives(*qc, primitives)) {
            std::cerr << "[SPMSPM] GatePrimitive build failed; aborting direct no-fusion path." << std::endl;
            return;
          }
          direct_primitives = std::move(primitives);
          direct_num_nonzero.reserve(direct_primitives.size());
          for (const auto& gate : direct_primitives) {
            const int ell_width = directPrimitiveELLWidth(gate);
            if (ell_width <= 0 || ell_width > 2) {
              std::cerr << "[SPMSPM] Unsupported no-fusion primitive ELL width="
                        << ell_width << "; aborting direct no-fusion path." << std::endl;
              return;
            }
            direct_num_nonzero.push_back(ell_width);
          }
          use_direct_primitive_path = true;
          auto end_convert = std::chrono::high_resolution_clock::now();
          const auto stage1_total_ms =
              std::chrono::duration_cast<std::chrono::milliseconds>(end_convert - begin_convert).count();

          std::cout << "[Stage 1: No Gate Fusion Preparation] time: "
                    << stage1_total_ms
                    << std::endl;
          if (enable_breakdown) {
            std::cout << "  Breakdown:" << std::endl;
            std::cout << "  - Gate Primitive Build:      "
                      << std::chrono::duration<double, std::milli>(end_convert - begin_convert).count()
                      << " ms" << std::endl;
            std::cout << "  - Direct Path Mode:          on-demand primitive->ELL, skip Stage-1 fusion only" << std::endl;
          }
        } else if (use_spm_pipeline) {
          std::vector<qc::GatePrimitive> primitives;
          if (!bqsim_rt::buildGatePrimitives(*qc, primitives)) {
            std::cerr << "[SPMSPM] GatePrimitive build failed; aborting SPMSPM pipeline." << std::endl;
            return;
          }
          const int row_nnz_limit = envInt("BQSIM_RT_SPM_ROW_NNZ_LIMIT", 4);
          bqsim_rt::GateFusionPlan fusion_plan;
          if (!bqsim_rt::buildGateFusionPlan(primitives, row_nnz_limit, fusion_plan)) {
            std::cerr << "[SPMSPM] DAG gate-fusion planning failed; aborting SPMSPM pipeline." << std::endl;
            return;
          }
          primitives = std::move(fusion_plan.ordered_primitives);
          const std::vector<std::size_t> planned_blocks = std::move(fusion_plan.block_sizes);
          auto begin_convert = std::chrono::high_resolution_clock::now();
          const size_t total_gates = primitives.size();
          if (batch_size % 32 != 0) {
            std::cerr << "[SPMSPM] Dense path: batch_size not multiple of 32; expect lower memory coalescing." << std::endl;
          }

          double total_h2d_ms = 0.0;
          double total_ray_gen_ms = 0.0;
          double total_geom_ms = 0.0;
          double total_bvh_ms = 0.0;
          double total_launch_ms = 0.0;
          double total_compact_ms = 0.0;
          double total_diagonal_ms = 0.0;
          double total_overhead_ms = 0.0;
          double total_cleanup_ms = 0.0;
          double total_ell_convert_ms = 0.0;
          std::size_t total_bvh_update_count = 0;
          std::size_t total_bvh_rebuild_count = 0;
          std::size_t total_bvh_skip_count = 0;

          fused_gates_val_d.reserve(total_gates);
          fused_gates_indices_d.reserve(total_gates);
          fused_gates_row_order_d.reserve(total_gates);
          fused_num_nonzero.reserve(total_gates);

	          size_t cursor = 0;
	          size_t block_id = 0;
	          const bool debug_info = envFlag("RT_DEBUG_INFO");
	          const bool dump_tree_owner_avg = envFlag("RT_DUMP_TREE_OWNER_AVG");
	          const bool dump_gate_traversal = envFlag("RT_DUMP_GATE_TRAVERSAL");
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
	          const auto dump_block_gate_details = [&](const qc::GatePrimitive* block_gates,
	                                                   std::size_t planned_count,
	                                                   std::size_t actual_count,
	                                                   int ell_width,
	                                                   const char* backend_name) {
	            if (!debug_info || ell_width <= 4) {
	              return;
	            }
	            const std::size_t dump_count = std::min(planned_count, actual_count);
	            std::cout << "[SPMSPM][debug] Wide fused block detected: backend=" << backend_name
	                      << " circuit=" << qc->getName()
	                      << " block_id=" << (block_id + 1)
	                      << " block_start_gate=" << cursor
	                      << " planned_gates=" << planned_count
	                      << " actual_fused_gates=" << actual_count
	                      << " ell_width=" << ell_width
	                      << std::endl;
	            for (std::size_t local_idx = 0; local_idx < dump_count; ++local_idx) {
	              const auto& gp = block_gates[local_idx];
	              const auto gate_type = static_cast<qc::OpType>(gp.gate_type);
	              const bool width_preserving = bqsim_rt::plannerGateIsWidthPreserving(gp);
	              const int row_nnz_ub = bqsim_rt::plannerGateRowNNZUpperBound(gp);
	              std::cout << "[SPMSPM][debug]   gate local=" << local_idx
	                        << " global=" << (cursor + local_idx)
	                        << " type=" << qc::toString(gate_type)
	                        << " acting={" << join_all_qubits(gp) << "}"
	                        << " targets={" << join_qubits(gp.targets, gp.target_count) << "}"
	                        << " controls={" << join_qubits(gp.controls, gp.control_count) << "}"
	                        << " matrix_dim=" << gp.matrix_dim
	                        << " row_nnz_ub=" << row_nnz_ub
	                        << " width_preserving=" << (width_preserving ? 1 : 0)
	                        << std::endl;
	            }
	          };
	          if (dump_tree_owner_avg) {
            const bool allow_update = envFlag("RT_GAS_ALLOW_UPDATE");
            const std::string dir = allow_update ? "../../log/refit_tree_owner"
                                                 : "../../log/no_refit_tree_owner";
            std::filesystem::create_directories(dir);
            const std::string csv_path =
                dir + "/" + qc->getName() + "_primitive_gates.csv";
            std::ofstream gate_csv(csv_path, std::ios::trunc);
            if (!gate_csv.is_open()) {
              std::cerr << "[SPMSPM] Failed to open build-gate CSV: " << csv_path << std::endl;
            } else {
              gate_csv << "block_id,block_start_gate,local_gate_idx,global_gate_idx,gate_name,"
                          "acting_qubits,target_qubits,control_qubits,target_count,control_count,"
                          "is_controlled,tree_build_row_nnz,tree_final_row_nnz,"
                          "traversal_average_ms,traversal_sample_count\n";
            }
          }
          if (dump_gate_traversal) {
            const bool allow_update = envFlag("RT_GAS_ALLOW_UPDATE");
            const std::string dir = allow_update ? "../../log/refit_per_gate"
                                                 : "../../log/no_refit_per_gate";
            std::filesystem::create_directories(dir);
            const std::string csv_path = dir + "/" + qc->getName() + "_per_gate.csv";
            std::ofstream gate_csv(csv_path, std::ios::trunc);
            if (!gate_csv.is_open()) {
              std::cerr << "[SPMSPM] Failed to open gate-traversal CSV: " << csv_path << std::endl;
            } else {
              gate_csv << "block_id,block_start_gate,local_gate_idx,global_gate_idx,gate_name,"
                          "acting_qubits,target_qubits,control_qubits,target_count,control_count,"
                          "is_controlled,tree_row_nnz_before,result_row_nnz_after,"
                          "traversal_ms,has_traversal\n";
            }
          }
          const auto stage1_setup_stop = std::chrono::high_resolution_clock::now();
          total_overhead_ms += std::chrono::duration<double, std::milli>(stage1_setup_stop - begin_convert).count();
          const auto run_spm_pipeline = [&](auto* engine, bool is_rt_backend) {
            const char* backend_name = is_rt_backend ? "rtspmspm" : "cusparse";
            std::deque<std::size_t> pending_blocks(planned_blocks.begin(), planned_blocks.end());
            std::cout << "[SPMSPM] Gate-fusion backend: " << backend_name
                      << " (threshold=" << gate_fusion_threshold
                      << ", nqubits=" << qc->getNqubits() << ")" << std::endl;
            while (cursor < total_gates && !pending_blocks.empty()) {
              const size_t remaining = total_gates - cursor;
              const size_t planned = std::min(pending_blocks.front(), remaining);
              if (planned == 0) {
                pending_blocks.pop_front();
                break;
              }
              const auto block_log_start = std::chrono::high_resolution_clock::now();
              std::cout << "[SPMSPM] Fusing block " << (block_id + 1)
                        << " starting at gate " << cursor
                        << " with up to " << planned << " gates" << std::endl;
              const auto block_log_stop = std::chrono::high_resolution_clock::now();
              total_overhead_ms += std::chrono::duration<double, std::milli>(block_log_stop - block_log_start).count();
              engine->setDebugContext(qc->getName(), cursor);
              engine->resetStats();
              if (!(engine->prepareGeometryFromGates(primitives.data() + cursor,
                                                     planned,
                                                     static_cast<int>(qc->getNqubits()),
                                                     nDim,
                                                     true) &&
                    engine->launchRTMultiply())) {
                std::cerr << "[SPMSPM] prepareGeometryFromGates/launchRTMultiply failed; aborting SPMSPM pipeline."
                          << std::endl;
                cleanup_spm();
                return false;
              }
              const auto post_engine_host_start = std::chrono::high_resolution_clock::now();
              int ell_width = engine->maxRowNNZ();
              if (ell_width <= 0) {
                ell_width = 1;
              }
              const auto& stats = engine->lastStats();
              total_h2d_ms += stats.h2d_ms;
              total_ray_gen_ms += stats.ray_gen_ms;
              total_geom_ms += stats.geom_ms;
              total_bvh_ms += stats.gas_ms;
              total_launch_ms += stats.launch_ms;
              total_compact_ms += stats.compact_ms;
              total_diagonal_ms += stats.diagonal_ms;
              total_overhead_ms += stats.overhead_ms;
              total_cleanup_ms += stats.cleanup_ms;
              total_bvh_rebuild_count += stats.bvh_rebuild_count;
              total_bvh_update_count += stats.bvh_update_count;
              total_bvh_skip_count += stats.bvh_skip_count;
              if (dump_tree_owner_avg && !stats.build_gate_events.empty()) {
                try {
                  const bool allow_update = envFlag("RT_GAS_ALLOW_UPDATE");
                  const std::string dir = allow_update ? "../../log/refit_tree_owner"
                                                       : "../../log/no_refit_tree_owner";
                  const std::string csv_path = dir + "/" + qc->getName() + "_primitive_gates.csv";
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
                               << (gp.is_controlled ? 1 : 0) << ','
                               << event.tree_build_row_nnz << ','
                               << event.tree_final_row_nnz << ','
                               << event.traversal_average_ms << ','
                               << event.traversal_sample_count << '\n';
                    }
                  }
                } catch (const std::exception& e) {
                  std::cerr << "[SPMSPM] Failed to append build-gate CSV: "
                            << e.what() << std::endl;
                }
              }
              if (dump_gate_traversal && !stats.gate_traversal_events.empty()) {
                try {
                  const bool allow_update = envFlag("RT_GAS_ALLOW_UPDATE");
                  const std::string dir = allow_update ? "../../log/refit_per_gate"
                                                       : "../../log/no_refit_per_gate";
                  const std::string csv_path = dir + "/" + qc->getName() + "_per_gate.csv";
                  std::ofstream gate_csv(csv_path, std::ios::app);
                  if (!gate_csv.is_open()) {
                    std::cerr << "[SPMSPM] Failed to append gate-traversal CSV: " << csv_path << std::endl;
                  } else {
                    for (const auto& event : stats.gate_traversal_events) {
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
                               << (gp.is_controlled ? 1 : 0) << ','
                               << event.tree_row_nnz_before << ','
                               << event.result_row_nnz_after << ','
                               << event.traversal_ms << ','
                               << (event.has_traversal ? 1 : 0) << '\n';
                    }
                  }
                } catch (const std::exception& e) {
                  std::cerr << "[SPMSPM] Failed to append gate-traversal CSV: "
                            << e.what() << std::endl;
                }
              }

              const auto post_engine_host_stop = std::chrono::high_resolution_clock::now();
              total_overhead_ms += std::chrono::duration<double, std::milli>(post_engine_host_stop - post_engine_host_start).count();
              auto ell_start = std::chrono::high_resolution_clock::now();
              bqsim_rt::Complex* fused_gate_val = nullptr;
              int* fused_gate_indices = nullptr;
              const std::size_t ell_value_bytes =
                  static_cast<std::size_t>(ell_width) * static_cast<std::size_t>(nDim) * sizeof(bqsim_rt::Complex);
              const std::size_t ell_index_bytes =
                  static_cast<std::size_t>(ell_width) * static_cast<std::size_t>(nDim) * sizeof(int);
              const cudaError_t ell_val_rc = cudaMalloc((void**)&fused_gate_val, ell_value_bytes);
              const cudaError_t ell_idx_rc =
                  (ell_val_rc == cudaSuccess) ? cudaMalloc((void**)&fused_gate_indices, ell_index_bytes)
                                              : cudaErrorMemoryAllocation;
              if (ell_val_rc != cudaSuccess || ell_idx_rc != cudaSuccess) {
                if (fused_gate_val) {
                  cudaFree(fused_gate_val);
                  fused_gate_val = nullptr;
                }
                if (fused_gate_indices) {
                  cudaFree(fused_gate_indices);
                  fused_gate_indices = nullptr;
                }
                cudaGetLastError();
                std::cerr << "[SPMSPM] cudaMalloc failed during ELL allocation "
                          << "(values=" << ell_value_bytes
                          << " bytes, indices=" << ell_index_bytes
                          << " bytes; " << cudaMemInfoString() << "); aborting SPMSPM pipeline."
                          << std::endl;
                cleanup_spm();
                return false;
              }
              checkCudaErrors(cudaMemset(fused_gate_val, 0, ell_width * nDim * sizeof(bqsim_rt::Complex)));
              checkCudaErrors(cudaMemset(fused_gate_indices, 0, ell_width * nDim * sizeof(int)));
              const auto post_ell_host_start = std::chrono::high_resolution_clock::now();
              if (engine->collectResultToELL(fused_gate_val, fused_gate_indices, ell_width, nDim)) {
                int* fused_gate_row_order = nullptr;
                RowSortKey* fused_gate_row_keys = nullptr;
                const std::size_t row_order_bytes = static_cast<std::size_t>(nDim) * sizeof(int);
                const std::size_t row_key_bytes = static_cast<std::size_t>(nDim) * sizeof(RowSortKey);
                constexpr int kThreadsPerBlock = 256;
                const int blocks = static_cast<int>((nDim + kThreadsPerBlock - 1) / kThreadsPerBlock);
                if (cudaMalloc((void**)&fused_gate_row_order, row_order_bytes) != cudaSuccess) {
                  if (fused_gate_row_order) {
                    checkCudaErrors(cudaFree(fused_gate_row_order));
                  }
                  cudaGetLastError();
                  std::cerr << "[SPMSPM] row-order allocation failed "
                            << "(row_order=" << row_order_bytes
                            << " bytes; " << cudaMemInfoString()
                            << "); aborting SPMSPM pipeline." << std::endl;
                  if (fused_gate_val) {
                    checkCudaErrors(cudaFree(fused_gate_val));
                  }
                  checkCudaErrors(cudaFree(fused_gate_indices));
                  cleanup_spm();
                  return false;
                }
                init_identity_row_order<<<blocks, kThreadsPerBlock>>>(
                    fused_gate_row_order, static_cast<int>(nDim));
                checkCudaErrors(cudaGetLastError());
                if (ell_width == 4) {
                  if (cudaMalloc((void**)&fused_gate_row_keys, row_key_bytes) == cudaSuccess) {
                    build_row_order_keys_w4<<<blocks, kThreadsPerBlock>>>(
                        fused_gate_indices, fused_gate_row_keys, fused_gate_row_order, static_cast<int>(nDim));
                    checkCudaErrors(cudaGetLastError());
                    thrust::stable_sort_by_key(thrust::device,
                                               thrust::device_pointer_cast(fused_gate_row_keys),
                                               thrust::device_pointer_cast(fused_gate_row_keys + nDim),
                                               thrust::device_pointer_cast(fused_gate_row_order));
                  } else {
                    cudaGetLastError();
                    std::cerr << "[SPMSPM] row-order key allocation failed "
                              << "(row_keys=" << row_key_bytes
                              << " bytes; " << cudaMemInfoString()
                              << "); using identity row order." << std::endl;
                  }
                }
                if (fused_gate_row_keys) {
                  checkCudaErrors(cudaFree(fused_gate_row_keys));
                }
                auto ell_stop = std::chrono::high_resolution_clock::now();
                total_ell_convert_ms += std::chrono::duration<double, std::milli>(ell_stop - ell_start).count();
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
                return false;
              }

              size_t actual = engine->lastFusedGateCount();
              const auto post_ell_host_stop = std::chrono::high_resolution_clock::now();
              total_overhead_ms += std::chrono::duration<double, std::milli>(post_ell_host_stop - post_ell_host_start).count();
              if (actual == 0) {
                std::cerr << "[SPMSPM] lastFusedGateCount returned 0; aborting SPMSPM pipeline."
                          << std::endl;
                cleanup_spm();
                return false;
              }
	              const auto fused_log_start = std::chrono::high_resolution_clock::now();
	              std::cout << "[SPMSPM]   fused " << actual << " gate(s), ELL width: " << ell_width << std::endl;
	              dump_block_gate_details(primitives.data() + cursor, planned, actual, ell_width, backend_name);
	              const auto fused_log_stop = std::chrono::high_resolution_clock::now();
	              total_overhead_ms += std::chrono::duration<double, std::milli>(fused_log_stop - fused_log_start).count();
              if (actual > planned) {
                std::cerr << "[SPMSPM] engine fused " << actual
                          << " gate(s) for planned block size " << planned
                          << "; aborting SPMSPM pipeline." << std::endl;
                cleanup_spm();
                return false;
              }
              if (actual != planned) {
                std::cerr << "[SPMSPM] engine fused only " << actual
                          << " gate(s) for planned block size " << planned
                          << "; runtime re-fusion/splitting is disabled, aborting SPMSPM pipeline."
                          << std::endl;
                cleanup_spm();
                return false;
              }
              cursor += actual;
              pending_blocks.pop_front();
              ++block_id;
            }
            if (cursor != total_gates || !pending_blocks.empty()) {
              std::cerr << "[SPMSPM] DAG plan execution ended early (cursor="
                        << cursor << "/" << total_gates
                        << ", remaining_blocks=" << pending_blocks.size()
                        << "); aborting SPMSPM pipeline." << std::endl;
              cleanup_spm();
              return false;
            }
            return true;
          };
          const bool stage1_ok = use_cusparse_backend
              ? run_spm_pipeline(cuSparseEngine.get(), false)
              : run_spm_pipeline(rtEngine.get(), true);
          if (!stage1_ok) {
            return;
          }
          auto end_convert = std::chrono::high_resolution_clock::now();
          const auto stage1_total_ms =
              std::chrono::duration_cast<std::chrono::milliseconds>(end_convert - begin_convert).count();
          if (use_cusparse_backend) {
            std::cout << "[Stage 1: cuSPARSE SpGEMM Gate Fusion] time: "
                      << stage1_total_ms
                      << std::endl;
            if (enable_breakdown) {
              std::cout << "  Breakdown:" << std::endl;
              std::cout << "  - Gate->CSR Build (GPU):     " << total_ray_gen_ms << " ms" << std::endl;
              std::cout << "  - SpGEMM Compute (cuSPARSE): " << total_launch_ms << " ms" << std::endl;
              std::cout << "  - RowNNZ Scan (D2H+CPU):     " << total_compact_ms << " ms" << std::endl;
              std::cout << "  - NNZ1 Multiplication:       " << total_diagonal_ms << " ms" << std::endl;
              std::cout << "  - Other Overhead:            " << (total_overhead_ms + total_cleanup_ms) << " ms" << std::endl;
              std::cout << "  - ELL Conversion (Result):   " << total_ell_convert_ms << " ms" << std::endl;
            }
          } else {
            const std::string geom_label =
                std::string("COO -> ") +
                ((std::strcmp(rtEngine->primitiveTypeName(), "sphere") == 0) ? "Sphere"
                                                                              : "Triangle");
            std::cout << "[Stage 1: RT Core Gate Fusion] time: "
                      << stage1_total_ms
                      << std::endl;
            if (enable_breakdown) {
              std::cout << "  Breakdown:" << std::endl;
              std::cout << "  - Ray Generation:            " << total_ray_gen_ms << " ms" << std::endl;
              if (total_geom_ms > 0.0) {
                std::cout << "  - " << geom_label;
                if (geom_label.size() < 25) {
                  std::cout << std::string(25 - geom_label.size(), ' ');
                }
                std::cout << total_geom_ms << " ms" << std::endl;
              }
              std::cout << "  - BVH Build (OptiX):         " << total_bvh_ms << " ms" << std::endl;
              std::cout << "  - bvh build update time :    " << total_bvh_update_count << " times" << std::endl;
              std::cout << "  - bvh build rebuild time :   " << total_bvh_rebuild_count << " times" << std::endl;
              std::cout << "  - bvh build skip time :      " << total_bvh_skip_count << " times" << std::endl;
              std::cout << "  - Ray Tracing (Launch):      " << (total_launch_ms + total_compact_ms) << " ms" << std::endl;
              std::cout << "  - NNZ1 Multiplication:       " << total_diagonal_ms << " ms" << std::endl;
              const double total_memory_overhead_ms = total_overhead_ms + total_h2d_ms + total_cleanup_ms;
              std::cout << "  - Memory & Overhead:         " << total_memory_overhead_ms << " ms" << std::endl;
              std::cout << "  - ELL Conversion (Result):   " << total_ell_convert_ms << " ms" << std::endl;
            }
          }
        } else {
          std::cerr << "[SPMSPM] No gate-fusion backend is available. "
                    << "Please ensure RT and/or cuSPARSE support is enabled in build and runtime." << std::endl;
          return;
        }

        const std::size_t stage2_gate_count =
            use_direct_primitive_path ? direct_primitives.size() : fused_num_nonzero.size();

        auto run_stage3_graph = [&]() {
          if (stage2_gate_count == 0) {
            final_state_idx = 0;
            final_state_idx_gpu = 0;
            std::cout << "[Stage 2: ELL-based batch simulation] time: 0" << std::endl;
            return;
          }
          if (use_direct_primitive_path) {
            bqsim_rt::Complex* direct_gate_val_d = nullptr;
            int* direct_gate_indices_d = nullptr;
            int* direct_gate_row_order_d = nullptr;
            const std::size_t direct_value_bytes =
                static_cast<std::size_t>(2) * static_cast<std::size_t>(nDim) * sizeof(bqsim_rt::Complex);
            const std::size_t direct_index_bytes =
                static_cast<std::size_t>(2) * static_cast<std::size_t>(nDim) * sizeof(int);
            const std::size_t direct_row_order_bytes =
                static_cast<std::size_t>(nDim) * sizeof(int);
            const cudaError_t val_rc = cudaMalloc((void**)&direct_gate_val_d, direct_value_bytes);
            const cudaError_t idx_rc =
                (val_rc == cudaSuccess) ? cudaMalloc((void**)&direct_gate_indices_d, direct_index_bytes)
                                        : cudaErrorMemoryAllocation;
            const cudaError_t row_order_rc =
                (idx_rc == cudaSuccess) ? cudaMalloc((void**)&direct_gate_row_order_d, direct_row_order_bytes)
                                        : cudaErrorMemoryAllocation;
            if (val_rc != cudaSuccess || idx_rc != cudaSuccess || row_order_rc != cudaSuccess) {
              if (direct_gate_val_d) {
                cudaFree(direct_gate_val_d);
              }
              if (direct_gate_indices_d) {
                cudaFree(direct_gate_indices_d);
              }
              if (direct_gate_row_order_d) {
                cudaFree(direct_gate_row_order_d);
              }
              cudaGetLastError();
              std::cerr << "[SPMSPM] cudaMalloc failed during no-fusion Stage-2 ELL buffer allocation "
                        << "(values=" << direct_value_bytes
                        << " bytes, indices=" << direct_index_bytes
                        << " bytes, row_order=" << direct_row_order_bytes
                        << " bytes; " << cudaMemInfoString() << "); aborting simulation."
                        << std::endl;
              return;
            }

            cudaStream_t stream{};
            checkCudaErrors(cudaStreamCreate(&stream));
            cudaEvent_t evt_start{};
            cudaEvent_t evt_stop{};
            checkCudaErrors(cudaEventCreate(&evt_start));
            checkCudaErrors(cudaEventCreate(&evt_stop));

            auto measure_ms = [&](const auto& work) {
              checkCudaErrors(cudaEventRecord(evt_start, stream));
              work();
              checkCudaErrors(cudaEventRecord(evt_stop, stream));
              checkCudaErrors(cudaEventSynchronize(evt_stop));
              float ms = 0.0F;
              checkCudaErrors(cudaEventElapsedTime(&ms, evt_start, evt_stop));
              return static_cast<double>(ms);
            };

            double total_pack_ms = 0.0;
            double total_sim_kernel_ms = 0.0;
            constexpr int kThreadsPerBlock = 256;
            const int pack_blocks = static_cast<int>((nDim + kThreadsPerBlock - 1) / kThreadsPerBlock);
            const int row_order_blocks = static_cast<int>((nDim + kThreadsPerBlock - 1) / kThreadsPerBlock);
            const int grid_size = (nDim > 8192) ? 8192 : static_cast<int>(nDim);
            dim3 block_size = dim3(batch_size, 1, 1);

            init_identity_row_order<<<row_order_blocks, kThreadsPerBlock, 0, stream>>>(
                direct_gate_row_order_d, static_cast<int>(nDim));
            checkCudaErrors(cudaGetLastError());

            auto begin_sim = std::chrono::high_resolution_clock::now();
            for (int batch_id = 0; batch_id < num_batch; ++batch_id) {
              const int initial_buffer_idx =
                  (batch_id % 2) * 2 + ((batch_id / 2) * (stage2_gate_count + 1)) % 2;
              checkCudaErrors(cudaMemcpyAsync(
                  d_batch[initial_buffer_idx], h_batch[0],
                  nDim * batch_size * sizeof(bqsim_rt::Complex),
                  cudaMemcpyHostToDevice, stream));

              for (std::size_t gate_idx = 0; gate_idx < stage2_gate_count; ++gate_idx) {
                const int input_buffer_idx =
                    (batch_id % 2) * 2 + ((batch_id / 2) * (stage2_gate_count + 1) + gate_idx) % 2;
                const int output_buffer_idx =
                    (batch_id % 2) * 2 + ((batch_id / 2) * (stage2_gate_count + 1) + gate_idx + 1) % 2;

                total_pack_ms += measure_ms([&]() {
                  pack_gate_primitive_to_ell<<<pack_blocks, kThreadsPerBlock, 0, stream>>>(
                      direct_primitives[gate_idx],
                      direct_gate_val_d,
                      direct_gate_indices_d,
                      direct_num_nonzero[gate_idx],
                      static_cast<int>(nDim));
                  checkCudaErrors(cudaGetLastError());
                });

                total_sim_kernel_ms += measure_ms([&]() {
                  run_fused_gate<<<grid_size,
                                   block_size,
                                   fusedGateSharedBytes(direct_num_nonzero[gate_idx]),
                                   stream>>>(
                      direct_gate_val_d, direct_gate_indices_d, direct_num_nonzero[gate_idx],
                      direct_gate_row_order_d,
                      d_batch[input_buffer_idx],
                      d_batch[output_buffer_idx], batch_size, nDim);
                  checkCudaErrors(cudaGetLastError());
                });
              }

              const int final_buffer_idx =
                  (batch_id % 2) * 2 + ((batch_id / 2) * (stage2_gate_count + 1) + stage2_gate_count) % 2;
              checkCudaErrors(cudaMemcpyAsync(
                  h_batch[1], d_batch[final_buffer_idx],
                  nDim * batch_size * sizeof(bqsim_rt::Complex),
                  cudaMemcpyDeviceToHost, stream));
            }
            checkCudaErrors(cudaStreamSynchronize(stream));
            auto end_sim = std::chrono::high_resolution_clock::now();

            checkCudaErrors(cudaEventDestroy(evt_start));
            checkCudaErrors(cudaEventDestroy(evt_stop));
            checkCudaErrors(cudaStreamDestroy(stream));
            checkCudaErrors(cudaFree(direct_gate_val_d));
            checkCudaErrors(cudaFree(direct_gate_indices_d));
            checkCudaErrors(cudaFree(direct_gate_row_order_d));

            final_state_idx = 1;
            final_state_idx_gpu = ((num_batch - 1) % 2) * 2 +
                (((num_batch - 1) / 2) * (stage2_gate_count + 1) + stage2_gate_count) % 2;
            const auto stage2_total_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(end_sim - begin_sim).count();
            std::cout << "[Stage 2: ELL-based batch simulation] time: "
                      << stage2_total_ms
                      << std::endl;
            if (enable_breakdown) {
              const double measured_total_ms =
                  std::chrono::duration<double, std::milli>(end_sim - begin_sim).count();
              const double other_overhead_ms =
                  std::max(0.0, measured_total_ms - total_pack_ms - total_sim_kernel_ms);
              std::cout << "  Breakdown:" << std::endl;
              std::cout << "  - Primitive -> ELL Pack:     " << total_pack_ms << " ms" << std::endl;
              std::cout << "  - ELL Batch Simulation:      " << total_sim_kernel_ms << " ms" << std::endl;
              std::cout << "  - Other Overhead:            " << other_overhead_ms << " ms" << std::endl;
            }
            return;
          }
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

                simulate_fused_gate.emplace_back(cudaflow.kernel(
                  grid_size,
                  block_size,
                  fusedGateSharedBytes(fused_num_nonzero[opNum]),
                  run_fused_gate,
                  fused_gates_val_d[opNum], fused_gates_indices_d[opNum], fused_num_nonzero[opNum],
                  fused_gates_row_order_d[opNum],
                  d_batch[input_buffer_idx],
                  d_batch[output_buffer_idx], batch_size, nDim
                ).name("fused_gate_" + std::to_string(opNum)));
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

          final_state_idx = 1;
          final_state_idx_gpu = ((num_batch - 1) % 2) * 2 +
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
    std::unique_ptr<CuSparseSpGEMMEngine> cuSparseEngine;
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

#endif //QBATCH_SIMULATOR_H
