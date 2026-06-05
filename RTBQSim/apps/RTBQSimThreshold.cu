#include "CuSparseSpGEMMEngine.hpp"
#include "CudaUtils.hpp"
#include "GatePrimitiveBuilder.hpp"
#include "GateFusionPlanner.hpp"
#include "RTSpMSpMEngine.hpp"
#include "QuantumComputation.hpp"
#include "cxxopts.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace {

struct ProbeResult {
  bool ok = false;
  double ms = std::numeric_limits<double>::infinity();
  std::size_t fused_gates = 0;
  std::size_t blocks = 0;
  std::string reason;
  double h2d_ms = 0.0;
  double ray_gen_ms = 0.0;
  double geom_ms = 0.0;
  double bvh_ms = 0.0;
  double launch_ms = 0.0;
  double compact_ms = 0.0;
  double diagonal_ms = 0.0;
  double overhead_ms = 0.0;
  double cleanup_ms = 0.0;
  double ell_ms = 0.0;
  std::size_t bvh_rebuild_count = 0;
  std::size_t bvh_update_count = 0;
  std::size_t bvh_skip_count = 0;
};

bool envFlag(const char* name) {
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

struct RowSortKey {
  int a;
  int b;
  int c;
  int d;

  __host__ __device__ bool operator<(const RowSortKey& other) const {
    if (a != other.a) return a < other.a;
    if (b != other.b) return b < other.b;
    if (c != other.c) return c < other.c;
    return d < other.d;
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
  const int base = row * 4;
  const int a = gates_indices[base + 0];
  const int b = gates_indices[base + 1];
  const int c = gates_indices[base + 2];
  const int d = gates_indices[base + 3];
  row_keys[row] = RowSortKey{a, b, c, d};
  row_order[row] = row;
}

template <class Engine>
ProbeResult measureStage1(Engine& engine,
                          const std::vector<qc::GatePrimitive>& primitives,
                          const std::vector<std::size_t>& planned_blocks,
                          int num_qubits,
                          std::size_t nDim) {
  ProbeResult result{};
  if (!engine.isAvailable()) {
    result.reason = "engine_unavailable";
    return result;
  }
  if (primitives.empty()) {
    result.reason = "no_primitives";
    return result;
  }

  auto begin = std::chrono::high_resolution_clock::now();
  std::size_t cursor = 0;
  std::size_t block_id = 0;
  const int ell_width_limit = static_cast<int>(envUInt64("BQSIM_RT_SPM_ROW_NNZ_LIMIT", 4ULL));
  std::deque<std::size_t> pending_blocks(planned_blocks.begin(), planned_blocks.end());
  bqsim_rt::Complex* fused_gate_val = nullptr;
  int* fused_gate_indices = nullptr;
  int* fused_gate_row_order = nullptr;
  RowSortKey* fused_gate_row_keys = nullptr;
  std::size_t ell_capacity = 0;
  std::size_t row_order_capacity = 0;
  auto cleanup_probe_buffers = [&]() {
    if (fused_gate_row_keys) {
      cudaFree(fused_gate_row_keys);
      fused_gate_row_keys = nullptr;
    }
    if (fused_gate_row_order) {
      cudaFree(fused_gate_row_order);
      fused_gate_row_order = nullptr;
    }
    if (fused_gate_val) {
      cudaFree(fused_gate_val);
      fused_gate_val = nullptr;
    }
    if (fused_gate_indices) {
      cudaFree(fused_gate_indices);
      fused_gate_indices = nullptr;
    }
    ell_capacity = 0;
    row_order_capacity = 0;
  };
  while (cursor < primitives.size() && !pending_blocks.empty()) {
    const std::size_t remaining = primitives.size() - cursor;
    const std::size_t planned = std::min(pending_blocks.front(), remaining);
    if (planned == 0) {
      pending_blocks.pop_front();
      break;
    }
    engine.resetStats();
    if (!(engine.prepareGeometryFromGates(primitives.data() + cursor,
                                          planned,
                                          num_qubits,
                                          nDim,
                                          true) &&
          engine.launchRTMultiply())) {
      result.reason = "prepare_or_launch_failed";
      cleanup_probe_buffers();
      return result;
    }
    int ell_width = engine.maxRowNNZ();
    if (ell_width <= 0) {
      ell_width = 1;
    }
    if (ell_width_limit > 0 && ell_width > ell_width_limit) {
      if (planned <= 1) {
        cleanup_probe_buffers();
        result.reason = "single_gate_exceeded_ell_width_limit";
        return result;
      }
      engine.resetStats();
      if (!(engine.prepareGeometryFromGates(primitives.data() + cursor,
                                            planned,
                                            num_qubits,
                                            nDim,
                                            false) &&
            engine.launchRTMultiply())) {
        cleanup_probe_buffers();
        result.reason = "runtime_fallback_failed";
        return result;
      }
      const std::size_t fallback_fused = engine.lastFusedGateCount();
      if (fallback_fused == 0 || fallback_fused > planned) {
        cleanup_probe_buffers();
        result.reason = "invalid_runtime_fallback_fused_count";
        return result;
      }
      const std::size_t remainder = planned - fallback_fused;
      std::cout << "[threshold]   DAG block rejected after backend fusion at block "
                << (block_id + 1)
                << " (ELL width " << ell_width
                << " > " << ell_width_limit
                << "); falling back to runtime row_nnz-limited fusion and accepting "
                << fallback_fused << " gate(s)";
      if (remainder > 0) {
        std::cout << ", leaving " << remainder << " gate(s) for the next block";
      }
      std::cout << std::endl;
      ell_width = engine.maxRowNNZ();
      if (ell_width <= 0) {
        ell_width = 1;
      }
      if (ell_width_limit > 0 && ell_width > ell_width_limit) {
        cleanup_probe_buffers();
        result.reason = "runtime_fallback_still_exceeded_ell_width_limit";
        return result;
      }
    }
    const std::size_t fused = engine.lastFusedGateCount();
    if (fused == 0 || fused > planned) {
      result.reason = "invalid_fused_gate_count";
      cleanup_probe_buffers();
      return result;
    }
    const auto& stats = engine.lastStats();
    result.h2d_ms += stats.h2d_ms;
    result.ray_gen_ms += stats.ray_gen_ms;
    result.geom_ms += stats.geom_ms;
    result.bvh_ms += stats.gas_ms;
    result.launch_ms += stats.launch_ms;
    result.compact_ms += stats.compact_ms;
    result.diagonal_ms += stats.diagonal_ms;
    result.overhead_ms += stats.overhead_ms;
    result.cleanup_ms += stats.cleanup_ms;
    result.bvh_rebuild_count += stats.bvh_rebuild_count;
    result.bvh_update_count += stats.bvh_update_count;
    result.bvh_skip_count += stats.bvh_skip_count;
    auto ell_start = std::chrono::high_resolution_clock::now();
    const bool use_row_reorder =
        !std::getenv("BQSIM_RT_ROW_REORDER") || envFlag("BQSIM_RT_ROW_REORDER");
    const std::size_t required_ell_capacity =
        static_cast<std::size_t>(ell_width) * nDim;

    if (required_ell_capacity > ell_capacity) {
      if (fused_gate_val) {
        cudaFree(fused_gate_val);
        fused_gate_val = nullptr;
      }
      if (fused_gate_indices) {
        cudaFree(fused_gate_indices);
        fused_gate_indices = nullptr;
      }
      if (cudaMalloc(reinterpret_cast<void**>(&fused_gate_val),
                     required_ell_capacity * sizeof(bqsim_rt::Complex)) != cudaSuccess ||
          cudaMalloc(reinterpret_cast<void**>(&fused_gate_indices),
                     required_ell_capacity * sizeof(int)) != cudaSuccess) {
        cleanup_probe_buffers();
        result.reason = "ell_allocation_failed";
        return result;
      }
      ell_capacity = required_ell_capacity;
    }
    checkCudaErrors(cudaMemset(fused_gate_val,
                               0,
                               required_ell_capacity * sizeof(bqsim_rt::Complex)));
    checkCudaErrors(cudaMemset(fused_gate_indices,
                               0,
                               required_ell_capacity * sizeof(int)));

    if (!engine.collectResultToELL(fused_gate_val, fused_gate_indices, ell_width, nDim)) {
      cleanup_probe_buffers();
      result.reason = "collect_result_to_ell_failed";
      return result;
    }

    if (use_row_reorder && ell_width == 4) {
      if (nDim > row_order_capacity) {
        if (fused_gate_row_order) {
          cudaFree(fused_gate_row_order);
          fused_gate_row_order = nullptr;
        }
        if (fused_gate_row_keys) {
          cudaFree(fused_gate_row_keys);
          fused_gate_row_keys = nullptr;
        }
        if (cudaMalloc(reinterpret_cast<void**>(&fused_gate_row_order),
                       nDim * sizeof(int)) == cudaSuccess &&
            cudaMalloc(reinterpret_cast<void**>(&fused_gate_row_keys),
                       nDim * sizeof(RowSortKey)) == cudaSuccess) {
          row_order_capacity = nDim;
        } else {
          if (fused_gate_row_keys) {
            cudaFree(fused_gate_row_keys);
            fused_gate_row_keys = nullptr;
          }
          if (fused_gate_row_order) {
            cudaFree(fused_gate_row_order);
            fused_gate_row_order = nullptr;
          }
          row_order_capacity = 0;
        }
      }
      if (fused_gate_row_order && fused_gate_row_keys) {
        constexpr int kThreadsPerBlock = 256;
        const int blocks = static_cast<int>((nDim + kThreadsPerBlock - 1) / kThreadsPerBlock);
        build_row_order_keys_w4<<<blocks, kThreadsPerBlock>>>(
            fused_gate_indices, fused_gate_row_keys, fused_gate_row_order, static_cast<int>(nDim));
        checkCudaErrors(cudaGetLastError());
        thrust::stable_sort_by_key(
            thrust::device,
            thrust::device_pointer_cast(fused_gate_row_keys),
            thrust::device_pointer_cast(fused_gate_row_keys + nDim),
            thrust::device_pointer_cast(fused_gate_row_order));
      }
    }
    auto ell_stop = std::chrono::high_resolution_clock::now();
    result.ell_ms += std::chrono::duration<double, std::milli>(ell_stop - ell_start).count();
    result.fused_gates += fused;
    ++result.blocks;
    cursor += fused;
    pending_blocks.pop_front();
    if (fused < planned) {
      pending_blocks.push_front(planned - fused);
    }
    ++block_id;
  }
  if (cursor != primitives.size() || !pending_blocks.empty()) {
    result.reason = "dag_plan_incomplete";
    cleanup_probe_buffers();
    return result;
  }
  auto end = std::chrono::high_resolution_clock::now();
  cleanup_probe_buffers();
  result.ok = true;
  result.ms = std::chrono::duration<double, std::milli>(end - begin).count();
  return result;
}

std::string thresholdCircuitPath(int qubits) {
  return std::string("../../circuits/threshold/threshold_n") + std::to_string(qubits) + ".qasm";
}

bool runIsolatedChild(const std::string& exe_path,
                      int qubits,
                      std::string& child_output,
                      int& child_threshold,
                      std::string& child_stop_reason) {
  child_output.clear();
  child_threshold = qubits - 1;
  child_stop_reason.clear();

  std::ostringstream cmd;
  cmd << '"' << exe_path << '"'
      << " --min-qubits " << qubits
      << " --max-qubits " << qubits
      << " --isolated-child";

  FILE* pipe = popen(cmd.str().c_str(), "r");
  if (!pipe) {
    child_stop_reason = "popen_failed";
    return false;
  }

  char buffer[4096];
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    child_output += buffer;
  }
  const int rc = pclose(pipe);
  if (rc != 0) {
    child_stop_reason = "child_process_failed";
    return false;
  }

  std::istringstream iss(child_output);
  std::string line;
  std::ostringstream filtered;
  bool saw_threshold = false;
  while (std::getline(iss, line)) {
    if (line.rfind("THRESHOLD=", 0) == 0) {
      child_threshold = std::stoi(line.substr(std::strlen("THRESHOLD=")));
      saw_threshold = true;
    } else if (line.rfind("[threshold] stop_reason=", 0) == 0) {
      child_stop_reason = line.substr(std::strlen("[threshold] stop_reason="));
    } else {
      filtered << line << '\n';
    }
  }
  child_output = filtered.str();
  return saw_threshold;
}

void printRtBreakdown(int qubits, const ProbeResult& result, const char* primitive_name) {
  std::cout << "[threshold] q=" << qubits << " backend=rt total_ms="
            << (result.ok ? std::to_string(result.ms) : std::string("FAIL"))
            << " blocks=" << result.blocks
            << " fused_gates=" << result.fused_gates
            << std::endl;
  if (!result.ok) {
    std::cout << "[threshold]   reason=" << result.reason << std::endl;
    return;
  }
  std::cout << "[threshold]   Breakdown:" << std::endl;
  std::cout << "[threshold]   - Ray Generation:            " << result.ray_gen_ms << " ms" << std::endl;
  if (result.geom_ms > 0.0) {
    const std::string geom_label =
        std::string("COO -> ") +
        ((primitive_name && std::strcmp(primitive_name, "sphere") == 0) ? "Sphere" : "Triangle");
    std::cout << "[threshold]   - " << geom_label;
    if (geom_label.size() < 25) {
      std::cout << std::string(25 - geom_label.size(), ' ');
    }
    std::cout << result.geom_ms << " ms" << std::endl;
  }
  std::cout << "[threshold]   - BVH Build (OptiX):         " << result.bvh_ms << " ms" << std::endl;
  std::cout << "[threshold]   - bvh build update time :    " << result.bvh_update_count << " times" << std::endl;
  std::cout << "[threshold]   - bvh build rebuild time :   " << result.bvh_rebuild_count << " times" << std::endl;
  std::cout << "[threshold]   - bvh build skip time :      " << result.bvh_skip_count << " times" << std::endl;
  std::cout << "[threshold]   - Ray Tracing (Launch):      "
            << (result.launch_ms + result.compact_ms) << " ms" << std::endl;
  std::cout << "[threshold]   - NNZ1 Multiplication:       " << result.diagonal_ms << " ms" << std::endl;
  std::cout << "[threshold]   - Memory & Overhead:         "
            << (result.overhead_ms + result.h2d_ms + result.cleanup_ms) << " ms" << std::endl;
  std::cout << "[threshold]   - ELL Conversion (Result):   " << result.ell_ms << " ms" << std::endl;
}

void printCuSparseBreakdown(int qubits, const ProbeResult& result) {
  std::cout << "[threshold] q=" << qubits << " backend=cusparse total_ms="
            << (result.ok ? std::to_string(result.ms) : std::string("FAIL"))
            << " blocks=" << result.blocks
            << " fused_gates=" << result.fused_gates
            << std::endl;
  if (!result.ok) {
    std::cout << "[threshold]   reason=" << result.reason << std::endl;
    return;
  }
  std::cout << "[threshold]   Breakdown:" << std::endl;
  std::cout << "[threshold]   - Gate->CSR Build (GPU):     " << result.ray_gen_ms << " ms" << std::endl;
  std::cout << "[threshold]   - SpGEMM Compute (cuSPARSE): " << result.launch_ms << " ms" << std::endl;
  std::cout << "[threshold]   - RowNNZ Scan (D2H+CPU):     " << result.compact_ms << " ms" << std::endl;
  std::cout << "[threshold]   - NNZ1 Multiplication:       " << result.diagonal_ms << " ms" << std::endl;
  std::cout << "[threshold]   - skip time :                " << result.bvh_skip_count << " times" << std::endl;
  std::cout << "[threshold]   - Other Overhead:            "
            << (result.overhead_ms + result.cleanup_ms) << " ms" << std::endl;
  std::cout << "[threshold]   - ELL Conversion (Result):   " << result.ell_ms << " ms" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  setenv("BQSIM_ENABLE_BREAKDOWN", "1", 1);
  cxxopts::Options options("RTBQSimThreshold", "Probe RTSpMSpM vs cuSPARSE gate-fusion threshold");
  options.add_options()
    ("h,help", "show help")
    ("min-qubits", "minimum qubit size", cxxopts::value<int>()->default_value("16"))
    ("max-qubits", "maximum qubit size", cxxopts::value<int>()->default_value("23"))
    ("isolated-child", "internal flag: run a single isolated q probe in this process")
    ("verbose", "print verbose probe information");

  const auto vm = options.parse(argc, argv);
  if (vm.count("help") > 0) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  const int min_qubits = vm["min-qubits"].as<int>();
  const int max_qubits = vm["max-qubits"].as<int>();
  const bool isolated_child = vm.count("isolated-child") > 0;
  const bool verbose = vm.count("verbose") > 0;

  if (!isolated_child && min_qubits != max_qubits) {
    int threshold = min_qubits - 1;
    bool found_switch = false;
    std::string stop_reason;
    for (int qubits = min_qubits; qubits <= max_qubits; ++qubits) {
      std::string child_output;
      int child_threshold = qubits - 1;
      std::string child_stop_reason;
      if (!runIsolatedChild(argv[0], qubits, child_output, child_threshold, child_stop_reason)) {
        std::cout << child_output;
        stop_reason = child_stop_reason.empty() ? "isolated_child_failed" : child_stop_reason;
        threshold = qubits - 1;
        found_switch = true;
        break;
      }
      std::cout << child_output;
      if (child_threshold < qubits) {
        threshold = child_threshold;
        stop_reason = child_stop_reason.empty() ? "cusparse_total_faster" : child_stop_reason;
        found_switch = true;
        break;
      }
      threshold = qubits;
    }
    if (!found_switch && stop_reason.empty()) {
      stop_reason = "rt_kept_advantage_through_max";
    }
    std::cout << "[threshold] stop_reason=" << stop_reason << std::endl;
    std::cout << "THRESHOLD=" << threshold << std::endl;
    return 0;
  }

  int threshold = min_qubits - 1;
  bool found_switch = false;
  std::string stop_reason;

  for (int qubits = min_qubits; qubits <= max_qubits; ++qubits) {
    const std::string circuit_path = thresholdCircuitPath(qubits);
    if (!std::filesystem::exists(circuit_path)) {
      stop_reason = "missing_circuit:" + circuit_path;
      break;
    }

    auto qc = std::make_unique<qc::QuantumComputation>(circuit_path);
    std::vector<qc::GatePrimitive> primitives;
    if (!bqsim_rt::buildGatePrimitives(*qc, primitives)) {
      stop_reason = "build_gate_primitives_failed";
      break;
    }
    const int row_nnz_limit = static_cast<int>(envUInt64("BQSIM_RT_SPM_ROW_NNZ_LIMIT", 4ULL));
    bqsim_rt::GateFusionPlan fusion_plan;
    if (!bqsim_rt::buildGateFusionPlan(primitives, row_nnz_limit, fusion_plan)) {
      stop_reason = "dag_fusion_planning_failed";
      break;
    }
    primitives = std::move(fusion_plan.ordered_primitives);
    std::vector<std::size_t> planned_blocks = std::move(fusion_plan.block_sizes);

    std::size_t nDim = static_cast<std::size_t>(1ULL) << static_cast<unsigned long long>(qubits);
    RTSpMSpMEngine rt_engine;
    rt_engine.setAvailable(true);
    rt_engine.warmup();

    CuSparseSpGEMMEngine cusparse_engine;
    cusparse_engine.setAvailable(true);
    cusparse_engine.warmup();

    ProbeResult rt_res = measureStage1(rt_engine, primitives, planned_blocks, qubits, nDim);
    ProbeResult cs_res = measureStage1(cusparse_engine, primitives, planned_blocks, qubits, nDim);

    if (verbose || true) {
      printRtBreakdown(qubits, rt_res, rt_engine.primitiveTypeName());
      printCuSparseBreakdown(qubits, cs_res);
    }

    if (!rt_res.ok || !cs_res.ok) {
      threshold = qubits - 1;
      stop_reason = !rt_res.ok ? ("rt_failed:" + rt_res.reason) : ("cusparse_failed:" + cs_res.reason);
      found_switch = true;
      break;
    }

    threshold = qubits;
    if (cs_res.ms < rt_res.ms) {
      threshold = qubits - 1;
      found_switch = true;
      stop_reason = "cusparse_total_faster";
      break;
    }
  }

  if (!found_switch && stop_reason.empty()) {
    stop_reason = "rt_kept_advantage_through_max";
  }

  std::cout << "[threshold] stop_reason=" << stop_reason << std::endl;
  std::cout << "THRESHOLD=" << threshold << std::endl;
  return 0;
}
