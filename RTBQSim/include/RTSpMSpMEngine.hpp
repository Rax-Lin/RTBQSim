#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <cuComplex.h>

#include "GatePrimitive.hpp"
#include "RTNumericPrecision.hpp"

class RTSpMSpMEngine {
public:
  struct BuildGateEvent {
    std::size_t gate_idx = 0;
    std::size_t traversal_begin_sample_idx = 0;
    qc::GatePrimitive gate{};
    int tree_build_row_nnz = 0;
    int tree_final_row_nnz = 0;
    double traversal_average_ms = 0.0;
    std::size_t traversal_sample_count = 0;
  };

  struct GateTraversalEvent {
    std::size_t gate_idx = 0;
    std::size_t traversal_sample_idx = static_cast<std::size_t>(-1);
    qc::GatePrimitive gate{};
    int tree_row_nnz_before = 0;
    int result_row_nnz_after = 0;
    double traversal_ms = 0.0;
    bool has_traversal = false;
  };

  struct Stats {
    double dd_ms = 0.0;
    double geom_ms = 0.0;
    double gas_ms = 0.0;
    double launch_ms = 0.0;
    double ray_gen_ms = 0.0;
    double diagonal_ms = 0.0;
    double compact_ms = 0.0;
    double overhead_ms = 0.0;
    double ell_ms = 0.0;
    double h2d_ms = 0.0;
    double compute_ms = 0.0;
    double d2h_ms = 0.0;
    std::size_t bvh_rebuild_count = 0;
    std::size_t bvh_update_count = 0;
    std::size_t bvh_skip_count = 0;
    std::vector<BuildGateEvent> build_gate_events{};
    std::vector<GateTraversalEvent> gate_traversal_events{};
  };

  RTSpMSpMEngine();
  ~RTSpMSpMEngine();

  bool isAvailable() const;
  void setAvailable(bool value);

  bool prepareGeometryFromGates(const qc::GatePrimitive* gates,
                                std::size_t gate_count,
                                int num_qubits,
                                std::size_t nDim,
                                bool force_full = false);
  bool launchRTMultiply();
  bool collectResultToELL(bqsim_rt::Complex* values,
                          int* indices,
                          int num_non_zeros,
                          std::size_t nDim);
  int maxRowNNZ() const;
  std::size_t lastFusedGateCount() const;

  const Stats& lastStats() const;
  void resetStats();
  void warmup();
  void setDebugContext(const std::string& circuit_name, std::size_t block_start_gate);

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
  bool  available = false;
  Stats last_stats{};
};
