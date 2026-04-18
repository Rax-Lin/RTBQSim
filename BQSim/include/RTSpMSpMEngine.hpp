#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include <cuComplex.h>

#include "GatePrimitive.hpp"
#include "RTNumericPrecision.hpp"

class RTSpMSpMEngine {
public:
  struct BuildGateEvent {
    std::size_t gate_idx = 0;
    qc::GatePrimitive gate{};
  };

  struct Stats {
    double dd_ms = 0.0;
    double geom_ms = 0.0;
    double gas_ms = 0.0;
    double launch_ms = 0.0;
    double ray_gen_ms = 0.0;
    double merge_ms = 0.0;
    double overhead_ms = 0.0;
    double ell_ms = 0.0;
    double h2d_ms = 0.0;
    double compute_ms = 0.0;
    double d2h_ms = 0.0;
    std::size_t bvh_rebuild_count = 0;
    std::size_t bvh_update_count = 0;
    std::size_t bvh_skip_count = 0;
    // Average primitive-position shift metric aggregation:
    // sample = mean_i( |delta_row_i| + |delta_col_i| ) / nDim, per GAS refresh event.
    // Rebuild contributes 0 with one sample; diagonal-only updates are excluded.
    double bvh_refit_shift_sum = 0.0;
    std::size_t bvh_refit_shift_samples = 0;
    std::vector<BuildGateEvent> build_gate_events{};
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

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
  bool  available = false;
  Stats last_stats{};
};
