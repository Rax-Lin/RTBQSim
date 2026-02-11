#pragma once

#include <cstddef>
#include <memory>

#include <cuComplex.h>

#include "GatePrimitive.hpp"

namespace qc {
class FusedGate;
}

namespace dd {
struct DDPackageConfig;
template <class Config>
class Package;
}

class RTSpMSpMEngine {
public:
  struct Stats {
    double dd_ms = 0.0;
    double gas_ms = 0.0;
    double launch_ms = 0.0;
    double merge_ms = 0.0;
    double overhead_ms = 0.0;
    double ell_ms = 0.0;
    double h2d_ms = 0.0;
    double compute_ms = 0.0;
    double d2h_ms = 0.0;
  };

  RTSpMSpMEngine();
  ~RTSpMSpMEngine();

  bool isAvailable() const;
  void setAvailable(bool value);

  // Return false to indicate fallback to the original DD-to-ELL path.
  bool prepareGeometry(const qc::FusedGate& gate,
                       dd::Package<dd::DDPackageConfig>* dd,
                       int num_qubits,
                       std::size_t nDim);
  bool prepareGeometryFromGates(const qc::GatePrimitive* gates,
                                std::size_t gate_count,
                                int num_qubits,
                                std::size_t nDim,
                                bool force_full = false);
  bool launchRTMultiply();
  bool collectResultToELL(cuDoubleComplex* values,
                          int* indices,
                          int num_non_zeros,
                          std::size_t nDim);
  double densityEstimate() const;
  int maxRowNNZ() const;
  int ellWidthHint(int fallback) const;
  bool useDenseMV() const;
  std::size_t lastFusedGateCount() const;
  bool lastReachedDensity() const;

  const Stats& lastStats() const;
  void resetStats();
  void warmup();

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
  bool  available = false;
  Stats last_stats{};
};
