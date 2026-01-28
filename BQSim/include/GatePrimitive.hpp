#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace qc {

static constexpr int MAX_TARGETS = 2;
static constexpr int MAX_CONTROLS = 4;

struct GatePrimitive {
  int gate_type = 0;
  int target_count = 0;
  int control_count = 0;
  int targets[MAX_TARGETS] = {0, 0};
  int controls[MAX_CONTROLS] = {0, 0, 0, 0};
  int matrix_dim = 0;
  float2 matrix[16]{};
  bool is_controlled = false;
};

}  // namespace qc
