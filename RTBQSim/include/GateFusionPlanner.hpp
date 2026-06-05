#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <unordered_map>
#include <vector>

#include "GatePrimitive.hpp"

namespace bqsim_rt {

struct GateFusionPlan {
  std::vector<qc::GatePrimitive> ordered_primitives{};
  std::vector<std::size_t> block_sizes{};
};

inline bool plannerIsZeroMatrixEntry(const bqsim_rt::MatrixElem& value) {
  return value.x == 0.0 && value.y == 0.0;
}

inline int plannerGateRowNNZUpperBound(const qc::GatePrimitive& gate) {
  if (gate.target_count <= 0 || gate.matrix_dim <= 0) {
    return 1;
  }
  int max_row = 0;
  for (int r = 0; r < gate.matrix_dim; ++r) {
    int nnz = 0;
    for (int c = 0; c < gate.matrix_dim; ++c) {
      if (!plannerIsZeroMatrixEntry(gate.matrix[r * gate.matrix_dim + c])) {
        ++nnz;
      }
    }
    max_row = std::max(max_row, nnz);
  }
  return std::max(1, max_row);
}

inline bool plannerIsDiagonalGate(const qc::GatePrimitive& gate) {
  const int dim = gate.matrix_dim;
  if (dim <= 0 || dim > 4) {
    return false;
  }
  for (int r = 0; r < dim; ++r) {
    for (int c = 0; c < dim; ++c) {
      if (r != c && !plannerIsZeroMatrixEntry(gate.matrix[r * dim + c])) {
        return false;
      }
    }
  }
  return true;
}

inline bool plannerGateHasOneRayPerRow(const qc::GatePrimitive& gate) {
  if (gate.target_count != 1 || gate.matrix_dim != 2) {
    return false;
  }
  for (int r = 0; r < 2; ++r) {
    int row_nnz = 0;
    for (int c = 0; c < 2; ++c) {
      if (!plannerIsZeroMatrixEntry(gate.matrix[r * 2 + c])) {
        ++row_nnz;
      }
    }
    if (row_nnz != 1) {
      return false;
    }
  }
  return true;
}

inline bool plannerGateIsWidthPreserving(const qc::GatePrimitive& gate) {
  return plannerIsDiagonalGate(gate) || plannerGateHasOneRayPerRow(gate);
}

inline std::vector<int> plannerTouchedQubits(const qc::GatePrimitive& gate) {
  std::vector<int> qubits;
  qubits.reserve(static_cast<std::size_t>(gate.control_count + gate.target_count));
  for (int i = 0; i < gate.control_count; ++i) {
    qubits.push_back(gate.controls[i]);
  }
  for (int i = 0; i < gate.target_count; ++i) {
    qubits.push_back(gate.targets[i]);
  }
  std::sort(qubits.begin(), qubits.end());
  qubits.erase(std::unique(qubits.begin(), qubits.end()), qubits.end());
  return qubits;
}

inline bool plannerIsSubsetOf(const std::vector<int>& needle,
                              const std::vector<int>& haystack) {
  return std::includes(haystack.begin(), haystack.end(),
                       needle.begin(), needle.end());
}

inline std::vector<int> plannerUnionQubits(const std::vector<int>& lhs,
                                           const std::vector<int>& rhs) {
  std::vector<int> merged;
  merged.reserve(lhs.size() + rhs.size());
  std::set_union(lhs.begin(), lhs.end(),
                 rhs.begin(), rhs.end(),
                 std::back_inserter(merged));
  return merged;
}

inline int plannerMaxGroupQubitsFromRowNNZLimit(int row_nnz_limit) {
  if (row_nnz_limit <= 0) {
    return std::numeric_limits<int>::max();
  }
  int max_qubits = 0;
  int capacity = 1;
  while (capacity < row_nnz_limit) {
    capacity <<= 1;
    ++max_qubits;
  }
  if (capacity > row_nnz_limit && max_qubits > 0) {
    --max_qubits;
  }
  return std::max(0, max_qubits);
}

inline bool buildGateFusionPlan(const std::vector<qc::GatePrimitive>& primitives,
                                int row_nnz_limit,
                                GateFusionPlan& out) {
  out.ordered_primitives.clear();
  out.block_sizes.clear();
  if (primitives.empty()) {
    return true;
  }
  const int max_group_qubits = plannerMaxGroupQubitsFromRowNNZLimit(row_nnz_limit);

  const std::size_t n = primitives.size();
  std::vector<std::vector<int>> gate_qubits(n);
  std::vector<std::vector<std::size_t>> predecessors(n);
  std::unordered_map<int, std::size_t> last_touch;
  last_touch.reserve(n * 2);

  for (std::size_t idx = 0; idx < n; ++idx) {
    gate_qubits[idx] = plannerTouchedQubits(primitives[idx]);
    const auto& qubits = gate_qubits[idx];
    std::vector<std::size_t> preds;
    preds.reserve(qubits.size());
    for (int q : qubits) {
      const auto it = last_touch.find(q);
      if (it != last_touch.end()) {
        const std::size_t pred = it->second;
        if (std::find(preds.begin(), preds.end(), pred) == preds.end()) {
          preds.push_back(pred);
        }
      }
    }
    predecessors[idx] = std::move(preds);
    for (int q : qubits) {
      last_touch[q] = idx;
    }
  }

  std::vector<char> globally_scheduled(n, 0);
  std::vector<char> in_current_block(n, 0);
  std::size_t scheduled_count = 0;

  out.ordered_primitives.reserve(n);
  out.block_sizes.reserve(n);

  auto is_ready_with_current_block = [&](std::size_t idx) {
    if (globally_scheduled[idx] || in_current_block[idx]) {
      return false;
    }
    for (std::size_t pred : predecessors[idx]) {
      if (!globally_scheduled[pred] && !in_current_block[pred]) {
        return false;
      }
    }
    return true;
  };

  while (scheduled_count < n) {
    std::fill(in_current_block.begin(), in_current_block.end(), 0);

    std::size_t seed = n;
    for (std::size_t idx = 0; idx < n; ++idx) {
      if (is_ready_with_current_block(idx)) {
        seed = idx;
        break;
      }
    }
    if (seed == n) {
      return false;
    }

    std::size_t block_size = 0;
    std::vector<int> dense_mask;

    auto add_gate = [&](std::size_t idx) {
      in_current_block[idx] = 1;
      ++block_size;
      out.ordered_primitives.push_back(primitives[idx]);
      if (!plannerGateIsWidthPreserving(primitives[idx])) {
        dense_mask = plannerUnionQubits(dense_mask, gate_qubits[idx]);
      }
    };

    add_gate(seed);

    while (true) {
      std::size_t same_mask_choice = n;
      std::size_t width_preserving_choice = n;
      std::size_t expand_two_qubit_choice = n;
      std::size_t expand_one_qubit_choice = n;

      for (std::size_t idx = seed + 1; idx < n; ++idx) {
        if (!is_ready_with_current_block(idx)) {
          continue;
        }

        const bool width_preserving = plannerGateIsWidthPreserving(primitives[idx]);

        if (!dense_mask.empty() && !width_preserving &&
            plannerIsSubsetOf(gate_qubits[idx], dense_mask)) {
          same_mask_choice = idx;
          break;
        }

        if (width_preserving) {
          if (width_preserving_choice == n) {
            width_preserving_choice = idx;
          }
          continue;
        }

        const auto expanded_mask = plannerUnionQubits(dense_mask, gate_qubits[idx]);
        if (static_cast<int>(expanded_mask.size()) > max_group_qubits) {
          continue;
        }
        if (static_cast<int>(dense_mask.size()) < max_group_qubits) {
          if (gate_qubits[idx].size() == 2) {
            expand_two_qubit_choice = idx;
            break;
          }
          if (static_cast<int>(expanded_mask.size()) <= max_group_qubits &&
              expand_one_qubit_choice == n) {
            expand_one_qubit_choice = idx;
          }
        }
      }

      std::size_t chosen = same_mask_choice;
      if (chosen == n) {
        if (width_preserving_choice != n) {
          chosen = width_preserving_choice;
        } else {
          chosen = (expand_two_qubit_choice != n) ? expand_two_qubit_choice : expand_one_qubit_choice;
        }
      }
      if (chosen == n) {
        break;
      }

      add_gate(chosen);
    }

    if (block_size == 0) {
      return false;
    }

    for (std::size_t idx = 0; idx < n; ++idx) {
      if (in_current_block[idx]) {
        globally_scheduled[idx] = 1;
        ++scheduled_count;
      }
    }
    out.block_sizes.push_back(block_size);
  }

  return out.ordered_primitives.size() == primitives.size();
}

}  // namespace bqsim_rt
