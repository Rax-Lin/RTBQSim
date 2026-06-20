#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "GatePrimitive.hpp"
#include "operations/OpType.hpp"

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

inline bool plannerGateIsBridgeControlledX(const qc::GatePrimitive& gate) {
  return gate.control_count > 0 &&
         gate.target_count == 1 &&
         static_cast<qc::OpType>(gate.gate_type) == qc::X &&
         plannerGateHasOneRayPerRow(gate);
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

inline bool plannerIntersectsQubits(const std::vector<int>& lhs,
                                    const std::vector<int>& rhs) {
  std::size_t i = 0;
  std::size_t j = 0;
  while (i < lhs.size() && j < rhs.size()) {
    if (lhs[i] == rhs[j]) {
      return true;
    }
    if (lhs[i] < rhs[j]) {
      ++i;
    } else {
      ++j;
    }
  }
  return false;
}

struct PlannerBlockState {
  std::vector<int> active_support{};
  std::vector<std::vector<int>> bridge_components{};
  std::set<std::pair<int, int>> bridge_dirs{};
};

struct PlannerBeamState {
  PlannerBlockState block_state{};
  std::vector<std::size_t> chosen{};
  std::vector<char> in_block{};
  std::size_t first_gate = std::numeric_limits<std::size_t>::max();
  std::size_t last_gate = 0;
};

inline std::size_t plannerBlockSpan(const PlannerBeamState& state) {
  if (state.chosen.empty()) {
    return 0;
  }
  return state.last_gate - state.first_gate + 1;
}

inline bool plannerBeamStateBetter(const PlannerBeamState& lhs,
                                   const PlannerBeamState& rhs) {
  if (lhs.chosen.size() != rhs.chosen.size()) {
    return lhs.chosen.size() > rhs.chosen.size();
  }
  if (lhs.block_state.active_support.size() != rhs.block_state.active_support.size()) {
    return lhs.block_state.active_support.size() > rhs.block_state.active_support.size();
  }
  if (lhs.block_state.bridge_components.size() != rhs.block_state.bridge_components.size()) {
    return lhs.block_state.bridge_components.size() < rhs.block_state.bridge_components.size();
  }
  const std::size_t lhs_span = plannerBlockSpan(lhs);
  const std::size_t rhs_span = plannerBlockSpan(rhs);
  if (lhs_span != rhs_span) {
    return lhs_span < rhs_span;
  }
  if (lhs.last_gate != rhs.last_gate) {
    return lhs.last_gate < rhs.last_gate;
  }
  return lhs.chosen < rhs.chosen;
}

inline bool plannerIsReadyForBlock(std::size_t idx,
                                   const std::vector<std::vector<std::size_t>>& predecessors,
                                   const std::vector<char>& globally_scheduled,
                                   const std::vector<char>& in_block) {
  if (globally_scheduled[idx] || in_block[idx]) {
    return false;
  }
  for (std::size_t pred : predecessors[idx]) {
    if (!globally_scheduled[pred] && !in_block[pred]) {
      return false;
    }
  }
  return true;
}

inline void plannerCollectReadyCandidates(
    const std::vector<std::vector<std::size_t>>& predecessors,
    const std::vector<char>& globally_scheduled,
    const std::vector<char>& in_block,
    std::size_t ready_window,
    std::vector<std::size_t>& out_candidates) {
  out_candidates.clear();
  for (std::size_t idx = 0; idx < predecessors.size(); ++idx) {
    if (!plannerIsReadyForBlock(idx, predecessors, globally_scheduled, in_block)) {
      continue;
    }
    out_candidates.push_back(idx);
    if (out_candidates.size() >= ready_window) {
      break;
    }
  }
}

inline bool plannerApplyGateToBlockState(const qc::GatePrimitive& gate,
                                         const std::vector<int>& gate_qubits,
                                         int max_group_qubits,
                                         PlannerBlockState& state) {
  if (plannerIsDiagonalGate(gate)) {
    return true;
  }

  if (plannerGateIsBridgeControlledX(gate)) {
    std::vector<int> merged_component = gate_qubits;
    std::vector<std::vector<int>> kept_components;
    kept_components.reserve(state.bridge_components.size());
    for (const auto& component : state.bridge_components) {
      if (plannerIntersectsQubits(component, gate_qubits)) {
        merged_component = plannerUnionQubits(merged_component, component);
      } else {
        kept_components.push_back(component);
      }
    }

    bool activate_component = false;
    const int target = gate.targets[0];
    for (int i = 0; i < gate.control_count; ++i) {
      const int control = gate.controls[i];
      if (state.bridge_dirs.count(std::make_pair(target, control)) != 0U) {
        activate_component = true;
      }
    }
    for (int i = 0; i < gate.control_count; ++i) {
      state.bridge_dirs.insert(std::make_pair(gate.controls[i], target));
    }

    if (activate_component) {
      const auto expanded_support =
          plannerUnionQubits(state.active_support, merged_component);
      if (static_cast<int>(expanded_support.size()) > max_group_qubits) {
        return false;
      }
      state.active_support = expanded_support;
    }

    kept_components.push_back(std::move(merged_component));
    state.bridge_components = std::move(kept_components);
    return true;
  }

  std::vector<int> expanded_support = plannerUnionQubits(state.active_support, gate_qubits);
  for (const auto& component : state.bridge_components) {
    if (plannerIntersectsQubits(component, gate_qubits)) {
      expanded_support = plannerUnionQubits(expanded_support, component);
    }
  }
  if (static_cast<int>(expanded_support.size()) > max_group_qubits) {
    return false;
  }
  state.active_support = std::move(expanded_support);
  return true;
}

inline bool plannerAppendGateRangeToState(const std::vector<qc::GatePrimitive>& gates,
                                          std::size_t begin,
                                          std::size_t count,
                                          int max_group_qubits,
                                          PlannerBlockState& state) {
  const std::size_t end = begin + count;
  for (std::size_t idx = begin; idx < end; ++idx) {
    const auto gate_qubits = plannerTouchedQubits(gates[idx]);
    if (!plannerApplyGateToBlockState(gates[idx],
                                      gate_qubits,
                                      max_group_qubits,
                                      state)) {
      return false;
    }
  }
  return true;
}

inline bool plannerChooseNextBlock(const std::vector<qc::GatePrimitive>& primitives,
                                   const std::vector<std::vector<int>>& gate_qubits,
                                   const std::vector<std::vector<std::size_t>>& predecessors,
                                   const std::vector<char>& globally_scheduled,
                                   int max_group_qubits,
                                   std::vector<std::size_t>& chosen_block) {
  constexpr std::size_t kBeamWidth = 24;
  constexpr std::size_t kReadyWindow = 32;

  chosen_block.clear();

  PlannerBeamState initial;
  initial.in_block.assign(primitives.size(), 0);

  std::vector<PlannerBeamState> beam;
  beam.push_back(initial);

  PlannerBeamState best_state;
  bool have_best_state = false;
  std::vector<std::size_t> ready_candidates;

  while (!beam.empty()) {
    std::vector<PlannerBeamState> next_beam;
    for (const PlannerBeamState& state : beam) {
      plannerCollectReadyCandidates(predecessors,
                                    globally_scheduled,
                                    state.in_block,
                                    kReadyWindow,
                                    ready_candidates);

      bool expanded = false;
      for (std::size_t idx : ready_candidates) {
        PlannerBeamState next_state = state;
        if (!plannerApplyGateToBlockState(primitives[idx],
                                          gate_qubits[idx],
                                          max_group_qubits,
                                          next_state.block_state)) {
          continue;
        }
        next_state.in_block[idx] = 1;
        next_state.chosen.push_back(idx);
        if (next_state.first_gate == std::numeric_limits<std::size_t>::max()) {
          next_state.first_gate = idx;
        }
        next_state.last_gate = idx;
        next_beam.push_back(std::move(next_state));
        expanded = true;
      }

      if (!state.chosen.empty() &&
          (!expanded || !have_best_state || plannerBeamStateBetter(state, best_state))) {
        best_state = state;
        have_best_state = true;
      }
    }

    if (next_beam.empty()) {
      break;
    }

    std::stable_sort(next_beam.begin(), next_beam.end(), plannerBeamStateBetter);
    if (next_beam.size() > kBeamWidth) {
      next_beam.resize(kBeamWidth);
    }
    beam = std::move(next_beam);
  }

  if (!have_best_state) {
    ready_candidates.clear();
    plannerCollectReadyCandidates(predecessors,
                                  globally_scheduled,
                                  initial.in_block,
                                  1,
                                  ready_candidates);
    if (ready_candidates.empty()) {
      return false;
    }
    const std::size_t fallback_idx = ready_candidates.front();
    PlannerBlockState fallback_state;
    if (!plannerApplyGateToBlockState(primitives[fallback_idx],
                                      gate_qubits[fallback_idx],
                                      max_group_qubits,
                                      fallback_state)) {
      return false;
    }
    best_state.chosen.push_back(fallback_idx);
    have_best_state = true;
  }

  chosen_block = std::move(best_state.chosen);
  return !chosen_block.empty();
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
  std::size_t scheduled_count = 0;

  out.ordered_primitives.reserve(n);
  out.block_sizes.reserve(n);

  while (scheduled_count < n) {
    std::vector<std::size_t> chosen_block;
    if (!plannerChooseNextBlock(primitives,
                                gate_qubits,
                                predecessors,
                                globally_scheduled,
                                max_group_qubits,
                                chosen_block)) {
      return false;
    }

    for (std::size_t idx : chosen_block) {
      if (globally_scheduled[idx]) {
        return false;
      }
      globally_scheduled[idx] = 1;
      ++scheduled_count;
      out.ordered_primitives.push_back(primitives[idx]);
    }
    out.block_sizes.push_back(chosen_block.size());
  }

  if (out.block_sizes.size() > 1) {
    std::vector<std::size_t> merged_block_sizes;
    merged_block_sizes.reserve(out.block_sizes.size());

    std::size_t gate_cursor = 0;
    std::size_t current_block_size = out.block_sizes[0];
    PlannerBlockState current_state;
    if (!plannerAppendGateRangeToState(out.ordered_primitives,
                                       gate_cursor,
                                       current_block_size,
                                       max_group_qubits,
                                       current_state)) {
      return false;
    }

    gate_cursor += current_block_size;
    for (std::size_t block_idx = 1; block_idx < out.block_sizes.size(); ++block_idx) {
      const std::size_t next_block_size = out.block_sizes[block_idx];
      PlannerBlockState trial_state = current_state;
      if (plannerAppendGateRangeToState(out.ordered_primitives,
                                        gate_cursor,
                                        next_block_size,
                                        max_group_qubits,
                                        trial_state)) {
        current_block_size += next_block_size;
        current_state = std::move(trial_state);
      } else {
        merged_block_sizes.push_back(current_block_size);
        current_block_size = next_block_size;
        current_state = PlannerBlockState{};
        if (!plannerAppendGateRangeToState(out.ordered_primitives,
                                           gate_cursor,
                                           current_block_size,
                                           max_group_qubits,
                                           current_state)) {
          return false;
        }
      }
      gate_cursor += next_block_size;
    }
    merged_block_sizes.push_back(current_block_size);
    out.block_sizes = std::move(merged_block_sizes);
  }

  return out.ordered_primitives.size() == primitives.size();
}

}  // namespace bqsim_rt
