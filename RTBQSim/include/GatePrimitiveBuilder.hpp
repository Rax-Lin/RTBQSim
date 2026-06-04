#pragma once

#include <cmath>
#include <initializer_list>
#include <iostream>
#include <vector>

#include "Definitions.hpp"
#include "GatePrimitive.hpp"
#include "QuantumComputation.hpp"

namespace bqsim_rt {

inline bool buildGatePrimitives(const qc::QuantumComputation& qc_obj,
                                std::vector<qc::GatePrimitive>& out) {
  out.clear();
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

  for (const auto& op : qc_obj) {
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

}  // namespace bqsim_rt
