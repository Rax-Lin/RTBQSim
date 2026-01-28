#pragma once

#include <memory>

#include "dd/Package.hpp"

namespace qc {

class FusedGate {
public:
  dd::mEdge fused_edge;
  int       num_edges;
  int       num_nodes;
  int       num_mac;
  bool      perm_or_dense;

  FusedGate() = default;
  FusedGate(dd::mEdge _fused_edge,
            int _num_mac,
            bool _perm_or_dense,
            std::unique_ptr<dd::Package<dd::DDPackageConfig>>&& dd)
      : fused_edge(_fused_edge),
        num_mac(_num_mac),
        perm_or_dense(_perm_or_dense) {
    num_nodes = dd->node_count(_fused_edge);
    num_edges = dd->edge_count(_fused_edge);
  }

  // Non-owning constructor for cases where dd is managed elsewhere.
  FusedGate(dd::mEdge _fused_edge,
            int _num_mac,
            bool _perm_or_dense,
            dd::Package<dd::DDPackageConfig>* dd)
      : fused_edge(_fused_edge),
        num_mac(_num_mac),
        perm_or_dense(_perm_or_dense) {
    if (dd) {
      num_nodes = dd->node_count(_fused_edge);
      num_edges = dd->edge_count(_fused_edge);
    } else {
      num_nodes = 0;
      num_edges = 0;
    }
  }
};

}  // namespace qc
