#!/bin/bash
cd build/cuquantum_test
# Args: num_qubits, batchsize, num_batches, use_fused_gates: (0-no fusion), and output_or_not (0 or 1)
./cuquantum tsp 9 32 50 0 0
./cuquantum tsp 16 32 50 0 0
./cuquantum vqe 12 32 50 0 0
./cuquantum vqe 14 32 50 0 0
./cuquantum vqe 16 32 50 0 0
./cuquantum qv 12 32 50 0 0
./cuquantum qv 14 32 50 0 0
./cuquantum qaoa 13 32 50 0 0
./cuquantum qaoa 15 32 50 0 0
./cuquantum qft 14 32 50 0 0
./cuquantum qft 16 32 50 0 0
./cuquantum qft 18 32 50 0 0
./cuquantum portfolio_vqe 16 32 50 0 0
./cuquantum portfolio_vqe 17 32 50 0 0
./cuquantum portfolio_vqe 18 32 50 0 0
./cuquantum graph_state 16 32 50 0 0
./cuquantum graph_state 18 32 50 0 0
