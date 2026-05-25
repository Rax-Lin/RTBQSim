#!/bin/bash
cd build/cuquantum_test
# Args: num_qubits, batchsize, num_batches, use_fused_gates: (0-no fusion), and output_or_not (0 or 1)
./cuquantum dnn 17 32 10 0 0
./cuquantum dnn 19 32 10 0 0
./cuquantum dnn 21 32 10 0 0
./cuquantum vqe 12 32 10 0 0
./cuquantum vqe 14 32 10 0 0
./cuquantum vqe 16 32 10 0 0
./cuquantum portfolio_vqe 16 32 10 0 0
./cuquantum portfolio_vqe 17 32 10 0 0
./cuquantum portfolio_vqe 18 32 10 0 0
./cuquantum graph_state 16 32 10 0 0
./cuquantum graph_state 18 32 10 0 0
./cuquantum graph_state 20 32 10 0 0
./cuquantum tsp 9 32 10 0 0
./cuquantum tsp 16 32 10 0 0
./cuquantum routing 6 32 10 0 0
./cuquantum routing 12 32 10 0 0