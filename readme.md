# RTBQSim Project Overview

This project uses **RTSpMSpM** for gate fusion and **ELL** for batched state-vector updates.
This document summarizes the project structure, execution workflow, and `rt_bqsim.sh` parameters (based on the current scripts), and compares results with NVIDIA cuQuantum.

---

## Current Goals and Recent Changes
- Stage 1 = original BQSim Stage 1 + Stage 2, Stage 2 = original BQSim Stage 3.
- Stage 2 (original Stage 3) has been switched back to ELL computation, so current experiments focus on gate fusion performance (Stage 1).
- Stage 1 RT-core gate fusion is instrumented with phased timing breakdowns.

---

## Origins and Acknowledgements

This project is originally forked from and inspired by the following repositories:
- **BQSim**: https://github.com/IDEA-CUHK/BQSim.git
- **RTSpMSpM**: https://github.com/escalab/RTSpMSpM.git

---

## High-Level Architecture
- **BQSim**: Main executable that reads QASM circuits, builds gate primitives, performs gate fusion, and runs state-vector simulation.
- **RTSpMSpM engine**: Used for gate fusion, with optional RT Core acceleration for geometry construction and multiplication.
- **Sparse compute path**:
  - ELL path (ELL format + CUDA kernel)

---

## `rt_bqsim.sh` Parameters (Current Script)

### Pre-launch Handling
- If `CUDA_VISIBLE_DEVICES` is an empty string, the script runs `unset CUDA_VISIBLE_DEVICES` first.
  - This avoids CUDA initialization errors.
- The script uses a single build directory: `build-rt/`.
  - If `build-rt/apps/RTBQSim` does not exist, it automatically rebuilds via `bash RTBQSim/rt_compile.sh`.

### Gate Fusion and Stop Behavior
- The project is fixed to the RTSpMSpM gate-fusion pipeline (SPMSPM).
- Row-NNZ early-stop is always enabled.
- Current row-NNZ limit is fixed at `4`.

### Numeric Precision Policy
- Stage-1/Stage-2 simulation numeric type is fixed to `fp64`.
- RTSpMSpM ray-hit geometry path keeps OptiX-required float-based geometry representation (`fp32`) where required by API/data layout.

### Stage-1 Timing and Scheduling Defaults
- Stage-1 timing is always measured in synchronized mode (CUDA event + synchronize) for stable phase-level timing.
- Stage-1 gate preparation is always scheduled in serial-prep-stream mode to reduce prep/main overlap and improve ray-generation timing fidelity.

### Optional Optimization Controls (for experiments)
- `RT_REUSE_BUFFER`
  - Reuse both GAS output buffers and sphere/ray geometry work buffers (avoids repeated `cudaMalloc/cudaFree`).
- `RT_GAS_ALLOW_UPDATE`
  - Allow GAS update instead of rebuild when primitive count is unchanged.
- `RT_DIAG_VALUE_ONLY`
  - For diagonal gates, update only values without rebuilding position/topology paths.

### GAS / BVH Update Strategy
- `RT_GAS_ALLOW_UPDATE=1`
  - Allow OptiX GAS update if primitive count is unchanged.
- `RT_REUSE_BUFFER=1`
  - Reuse GAS output and geometry work buffers during rebuild/update to reduce allocation overhead.

> `rt_bqsim.sh` has removed old dense/graph/mega-kernel environment parameters.
> The current focus is SPMSPM + ELL.

---

## Execution Flow (Current Script Behavior)
`rt_bqsim.sh` runs multiple QASM circuit sets in sequence (for example: tsp/routing/vqe/dnn/graph_state/portfolio), with fixed settings per case:

- `--ps --pv`
- `--batch_size 256`
- `--num_batch 200`
- `--conversion_type 2`

This is used for apples-to-apples performance comparisons and state output generation on the current RT gate-fusion path.

---

## Build Requirements and Environment
Current default dependencies in `rt_compile.sh` (overridable via environment variables):
- **CUDA Toolkit** (including `nvcc` and matching driver)
- **OptiX SDK** (`OptiX_INSTALL_DIR`)
- **GCC/G++** (default `/usr/bin/gcc-9`, `/usr/bin/g++-9`)
- **OpenMP** (default `libgomp`)
- **cuQuantum** (`CUQUANTUM_ROOT`; adjust if not required in your setup)

Recommended environment variables to verify (set in shell or scripts):
- `OptiX_INSTALL_DIR`
- `CMAKE_CUDA_ARCHITECTURES`
- `CMAKE_CUDA_HOST_COMPILER`
- `CMAKE_C_COMPILER`
- `CMAKE_CXX_COMPILER`
- `CUQUANTUM_ROOT`

Notes:
- If `CMAKE_CUDA_ARCHITECTURES` is not set, `rt_compile.sh` auto-detects GPU compute capability.
- If the exact architecture is unsupported by current `nvcc` (for example, very new GPUs), it falls back to the highest compatible architecture.

---

## Run (No Docker)
(Simulation precision is fixed to fp64.)
```bash
bash rt_compile.sh
```
```bash
bash rt_bqsim.sh
```

## Run (With Docker)
Method 1. Build image and enter container (interactive mode)
```bash
./run_docker.sh --build
```
Then run inside container:
```bash
bash RTBQSim/rt_compile.sh
bash RTBQSim/rt_bqsim.sh
```

Method 2. Auto-run compile + execute inside container (`rt_compile.sh` + `rt_bqsim.sh`)
```bash
./run_docker.sh --auto-run
```

Notes:
- `run_docker.sh` prioritizes `RTBQSIM_OPTIX_DIR`. If unset, it searches upward from the project path and recursively scans downward for a complete OptiX SDK (must include `include/optix.h` and `SDK/sutil/Preprocessor.h`).
- `RTBQSIM_OPTIX_SEARCH_DEPTH` controls recursive search depth (default: 6).
- Docker build output uses a volume (default `rtbqsim-build`) mounted to `RTBQSim/build-rt` in the container to avoid polluting local repo files.
- `RTBQSim/log/...` outputs are written back to host because the repo directory is bind-mounted.

---

## Baseline (cuQuantum) Workflow
The project keeps a cuQuantum baseline path for comparison against `RTBQSim`.

- Compile baseline target:
```bash
bash RTBQSim/cuquantum_compile.sh
```
- Run baseline batch suite:
```bash
bash RTBQSim/cuquantum.sh
```
### Docker usage for baseline
Run inside container:
```bash
bash RTBQSim/cuquantum_compile.sh
bash RTBQSim/cuquantum.sh
```
Or directly from host through docker runner:
```bash
./run_docker.sh -- bash -lc "bash RTBQSim/cuquantum_compile.sh && bash RTBQSim/cuquantum.sh"
```

---

## Outputs and Logs
Common output paths (may vary slightly by runtime options):
- `RTBQSim/log/`: run outputs and state dumps (for example, `log/results/state/*.txt`)
- `RTBQSim/log/fused_gates/`: fused-gate exports when enabled
- `RTBQSim/build-rt/`: build artifacts
