# BQSim 專案說明

本專案透過 **RTSpMSpM** 進行 gate fusion，並以 **ELL** 完成批次狀態向量更新。  
下方整理專案架構、執行方式與 `bqsim_rt.sh` 參數（以目前腳本為準）。

---

## 當前目標與改動
 - stage 1 = BQSim 的 stage 1 + 2, stage 2 = BQSim 的 stage 3
 - 已經將 stage 2(原stage 3) 改回 ell 計算，先單純比較 gate fusion (stage 1)
 - stage 1 的 rtcore gate fusion 分階段進行分析時間
---
## Origins and Acknowledgements

This project is originally forked from and inspired by the following repositories:
* **BQSim**: https://github.com/IDEA-CUHK/BQSim.git
* **RTSpMSpM**: https://github.com/escalab/RTSpMSpM.git

---
## 專案大致架構
- **BQSim**：主程式，讀取 QASM 電路、建立 gate primitives、做 gate fusion，並執行狀態向量模擬。
- **RTSpMSpM 引擎**：用來做 gate fusion ，可用 RT Core 加速幾何建構與乘法。
- **稀疏計算路徑**：
  - ELL 路徑（ELL 格式 + CUDA kernel）

---
## `bqsim_rt.sh` 參數說明（以目前腳本為準）

### 腳本啟動前處理
- 若 `CUDA_VISIBLE_DEVICES` 是空字串，腳本會先 `unset CUDA_VISIBLE_DEVICES`
  - 避免 CUDA 初始化時出現 `initialization error`。
- 腳本會檢查 `build-rt/apps/BQSim` 是否存在
  - 若不存在，提示先執行 `bash BQSim/rt_compile.sh`。

### Gate Fusion 與停止條件
- `BQSIM_RT_PIPELINE_MODE=SPMSPM`
  - 啟用 RTSpMSpM gate fusion 流程（目前腳本預設）。
- `BQSIM_RT_FORCE_FULL_FUSION=0`
  - `1`：不因 row nnz 限制提前停止，盡量 fuse 到 block 上限。
  - `0`：維持目前的提前停止策略。

### 參數啟用優化(可供實驗使用)
- `BQSIM_RT_GAS_REUSE_OUTPUT_BUFFER`
  - 控制 GAS output buffer 是否重用（容量足夠時可避免重複 `cudaMalloc/cudaFree`）。
- `BQSIM_RT_REUSE_GEOMETRY_BUFFER`
  - 控制 sphere/ray 幾何工作區是否採用容量重用（預設跟隨 `BQSIM_RT_GAS_REUSE_OUTPUT_BUFFER`）。
- `BQSIM_RT_GAS_ALLOW_UPDATE`
  - primitive 數量不變時，允許 GAS 用 update 取代 rebuild（等同於每一 row 之 nnz 數量不變的話，進行更新）。
- `BQSIM_RT_DIAG_VALUE_ONLY`
  - 對 diagonal gate 啟用只更新值、不重算位置路徑。


### GAS / BVH 更新策略
- `BQSIM_RT_GAS_ALLOW_UPDATE=1`
  - primitive 數量不變時，允許 OptiX GAS update。
- `BQSIM_RT_GAS_UPDATE_INTERVAL=0`
  - 連續 update 的上限；達上限後做一次 rebuild（`0` 代表不限制）。
- `BQSIM_RT_GAS_REUSE_OUTPUT_BUFFER=1`
  - 重建 GAS 時重用 output buffer（容量足夠時），降低 `cudaMalloc/cudaFree` 開銷。

> 目前 `bqsim_rt.sh` 已移除舊 dense/graph/mega-kernel 相關環境參數，
> 以 SPMSPM + ELL 路徑為主。

---
## 執行方式（腳本內容）
`bqsim_rt.sh` 會依序跑多個 QASM 電路範例（tsp/routing/vqe/dnn/graph_state/portfolio 等），每個 case 目前固定使用：

- `--ps --pv`
- `--batch_size 1`
- `--num_batch 1`
- `--conversion_type 2`

用途是針對目前 RT gate fusion 路徑做一致條件的效能比較與狀態輸出。

---
## 建置需求與環境
以下為 `rt_compile.sh` 目前預設的建置依賴（可用環境變數覆蓋）：
- **CUDA Toolkit**（需含 nvcc 與對應 driver）
- **OptiX SDK**（`OptiX_INSTALL_DIR`）
- **GCC/G++**（預設 `/usr/bin/gcc-9`, `/usr/bin/g++-9`）
- **OpenMP**（預設使用 `libgomp`）
- **cuQuantum**（`CUQUANTUM_ROOT`，若未用到可視需求調整）

建議確認下列環境變數（可在 shell 或腳本內設定）：
- `OptiX_INSTALL_DIR`
- `CMAKE_CUDA_ARCHITECTURES`
- `CMAKE_CUDA_HOST_COMPILER`
- `CMAKE_C_COMPILER`
- `CMAKE_CXX_COMPILER`
- `CUQUANTUM_ROOT`

---
## Run（no docker）
```bash
bash BQSim/rt_compile.sh
```
---
```bash
bash BQSim/bqsim_rt.sh
```

## Run (with docker)
方法 1. 建 image 並進入 container 
```bash
./run_docker.sh --build
bash BQSim/rt_compile.sh
bash BQSim/bqsim_rt.sh
```

方法 2. 在 container 內自動執行編譯 + 執行（`rt_compile.sh` + `bqsim_rt.sh`）
```bash
./run_docker.sh --auto-run
```

---
## 輸出與紀錄檔
常見的輸出路徑如下（依實際執行參數可能略有調整）：
- `BQSim/log/`：執行結果與狀態輸出（例如 `log/results/state/*.txt`）
- `BQSim/log/fused_gates/`：若啟用匯出 fused gate，會輸出融合後的 gate 資訊
- `BQSim/build-rt/`：建置輸出
