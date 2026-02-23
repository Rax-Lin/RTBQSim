# BQSim 專案說明

本專案透過 **RTSpMSpM** 進行 gate fusion，並以 **ELL** 完成批次狀態向量更新。  
下方整理專案架構、執行方式與 `bqsim_rt.sh` 參數（以目前腳本為準）。

---

## Demo 與測試建議
```bash
cd BQSim
bash rt_compile.sh
bash bqsim_rt.sh
```

## 當前目標與改動
 - stage 1 = BQSim 的 stage 1 + 2, stage 2 = BQSim 的 stage 3
 - 已經將 stage 2(原stage 3) 改回 ell 計算，先單純比較 gate fusion (stage 1)
 - 將計算與 Baseline(BQSim)對齊，優先對 stage 1 的 rtcore fusion 進行分析與優化
 - stage 1 的 rtcore gate fusion 分階段分析時間
 - 目前實作的 rt gate fusion 尚未採行蓋大樓的方式(之前的方法算出錯的結果)
 - bvh build time 需優化
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
- **批次模擬**：支援多 batch 狀態向量並行模擬，透過 ping-pong buffer 在 GPU 上交替存放結果。

---
## `bqsim_rt.sh` 參數說明（以目前腳本為準）

### Gate Fusion 相關
- `BQSIM_RT_PIPELINE_MODE=SPMSPM`
  - 使用 RTSpMSpM pipeline 做 gate fusion（取代 DD 的傳統路徑）。

### CSR 計算（cuSPARSE SpMM）相關
- `BQSIM_RT_HYBRID_DENSE=1`
  - 開啟 hybrid 模式，允許在 ELL 與 CSR 之間切換。


### GPU Kernel 執行相關
- `BQSIM_RT_DENSE_TILE=256`
  - 目前保留參數（舊 dense 路徑用，現階段不影響 CSR/ELL）。
- `BQSIM_RT_DENSE_ASSUME_DENSE=0`
  - 目前保留參數（舊 dense 路徑用，現階段不影響 CSR/ELL）。
- `BQSIM_RT_COMPACT_LAUNCH=1`
  - 使用較緊湊的 kernel launch 流程（減少 CPU overhead）。
- `BQSIM_RT_USE_CUDA_GRAPH=1`
  - 使用 CUDA Graph 捕捉 kernel launch，降低 launch overhead（但增加 GPU memory）。
- `BQSIM_RT_MEGA_KERNEL=0`
  - 若為 1，使用 mega-kernel 一次執行全部 gate（需 GPU 支援 grid-wide sync，目前也用不到）。

### GAS / BVH 更新策略
- `BQSIM_RT_GAS_ALLOW_UPDATE=1`
  - 允許在 primitive 數量不變時使用 OptiX GAS update。
- `BQSIM_RT_GAS_UPDATE_INTERVAL=16`
  - 連續 update 次數上限；達上限後強制做一次 rebuild（`0` 代表不限制）。
- `BQSIM_RT_GAS_UPDATE_MIN_PRIMS=0`
  - primitive 數量小於此值時，改走 rebuild（避免小規模場景 update 不划算）。
- `BQSIM_RT_GAS_REUSE_OUTPUT_BUFFER=1`
  - rebuild 時重用既有 GAS output buffer（容量足夠時），減少 `cudaMalloc/cudaFree` 開銷。

---
## 執行方式（腳本內容）
`bqsim_rt.sh` 會依序跑多個 QASM 電路範例（tsp/routing/vqe/dnn/graph_state/portfolio 等），主要用來測試與產生效能結果：

```bash
bash bqsim_rt.sh
```

執行前請先完成 build（例如已有 `build-rt` 目錄與可執行檔 `BQSim`）。

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
## 建置流程（建議）
```bash
bash BQSim/rt_compile.sh
```

---
## 執行流程
```bash
bash BQSim/bqsim_rt.sh
```

---
## 輸出與紀錄檔
常見的輸出路徑如下（依實際執行參數可能略有調整）：
- `BQSim/log/`：執行結果與狀態輸出（例如 `log/results/state/*.txt`）
- `BQSim/log/fused_gates/`：若啟用匯出 fused gate，會輸出融合後的 gate 資訊
- `BQSim/build-rt/`：建置輸出（可忽略不提交）
