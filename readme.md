# BQSim 專案說明

我們透過 RTSpMSpM 進行 gate fusion，並以稀疏/CSR 稀疏矩陣乘法完成狀態向量更新。以下為專案的大致架構與 `bqsim_rt.sh` 參數說明。

## Demo 與測試建議
`cd BQSim`
`bash rt_compile.sh`
`bash bqsim_rt.sh`
如果想嘗試不同密度下的測試結果請主要修正位於bqsim_rt.sh 下的
* **BQSIM_RT_DENSITY_TARGET** : Gate 的 fusion 程度，密度限制
* **BQSIM_RT_DENSE_THRESHOLD** : 對於經過 fuse 的 Gate, 達到多少 density 後要使用 CSR/cuSPARSE 計算，低於此值則使用BQSim的 ell spmv 算法

目前我們的專案如下:
1. **gate fusion (RT)** : 與上個版本沒有變化，但是原本在 density 到達門檻後改成 CSR/cuSPARSE 而非原本的 GEMV ， 以避免過大的矩陣會爆炸
2. **batch simulation** : 同上，目前已經改掉了原本使用 gemv的部分。

會改掉原本的 GEMV 是因為對於 qbits >= 14 後，幾乎不可能 fused 出 density >= 1% 的 matrix，否則 VRAM 會爆炸。 由於我們使用 rtcore 進行 gate fusion 後，density 會比 BQSim 高。使用 cuSparse 搭配 tensor core 能夠更有效率地計算 `1% >= density >= 0.05%` 的 matrix。

### 改動結果 routing_n12、vqe_n12、vqe_n14、vqe_n16
* **優化方針** :  `BQSIM_RT_DENSITY_TARGET`、 `BQSIM_RT_DENSE_THRESHOLD` 兩個都盡量不要超過 0.01，也盡量不小於 0.00001。

#### routing_n12
[Stage 1: Gate Fusion] time: 0
[Stage 2: DD-to-ELL Conversion] time: 153
[Stage 3: ELL-based batch simulation] time: 618

#### vqe_n12
[Stage 1: Gate Fusion] time: 0
[Stage 2: DD-to-ELL Conversion] time: 131
[Stage 3: ELL-based batch simulation] time: 591

#### vqe_n14
[Stage 1: Gate Fusion] time: 0
[Stage 2: DD-to-ELL Conversion] time: 266
[Stage 3: ELL-based batch simulation] time: 2690

### vqe_n16
[Stage 1: Gate Fusion] time: 0
[Stage 2: DD-to-ELL Conversion] time: 1770
[Stage 3: ELL-based batch simulation] time: 18799

## Origins and Acknowledgements

This project is originally forked from and inspired by the following repositories:
* **BQSim**: https://github.com/IDEA-CUHK/BQSim.git
* **RTSpMSpM**: https://github.com/escalab/RTSpMSpM.git

## 專案大致架構
- **BQSim**：主程式，讀取 QASM 電路、建立 gate primitives、做 gate fusion，並執行狀態向量模擬。
- **RTSpMSpM 引擎**：用來做 gate fusion ，可用 RT Core 加速幾何建構與乘法。
- **稀疏計算路徑**：
  - ELL 路徑（ELL 格式 + CUDA kernel）
  - CSR 路徑（由 ELL 轉 CSR，使用 cuSPARSE SpMM）
- **批次模擬**：支援多 batch 狀態向量並行模擬，透過 ping-pong buffer 在 GPU 上交替存放結果。

## `bqsim_rt.sh` 參數說明（以目前腳本為準）

### Gate Fusion 相關
- `BQSIM_RT_PIPELINE_MODE=SPMSPM`
  - 使用 RTSpMSpM pipeline 做 gate fusion（取代 DD 的傳統路徑）。
- `BQSIM_RT_TARGET_FUSED_COUNT=4`
  - 目標融合區塊數量（把整個電路切成約 4 個 block 進行 fusion）。
- `BQSIM_RT_SPM_BLOCK_GATES=100`
  - 每個 block 最多容許的 gate 數量上限(通常用不到，可不改動此參數)。
- `BQSIM_RT_DENSITY_TARGET=0.1`
  - gate fusion 時的密度門檻（密度越高越容易切換為 dense）。
- `BQSIM_RT_BYPASS_DD_CACHE=1`
  - 在 pipeline 模式下略過 DD cache，節省記憶體。

### CSR 計算（cuSPARSE SpMM）相關
- `BQSIM_RT_HYBRID_DENSE=1`
  - 開啟 hybrid 模式，允許在 ELL 與 CSR 之間切換。
- `BQSIM_RT_CUSPARSE_TENSOR=1`
  - 使用 cuSPARSE SpMM 跑 CSR 路徑（是否用到 Tensor Core 取決於 GPU/驅動/庫版本與矩陣維度）。
- `BQSIM_RT_DENSE_THRESHOLD=0.05`
  - CSR 門檻：密度 ≥ 此值會改用 CSR + cuSPARSE。

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

## 執行方式（腳本內容）
`bqsim_rt.sh` 會依序跑多個 QASM 電路範例（tsp/routing/vqe/dnn/graph_state/portfolio 等），主要用來測試與產生效能結果：

```bash
bash bqsim_rt.sh
```

執行前請先完成 build（例如已有 `build-rt` 目錄與可執行檔 `BQSim`）。

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

## 建置流程（建議）
```bash
bash BQSim/rt_compile.sh
```

## 執行流程
```bash
bash BQSim/bqsim_rt.sh
```

## 輸出與紀錄檔
常見的輸出路徑如下（依實際執行參數可能略有調整）：
- `BQSim/log/`：執行結果與狀態輸出（例如 `log/results/state/*.txt`）
- `BQSim/log/fused_gates/`：若啟用匯出 fused gate，會輸出融合後的 gate 資訊
- `BQSim/build-rt/`：建置輸出（可忽略不提交）
