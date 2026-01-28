# BQSim 專案說明

我們透過 RTSpMSpM（光線追蹤式 SpM×SpM）進行 gate fusion，並以稀疏/稠密矩陣乘法完成狀態向量更新。以下為專案的大致架構與 `bqsim_rt.sh` 參數說明。
## Origins and Acknowledgements

This project is originally forked from and inspired by the following repositories:
* **BQSim**: https://github.com/IDEA-CUHK/BQSim.git
* **RTSpMSpM**: https://github.com/escalab/RTSpMSpM.git

## 專案大致架構
- **BQSim**：主程式，讀取 QASM 電路、建立 gate primitives、做 gate fusion，並執行狀態向量模擬。
- **RTSpMSpM 引擎**：用來做 gate fusion ，可用 RT Core 加速幾何建構與乘法。
- **稀疏/稠密計算路徑**：
  - 稀疏路徑（ELL 格式 + CUDA kernel）
  - 稠密路徑（自寫 GEMV 或 cuBLAS）
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

### Dense 計算（SPMV/GEMV）相關
- `BQSIM_RT_HYBRID_DENSE=1`
  - 開啟 hybrid dense 模式，允許在稀疏與稠密表示之間切換。
- `BQSIM_RT_DENSE_GEMV=1`
  - 使用自訂 GEMV kernel 進行稠密矩陣-向量乘法(通常用不到)。
- `BQSIM_RT_DENSE_CUBLAS=1`
  - 使用 cuBLAS（可用 tensor core）進行稠密矩陣-向量乘法(若與上述 GEMV 同時為1，以 tensor 為主)。
- `BQSIM_RT_DENSE_THRESHOLD=0.05`
  - 稠密化門檻：密度 ≥ 此值會改用 dense 路徑。
- `BQSIM_RT_DENSE_MAX_BYTES=536870912`
  - dense 矩陣最大允許大小（單位 bytes），避免 OOM。

### GPU Kernel 執行相關
- `BQSIM_RT_DENSE_TILE=256`
  - dense kernel 的 tile 大小。
- `BQSIM_RT_DENSE_ASSUME_DENSE=0`
  - 若為 1，視為完全稠密，不做稀疏判斷(目前用不到，做 debug tensor core 的計算時間使用)。
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
