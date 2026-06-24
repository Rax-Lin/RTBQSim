#pragma once

#include "base.hpp"
#include "util.hpp"
#include <chrono>
#include <iomanip>
#include <thrust/scan.h>

__global__ void replicate(cuDoubleComplex *input_arr_d, int N) {
  input_arr_d[N*threadIdx.x+blockIdx.x] = input_arr_d[blockIdx.x];
}

__global__ void initial_check(cuDoubleComplex *input_arr_d, bool *identical, int N) {
  extern __shared__ bool s[];
  __shared__ int res[1];
  if (threadIdx.x == 0) {
    res[0] = true;
  }
  __syncthreads();
  s[threadIdx.x] = ((input_arr_d[N*threadIdx.x+blockIdx.x].x == input_arr_d[blockIdx.x].x) && 
    (input_arr_d[N*threadIdx.x+blockIdx.x].y == input_arr_d[blockIdx.x].y));
  __syncthreads();
  atomicAnd(res, (int)s[threadIdx.x]);
  __syncthreads();
  if (threadIdx.x == 0) {
    identical[blockIdx.x] = res[0];
  }
}


class CuQBatch : public Base
{

public:
  CuQBatch(
    int _n_qubit,
    int _nSvSize,
    int _batchSize,
    std::vector<qpp::QCircuit::double2 *> _mat_vec,
    std::vector<int> _ctrl_vec,
    std::vector<std::vector<int>> _target_vec,
    int _n_batch
  );
  CuQBatch(
    int _n_qubit,
    int _nSvSize,
    int _batchSize,
    std::vector<cuDoubleComplex *> _mat_vec,
    std::vector<int> _ctrl_vec,
    std::vector<std::vector<int>> _target_vec,
    int _n_batch
  );
  void BatchSim() override;
  void ReadInputs() override;
  ~CuQBatch();
};

void CuQBatch::ReadInputs() {
  const auto begin = std::chrono::steady_clock::now();
  std::ifstream file;
  file.open((filename).c_str());

  if (!file.is_open()) {
    std::cerr << "Failed to open file." << std::endl;
    exit(-1);
  }
  std::string line;
  while (getline(file, line)) {
    std::istringstream iss(line);
    double real, imag;
    int amp_id = 0;
    while (iss >> real >> imag) {
      input_arr[amp_id] = {real, imag};
      amp_id++;
    }
  }

  file.close();

  // replicate
  cuDoubleComplex *input_arr_d;
  timedCudaMallocDevice((void**)&input_arr_d, nSvSize * batchSize * sizeof(cuDoubleComplex), init_alloc_ms);
  cudaMemcpy(input_arr_d, input_arr, nSvSize * batchSize * sizeof(cuDoubleComplex),
                cudaMemcpyHostToDevice);
  replicate<<<nSvSize, batchSize>>>(input_arr_d, nSvSize);
  cudaMemcpy(input_arr, input_arr_d, nSvSize * batchSize * sizeof(cuDoubleComplex),
                cudaMemcpyDeviceToHost);
  cudaFree(input_arr_d);
  const auto end = std::chrono::steady_clock::now();
  last_read_inputs_ms =
      std::chrono::duration<double, std::milli>(end - begin).count();
}


CuQBatch::CuQBatch(
  int _n_qubit,
  int _nSvSize,
  int _batchSize,
  std::vector<qpp::QCircuit::double2 *> _mat_vec,
  std::vector<int> _ctrl_vec,
  std::vector<std::vector<int>> _target_vec,
  int _n_batch
) : Base(_n_qubit, _nSvSize, _batchSize, _mat_vec, _ctrl_vec, _target_vec, _n_batch)
{  }

CuQBatch::CuQBatch(
  int _n_qubit,
  int _nSvSize,
  int _batchSize,
  std::vector<cuDoubleComplex *> _mat_vec,
  std::vector<int> _ctrl_vec,
  std::vector<std::vector<int>> _target_vec,
  int _n_batch
) : Base(_n_qubit, _nSvSize, _batchSize, _mat_vec, _ctrl_vec, _target_vec, _n_batch)
{  }

void CuQBatch::BatchSim() {
  ReadInputs();
  const auto setup_begin = std::chrono::steady_clock::now();
  int adjoint    = 0;
  cuDoubleComplex *h_batchsv_result;
  timedCudaMallocHost((void**)&h_batchsv_result, nSvSize * batchSize * sizeof(cuDoubleComplex), init_alloc_ms);

  cuDoubleComplex *d_batchsv;
  custatevecHandle_t handle;
  custatevecCreate(&handle);
  timedCudaMallocDevice((void**)&d_batchsv, nSvSize * batchSize * sizeof(cuDoubleComplex), init_alloc_ms);
  const auto setup_end = std::chrono::steady_clock::now();
  last_setup_ms =
      std::chrono::duration<double, std::milli>(setup_end - setup_begin).count();
  auto begin = std::chrono::steady_clock::now();
  for (size_t bid = 0; bid < n_batch; bid++)
  {
    cudaMemcpy(d_batchsv, input_arr, nSvSize * batchSize * sizeof(cuDoubleComplex),
                cudaMemcpyHostToDevice);

    for (int gid = 0; gid < matrix_vec.size(); gid++)  { 
      void* extraWorkspace = nullptr;
      size_t extraWorkspaceSizeInBytes = 0;
      int* targets  = new int[target_vec[gid].size()];
      for (int tar_id = 0; tar_id < target_vec[gid].size(); tar_id++) {
        targets[tar_id] = target_vec[gid][tar_id];
      }
      int controls[] = {ctrl_vec[gid]};
      custatevecApplyMatrixBatchedGetWorkspaceSize(
        handle, 
        CUDA_C_64F, 
        n_qubit, 
        batchSize,
        nSvSize,
        CUSTATEVEC_MATRIX_MAP_TYPE_BROADCAST,
        nullptr,
        matrix_vec[gid], 
        CUDA_C_64F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, 
        adjoint, 
        1,
        target_vec[gid].size(), // nTargets, 
        controls[0] != -1? 1 : 0, // nControls,
        CUSTATEVEC_COMPUTE_64F, 
        &extraWorkspaceSizeInBytes
      );

      // allocate external workspace if necessary
      if (extraWorkspaceSizeInBytes > 0)
          cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes);

      // apply gate
      custatevecApplyMatrixBatched(
        handle, 
        d_batchsv, 
        CUDA_C_64F, 
        n_qubit, 
        batchSize,
        nSvSize,
        CUSTATEVEC_MATRIX_MAP_TYPE_BROADCAST,
        nullptr,
        matrix_vec[gid], 
        CUDA_C_64F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, 
        adjoint, 
        1,
        targets, 
        target_vec[gid].size(), // nTargets, 
        controls,
        nullptr, 
        controls[0] != -1? 1 : 0, // nControls, 
        CUSTATEVEC_COMPUTE_64F,
        extraWorkspace, 
        extraWorkspaceSizeInBytes
      );
          // cudaDeviceSynchronize();
      if (extraWorkspaceSizeInBytes)
        cudaFree(extraWorkspace);

      delete [] targets;
    }
    cudaMemcpy(h_batchsv_result, d_batchsv, nSvSize * batchSize * sizeof(cuDoubleComplex),
                cudaMemcpyDeviceToHost);
  }
  auto end = std::chrono::steady_clock::now();
  const double runtime_ms =
      std::chrono::duration<double, std::milli>(end - begin).count();
  last_runtime_ms = runtime_ms;
  std::cout << "cuQuantum runtime: "
            << std::fixed << std::setprecision(2) << runtime_ms
            << " [ms]" << std::endl;

  bool *identical_d;
  bool *identical_h;
  timedCudaMallocDevice((void**)&identical_d, nSvSize*sizeof(bool), init_alloc_ms);
  timedCudaMallocHost((void**)&identical_h, nSvSize*sizeof(bool), init_alloc_ms);
  initial_check<<<nSvSize, batchSize, batchSize*sizeof(bool)>>>(d_batchsv, identical_d, nSvSize);
  cudaMemcpy(identical_h, identical_d, nSvSize*sizeof(bool), cudaMemcpyDeviceToHost);
  bool identical_res = true;
  for (int i = 0; i < nSvSize; i++) {
    if (!identical_h[i]) {
      identical_res = false;
      break;
    }
  }
  std::cout << "Initial check: "<<identical_res << std::endl;
  // cudaMallocHost((void**)&output_arr, nSvSize * sizeof(cuDoubleComplex));
  output_arr = h_batchsv_result;

  cudaFree(d_batchsv);
  // destroy handle
  custatevecDestroy(handle);

  cudaFreeHost(input_arr);
  std::cout << std::endl;
}

CuQBatch::~CuQBatch() {
  for (size_t i = 0; i < matrix_vec.size(); i++)
  {
    cudaFreeHost(matrix_vec[i]);
  }
  cudaFreeHost(input_arr);
}
