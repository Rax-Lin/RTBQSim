/**
 * cuSparse spmspm
 * 
 * Purpose: Baseline for spmspm on gpu side.
 *
 * To Compile:
 *    Run command "make" in the build folder
 * To run: 
 * /home/OptixSDK/cuSparse/src/cuSparse -m1 "/home/OptixSDK/Tool/PythonTool/output/spmGen1024_1048.mtx" -m2 "/home/OptixSDK/Tool/PythonTool/output/spmGen1024_1048.mtx" -o "./out.mtx"
 * /home/OptixSDK/cuSparse/src/cuSparse -m1 "/home/OptixSDK/Tool/PythonTool/output/spmGen8192_67108.mtx" -m2 "/home/OptixSDK/Tool/PythonTool/output/spmGen8192_67108.mtx" -o "./out.mtx"
 * /home/OptixSDK/cuSparse/src/cuSparse -m1 "/home/OptixSDK/Tool/PythonTool/output/spmGen4096_16777.mtx" -m2 "/home/OptixSDK/Tool/PythonTool/output/spmGen4096_16777.mtx" -o "./out.mtx"
 *
 * Nsys command profiling:
 * nsys nvprof ./cuSparse -m1 "../../../Tool/PythonTool/output/spmGen1024_1048.mtx" -m2 "../../../Tool/PythonTool/output/spmGen1024_1048.mtx”
 * 
 * To Debug:
 * cuda-gdb /home/OptixSDK/cuSparse/src/cuSparse 
 * run -m1 "/home/OptixSDK/Tool/PythonTool/output/spmGen16384_268435.mtx" -m2 "/home/OptixSDK/Tool/PythonTool/output/spmGen16384_268435.mtx" -o "./out.mtx"
*/

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpSM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono> 
using namespace std::chrono;

#include "util.cpp"
#include "Timing.cpp"

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --output | -o <filename>    Specify file for image output\n";
    std::cerr << "         --mat1 | -m1 <filename>     Specify file for matrix 1\n";
    std::cerr << "         --mat2 | -m2 <filename>     Specify file for matrix 2\n";
    std::cerr << "         --log  | -l  <filename>     Specify file for log\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    exit( 1 );
}

void getComputeType( cusparseSpMatDescr_t mat )
{
    cudaDataType dataType;
    size_t dataSize = sizeof(cudaDataType);

    // Query the data type of the sparse matrix
    cusparseSpMatGetAttribute(
        mat, 
        CUSPARSE_SPMAT_DIAG_TYPE, 
        &dataType, 
        dataSize
    );

    switch (dataType) {
        case CUDA_R_32F:
            std::cout << "Matrix data type: CUDA_R_32F (float)" << std::endl;
            break;
        case CUDA_R_64F:
            std::cout << "Matrix data type: CUDA_R_64F (double)" << std::endl;
            break;
        case CUDA_C_32F:
            std::cout << "Matrix data type: CUDA_C_32F (complex float)" << std::endl;
            break;
        case CUDA_C_64F:
            std::cout << "Matrix data type: CUDA_C_64F (complex double)" << std::endl;
            break;
        default:
            std::cout << "Unknown data type" << std::endl;
            break;
    }
}

int reuseCompute( std::string matrix1File, std::string matrix2File, std:: string outfile, std:: string logFile )
{
    cudaEvent_t start, stop;
    float h2d_ms = 0, work_est_ms = 0, compute_ms = 0, d2h_ms = 0;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Host Matrix Definition 
    int* hA_mat_rows, *hB_mat_rows, *hC_mat_rows;
    int* hA_mat_cols, *hB_mat_cols, *hC_mat_cols;
    float* hA_mat_values, *hB_mat_values, *hC_mat_svalues;
    int hA_mat_num_rows, hB_mat_num_rows, hC_mat_num_rows;
    int hA_mat_num_cols, hB_mat_num_cols, hC_mat_num_cols;
    uint64_t hA_mat_nnz, hB_mat_nnz, hC_mat_nnz;

    cooFromFile(matrix1File, &hA_mat_rows, &hA_mat_cols, &hA_mat_values, &hA_mat_num_rows,
                &hA_mat_num_cols, &hA_mat_nnz);
    cooFromFile(matrix2File, &hB_mat_rows, &hB_mat_cols, &hB_mat_values, &hB_mat_num_rows,
                &hB_mat_num_cols, &hB_mat_nnz);

    float               alpha       = 1.0f;
    float               beta        = 0.0f;
    cusparseOperation_t opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType        computeType = CUDA_R_32F;
    //--------------------------------------------------------------------------
    // Device Memory Management
    int   *dA_rows, *dA_cols, *dB_rows, *dB_cols, *dC_rows, *dC_cols;
    float *dA_values, *dB_values, *dC_values;
    int   *dA_csrPtr, *dB_csrPtr, *dC_csrPtr;
    // Change coo to csr Uncomment if >64
    // hA_csr_rows = new int[hA_mat_num_rows + 1];
    // hB_csr_rows = new int[hB_mat_num_rows + 1];
    // coo_to_csr(hA_mat_rows, hA_mat_nnz, hA_mat_num_rows, hA_csr_rows);
    // coo_to_csr(hB_mat_rows, hB_mat_nnz, hB_mat_num_rows, hB_csr_rows);

    // ==========================================================================
    // STAGE 1: H2D Transfer & Allocation (Wall-clock)
    // ==========================================================================
    auto h2d_start = high_resolution_clock::now();

    // Allocate device memory for matrix A
    CHECK_CUDA( cudaMalloc((void**)&dA_cols, hA_mat_nnz * sizeof(int)));
    CHECK_CUDA( cudaMalloc((void**)&dA_values, hA_mat_nnz * sizeof(float)));
    CHECK_CUDA( cudaMalloc((void**)&dA_csrPtr, (hA_mat_num_rows + 1) * sizeof(int)));

    // Allocate device memory for matrix B
    CHECK_CUDA( cudaMalloc((void**)&dB_cols, hB_mat_nnz * sizeof(int)));
    CHECK_CUDA( cudaMalloc((void**)&dB_values, hB_mat_nnz * sizeof(float)));
    CHECK_CUDA( cudaMalloc((void**)&dB_csrPtr, (hB_mat_num_rows + 1) * sizeof(int)));

    CHECK_CUDA( cudaMalloc((void**) &dC_csrPtr,
                        (hA_mat_num_rows + 1) * sizeof(int)) );

    // Copy matrix A data from host to device
    CHECK_CUDA( cudaMemcpy(dA_cols, hA_mat_cols, hA_mat_nnz * sizeof(int),
                        cudaMemcpyHostToDevice));
    CHECK_CUDA( cudaMemcpy(dA_values, hA_mat_values, hA_mat_nnz * sizeof(float),
                        cudaMemcpyHostToDevice));

    // Copy matrix B data from host to device
    CHECK_CUDA( cudaMemcpy(dB_cols, hB_mat_cols, hB_mat_nnz * sizeof(int), 
                        cudaMemcpyHostToDevice));
    CHECK_CUDA( cudaMemcpy(dB_values, hB_mat_values, hB_mat_nnz * sizeof(float), 
                        cudaMemcpyHostToDevice));
    
    // Comment out if >64
    CHECK_CUDA( cudaMalloc((void**)&dA_rows, hA_mat_nnz * sizeof(int)));
    CHECK_CUDA( cudaMalloc((void**)&dB_rows, hB_mat_nnz * sizeof(int)));
    CHECK_CUDA( cudaMemcpy(dA_rows, hA_mat_rows, hA_mat_nnz * sizeof(int),
                        cudaMemcpyHostToDevice));
    CHECK_CUDA( cudaMemcpy(dB_rows, hB_mat_rows, hB_mat_nnz * sizeof(int), 
                        cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaDeviceSynchronize());
    auto h2d_stop = high_resolution_clock::now();
    h2d_ms = duration_cast<duration<double, std::milli>>(h2d_stop - h2d_start).count();
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    // ==========================================================================
    // STAGE 2: Work Estimation & Analyze (Wall-clock)
    // ==========================================================================
    auto work_start = high_resolution_clock::now();

    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA, matB, matC;
    void*  dBuffer1    = NULL;
    void*  dBuffer2    = NULL;
    void*  dBuffer3    = NULL;
    void*  dBuffer4    = NULL;
    void*  dBuffer5    = NULL;
    size_t bufferSize3 = 0;
    size_t bufferSize4 = 0;
    size_t bufferSize5 = 0;
    size_t bufferSize1 = 0,    bufferSize2 = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) );
    // Change matrix row ptr to CSR format (Uncomment if >64)
    // CHECK_CUDA( cudaMemcpy(dA_csrPtr, hA_csr_rows, (hA_mat_num_rows + 1) * sizeof(int),
    //                     cudaMemcpyHostToDevice));
    // CHECK_CUDA( cudaMemcpy(dB_csrPtr, hB_csr_rows, (hB_mat_num_rows + 1) * sizeof(int),
    //                     cudaMemcpyHostToDevice));

    // Comment out if >64
    cusparseXcoo2csr(handle, dA_rows, hA_mat_nnz, hA_mat_num_rows, dA_csrPtr,
                    CUSPARSE_INDEX_BASE_ONE );
    cusparseXcoo2csr(handle, dB_rows, hB_mat_nnz, hB_mat_num_rows, dB_csrPtr,
                    CUSPARSE_INDEX_BASE_ONE );

    // Create sparse matrix in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, hA_mat_num_rows, hA_mat_num_cols, hA_mat_nnz,
                                    dA_csrPtr, dA_cols, dA_values,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ONE, 
                                    CUDA_R_32F) );
    CHECK_CUSPARSE( cusparseCreateCsr(&matB, hB_mat_num_rows, hB_mat_num_cols, hB_mat_nnz,
                                    dB_csrPtr, dB_cols, dB_values,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ONE, 
                                    CUDA_R_32F) );
    CHECK_CUSPARSE( cusparseCreateCsr(&matC, hA_mat_num_rows, hB_mat_num_cols, 0,
                                    NULL, NULL, NULL,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ONE,
                                    CUDA_R_32F) );

    //==========================================================================
    // SpGEMM Computation
    //==========================================================================
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE( cusparseSpGEMM_createDescr(&spgemmDesc) )

    // ask bufferSize1 bytes for external memory
    CHECK_CUSPARSE(
        cusparseSpGEMMreuse_workEstimation(handle, opA, opB, matA, matB, matC,
                                        CUSPARSE_SPGEMM_ALG2,
                                        spgemmDesc, &bufferSize1, NULL)
    )
    CHECK_CUDA( cudaMalloc((void**) &dBuffer1, bufferSize1) )
    
    // inspect the matrices A and B to understand the memory requirement for
    // the next step
    CHECK_CUSPARSE(
        cusparseSpGEMMreuse_workEstimation(handle, opA, opB, matA, matB, matC,
                                        CUSPARSE_SPGEMM_ALG2,
                                        spgemmDesc, &bufferSize1, dBuffer1)
    )
    //--------------------------------------------------------------------------

    CHECK_CUSPARSE(
        cusparseSpGEMMreuse_nnz(handle, opA, opB, matA, matB,
                                matC, CUSPARSE_SPGEMM_ALG2, spgemmDesc,
                                &bufferSize2, NULL, &bufferSize3, NULL,
                                &bufferSize4, NULL)
    )
    CHECK_CUDA( cudaMalloc((void**) &dBuffer2, bufferSize2) )
    CHECK_CUDA( cudaMalloc((void**) &dBuffer3, bufferSize3) )
    CHECK_CUDA( cudaMalloc((void**) &dBuffer4, bufferSize4) )
    CHECK_CUSPARSE(
        cusparseSpGEMMreuse_nnz(handle, opA, opB, matA, matB,
                                matC, CUSPARSE_SPGEMM_ALG2, spgemmDesc,
                                &bufferSize2, dBuffer2, &bufferSize3, dBuffer3,
                                &bufferSize4, dBuffer4)
    )
    CHECK_CUDA( cudaFree(dBuffer1) )
    CHECK_CUDA( cudaFree(dBuffer2) )
    //--------------------------------------------------------------------------

    // get matrix C non-zero entries C_nnz1
    int64_t C_num_rows, C_num_cols, C_nnz;
    CHECK_CUSPARSE( cusparseSpMatGetSize(matC, &C_num_rows, &C_num_cols,
                                        &C_nnz) )
    // allocate matrix C
    hC_mat_num_rows = (int)C_num_rows;
    hC_mat_num_cols = (int)C_num_cols;
    hC_mat_nnz = (uint64_t)C_nnz;

    CHECK_CUDA( cudaMalloc((void**) &dC_rows, C_nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dC_cols, C_nnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dC_values,  C_nnz * sizeof(float)) )
    CHECK_CUDA( cudaMemset(dC_values, 0x0, C_nnz * sizeof(float)) )
    // fill dC_values if needed
    // update matC with the new pointers
    CHECK_CUSPARSE(
        cusparseCsrSetPointers(matC, dC_csrPtr, dC_cols, dC_values) )
    //--------------------------------------------------------------------------

    CHECK_CUSPARSE(
        cusparseSpGEMMreuse_copy(handle, opA, opB, matA, matB, matC,
                                CUSPARSE_SPGEMM_ALG2, spgemmDesc,
                                &bufferSize5, NULL)
    )
    CHECK_CUDA( cudaMalloc((void**) &dBuffer5, bufferSize5) )
    CHECK_CUSPARSE(
        cusparseSpGEMMreuse_copy(handle, opA, opB, matA, matB, matC,
                                CUSPARSE_SPGEMM_ALG2, spgemmDesc,
                                &bufferSize5, dBuffer5)
    )
    CHECK_CUDA( cudaFree(dBuffer3) )

    CHECK_CUDA(cudaDeviceSynchronize());
    auto work_stop = high_resolution_clock::now();
    work_est_ms = duration_cast<duration<double, std::milli>>(work_stop - work_start).count();

    //--------------------------------------------------------------------------
    // STAGE 3: Actual Computation (GPU)
    //--------------------------------------------------------------------------
    float temp_ms = 0.0f;
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUSPARSE(
        cusparseSpGEMMreuse_compute(handle, opA, opB, &alpha, matA, matB, &beta,
                                    matC, computeType, CUSPARSE_SPGEMM_ALG2,
                                    spgemmDesc)
    )
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&temp_ms, start, stop));
    float first_ms = temp_ms;

    // update dA_values, dB_values
    CHECK_CUDA( cudaMemcpy(dA_values, hA_mat_values,
                        hA_mat_nnz * sizeof(float), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB_values, hB_mat_values,
                        hB_mat_nnz * sizeof(float), cudaMemcpyHostToDevice) )

    // second run
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUSPARSE(
        cusparseSpGEMMreuse_compute(handle, opA, opB, &alpha, matA, matB, &beta,
                                    matC, computeType, CUSPARSE_SPGEMM_ALG2,
                                    spgemmDesc)
    )
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&temp_ms, start, stop));
    compute_ms = temp_ms;

    //--------------------------------------------------------------------------
    // STAGE 4: D2H Transfer & Output Prep
    //--------------------------------------------------------------------------
    // device result load
    int* hC_rows_tmp = (int*)malloc(hC_mat_nnz * sizeof(int));
    int* hC_columns_tmp = (int*)malloc(hC_mat_nnz * sizeof(int));
    float* hC_values_tmp = (float*)malloc(hC_mat_nnz * sizeof(float));

    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUSPARSE(
        cusparseXcsr2coo(handle, dC_csrPtr, hC_mat_nnz, hC_mat_num_rows, dC_rows,
                    CUSPARSE_INDEX_BASE_ONE)  )
    CHECK_CUDA( cudaMemcpy(hC_rows_tmp, dC_rows, hC_mat_nnz * sizeof(int),
                        cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hC_columns_tmp, dC_cols, hC_mat_nnz * sizeof(int),
                        cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hC_values_tmp, dC_values, hC_mat_nnz * sizeof(float),
                        cudaMemcpyDeviceToHost) )
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&d2h_ms, start, stop));

    // ==========================================================================
    // 輸出細分時間結果 (寫入檔案以供 Python 讀取)
    // ==========================================================================
    std::ofstream log(logFile);
    if (log.is_open()) {
        log << "H2D_Transfer," << h2d_ms << std::endl;
        log << "Work_Estimation," << work_est_ms << std::endl;
        log << "Actual_Compute," << compute_ms << std::endl;
        log << "D2H_Transfer," << d2h_ms << std::endl;
        log.close();
    }

    printCooToFile(outfile, hC_rows_tmp, hC_columns_tmp, hC_values_tmp, hC_mat_num_rows, hC_mat_num_cols, hC_mat_nnz);

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseSpGEMM_destroyDescr(spgemmDesc) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matB) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
    cudaDeviceSynchronize();
    CHECK_CUSPARSE( cusparseDestroy(handle) )

    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer4) )
    CHECK_CUDA( cudaFree(dBuffer5) )
    CHECK_CUDA( cudaFree(dA_csrPtr) )
    CHECK_CUDA( cudaFree(dA_cols) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dB_csrPtr) )
    CHECK_CUDA( cudaFree(dB_cols) )
    CHECK_CUDA( cudaFree(dB_values) )
    CHECK_CUDA( cudaFree(dC_csrPtr) )
    CHECK_CUDA( cudaFree(dC_cols) )
    CHECK_CUDA( cudaFree(dC_values) )
    free(hC_rows_tmp);
    free(hC_columns_tmp);
    free(hC_values_tmp);
    delete[] hA_mat_rows;
    delete[] hA_mat_cols;
    delete[] hA_mat_values;
    delete[] hB_mat_rows;
    delete[] hB_mat_cols;
    delete[] hB_mat_values;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    // auto stop = high_resolution_clock::now();
    // auto end2end = duration_cast<nanoseconds>(stop - start);
    // std::cout << "END_TO_END_LATENCY = " << end2end.count() << std::endl;

    return EXIT_SUCCESS;
}

int compute(std::string matrix1File, std::string matrix2File, std::string outfile, std::string logFile) {
    // --- 1. 初始化計時器 ---
    Timing::reset();
    cudaEvent_t start, stop;
    float h2d_ms = 0, work_est_ms = 0, compute_ms = 0, d2h_ms = 0;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Host Matrix 定義
    int *hA_mat_rows, *hB_mat_rows, *hA_mat_cols, *hB_mat_cols;
    float *hA_mat_values, *hB_mat_values;
    int hA_mat_num_rows, hB_mat_num_rows, hA_mat_num_cols, hB_mat_num_cols;
    uint64_t hA_mat_nnz, hB_mat_nnz, hC_mat_nnz;

    // 讀取檔案
    cooFromFile(matrix1File, &hA_mat_rows, &hA_mat_cols, &hA_mat_values, &hA_mat_num_rows, &hA_mat_num_cols, &hA_mat_nnz);
    cooFromFile(matrix2File, &hB_mat_rows, &hB_mat_cols, &hB_mat_values, &hB_mat_num_rows, &hB_mat_num_cols, &hB_mat_nnz);

    Timing::startTiming("computation time no io");

    float alpha = 1.0f, beta = 0.0f;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType = CUDA_R_32F;

    // Device 指標宣告
    int *dA_rows, *dA_cols, *dB_rows, *dB_cols, *dC_rows, *dC_cols, *dA_csrPtr, *dB_csrPtr, *dC_csrPtr;
    float *dA_values, *dB_values, *dC_values;

    // ==========================================================================
    // STAGE 1: H2D Transfer & Allocation (Wall-clock)
    // ==========================================================================
    auto h2d_start = high_resolution_clock::now();

    CHECK_CUDA(cudaMalloc((void**)&dA_cols, hA_mat_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&dA_values, hA_mat_nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dA_csrPtr, (hA_mat_num_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&dB_cols, hB_mat_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&dB_values, hB_mat_nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dB_csrPtr, (hB_mat_num_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&dC_csrPtr, (hA_mat_num_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&dA_rows, hA_mat_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&dB_rows, hB_mat_nnz * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(dA_cols, hA_mat_cols, hA_mat_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_values, hA_mat_values, hA_mat_nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_cols, hB_mat_cols, hB_mat_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_values, hB_mat_values, hB_mat_nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_rows, hA_mat_rows, hA_mat_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_rows, hB_mat_rows, hB_mat_nnz * sizeof(int), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaDeviceSynchronize());
    auto h2d_stop = high_resolution_clock::now();
    h2d_ms = duration_cast<duration<double, std::milli>>(h2d_stop - h2d_start).count();

    // ==========================================================================
    // STAGE 2: Work Estimation & Analyze (Wall-clock)
    // ==========================================================================
    auto work_start = high_resolution_clock::now();
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // COO to CSR
    CHECK_CUSPARSE(cusparseXcoo2csr(handle, dA_rows, hA_mat_nnz, hA_mat_num_rows, dA_csrPtr, CUSPARSE_INDEX_BASE_ONE));
    CHECK_CUSPARSE(cusparseXcoo2csr(handle, dB_rows, hB_mat_nnz, hB_mat_num_rows, dB_csrPtr, CUSPARSE_INDEX_BASE_ONE));

    cusparseSpMatDescr_t matA, matB, matC;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, hA_mat_num_rows, hA_mat_num_cols, hA_mat_nnz, dA_csrPtr, dA_cols, dA_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ONE, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateCsr(&matB, hB_mat_num_rows, hB_mat_num_cols, hB_mat_nnz, dB_csrPtr, dB_cols, dB_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ONE, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateCsr(&matC, hA_mat_num_rows, hB_mat_num_cols, 0, NULL, NULL, NULL, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ONE, CUDA_R_32F));

    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc));

    size_t b1 = 0, b2 = 0, b3 = 0;
    void *db1 = NULL, *db2 = NULL, *db3 = NULL;
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType, CUSPARSE_SPGEMM_ALG3, spgemmDesc, &b1, NULL));
    CHECK_CUDA(cudaMalloc(&db1, b1));
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType, CUSPARSE_SPGEMM_ALG3, spgemmDesc, &b1, db1));
    CHECK_CUSPARSE(cusparseSpGEMM_estimateMemory(handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType, CUSPARSE_SPGEMM_ALG3, spgemmDesc, 0.01, &b3, NULL, NULL));
    CHECK_CUDA(cudaMalloc(&db3, b3));
    CHECK_CUSPARSE(cusparseSpGEMM_estimateMemory(handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType, CUSPARSE_SPGEMM_ALG3, spgemmDesc, 0.01, &b3, db3, &b2));
    CHECK_CUDA(cudaMalloc(&db2, b2));

    CHECK_CUDA(cudaDeviceSynchronize());
    auto work_stop = high_resolution_clock::now();
    work_est_ms = duration_cast<duration<double, std::milli>>(work_stop - work_start).count();

    // ==========================================================================
    // STAGE 3: Actual Computation
    // ==========================================================================
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType, CUSPARSE_SPGEMM_ALG3, spgemmDesc, &b2, db2));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&compute_ms, start, stop));

    int64_t c_rows, c_cols, c_nnz;
    CHECK_CUSPARSE(cusparseSpMatGetSize(matC, &c_rows, &c_cols, &c_nnz));
    hC_mat_nnz = (uint64_t)c_nnz;

    CHECK_CUDA(cudaMalloc((void**)&dC_rows, hC_mat_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&dC_cols, hC_mat_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&dC_values, hC_mat_nnz * sizeof(float)));
    CHECK_CUDA(cudaMemset(dC_values, 0x0, hC_mat_nnz * sizeof(float)));

    CHECK_CUSPARSE(cusparseCsrSetPointers(matC, dC_csrPtr, dC_cols, dC_values));

    // ==========================================================================
    // STAGE 4: D2H Transfer
    // ==========================================================================
    int* hC_rows_res = (int*)malloc(hC_mat_nnz * sizeof(int));
    int* hC_cols_res = (int*)malloc(hC_mat_nnz * sizeof(int));
    float* hC_vals_res = (float*)malloc(hC_mat_nnz * sizeof(float));

    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUSPARSE(cusparseSpGEMM_copy(handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType, CUSPARSE_SPGEMM_ALG3, spgemmDesc));
    // 將 CSR 轉回 COO 以便輸出 Row Indices
    CHECK_CUSPARSE(cusparseXcsr2coo(handle, dC_csrPtr, hC_mat_nnz, (int)c_rows, dC_rows, CUSPARSE_INDEX_BASE_ONE));

    CHECK_CUDA(cudaMemcpy(hC_rows_res, dC_rows, hC_mat_nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_cols_res, dC_cols, hC_mat_nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_vals_res, dC_values, hC_mat_nnz * sizeof(float), cudaMemcpyDeviceToHost));
    
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&d2h_ms, start, stop));

    // ==========================================================================
    // 輸出細分時間結果 (寫入檔案以供 Python 讀取)
    // ==========================================================================
    std::ofstream log(logFile);
    if (log.is_open()) {
        log << "H2D_Transfer," << h2d_ms << std::endl;
        log << "Work_Estimation," << work_est_ms << std::endl;
        log << "Actual_Compute," << compute_ms << std::endl;
        log << "D2H_Transfer," << d2h_ms << std::endl;
        log.close();
    }

    // ==========================================================================
    // 清理資源 (原本邏輯補完)
    // ==========================================================================
    printCooToFile(outfile, hC_rows_res, hC_cols_res, hC_vals_res, (int)c_rows, (int)c_cols, hC_mat_nnz);

    CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemmDesc));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroySpMat(matB));
    CHECK_CUSPARSE(cusparseDestroySpMat(matC));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    // 釋放 GPU 記憶體
    CHECK_CUDA(cudaFree(db1));
    CHECK_CUDA(cudaFree(db2));
    CHECK_CUDA(cudaFree(db3)); // 雖然前面可能 free 過，但這裡確保安全
    CHECK_CUDA(cudaFree(dA_csrPtr));
    CHECK_CUDA(cudaFree(dA_cols));
    CHECK_CUDA(cudaFree(dA_values));
    CHECK_CUDA(cudaFree(dA_rows));
    CHECK_CUDA(cudaFree(dB_csrPtr));
    CHECK_CUDA(cudaFree(dB_cols));
    CHECK_CUDA(cudaFree(dB_values));
    CHECK_CUDA(cudaFree(dB_rows));
    CHECK_CUDA(cudaFree(dC_csrPtr));
    CHECK_CUDA(cudaFree(dC_rows));
    CHECK_CUDA(cudaFree(dC_cols));
    CHECK_CUDA(cudaFree(dC_values));

    // 釋放 Host 記憶體
    free(hC_rows_res);
    free(hC_cols_res);
    free(hC_vals_res);
    delete[] hA_mat_rows;
    delete[] hA_mat_cols;
    delete[] hA_mat_values;
    delete[] hB_mat_rows;
    delete[] hB_mat_cols;
    delete[] hB_mat_values;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    Timing::stopTiming(true);
    
    return EXIT_SUCCESS;
}


int main( int argc, char* argv[] )
{
    std::string      matrix1File( "" );
    std::string      matrix2File( "" );
    std::string      outfile( "" );
    std::string      logFile( "" );

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );
        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "--output" || arg == "-o" )
        {
            if( i < argc - 1 )
            {
                outfile = argv[++i];
            }
            else
            {
                printUsageAndExit( argv[0] );
            }
        }
        else if (arg == "--mat1" || arg == "-m1") {
            if (i < argc - 1) {
                matrix1File = argv[++i];
            } else {
                printUsageAndExit(argv[0]);
            }
        }
        else if (arg == "--mat2" || arg == "-m2") {
            if (i < argc - 1) {
                matrix2File = argv[++i];
            } else {
                printUsageAndExit(argv[0]);
            }
        }
        else if (arg == "--log" || arg == "-l") {
            if (i < argc - 1) {
                logFile = argv[++i];
            } else {
                printUsageAndExit(argv[0]);
            }
        }
        else
        {
            std::cerr << "Unknown option " << arg << "\n";
            printUsageAndExit( argv[0] );
        }
    }

    try{
        #ifdef REUSE
            if (reuseCompute(matrix1File, matrix2File, outfile, logFile) == EXIT_FAILURE) {
                throw std::runtime_error("reuseCompute() returned EXIT_FAILURE");
            }
            // std::cerr << "Running Reuse" << "\n";
        #else
            if (compute(matrix1File, matrix2File, outfile, logFile) == EXIT_FAILURE) {
                throw std::runtime_error("compute() returned EXIT_FAILURE");
            }
            // std::cerr << "Running Compute" << "\n"; 
        #endif
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
