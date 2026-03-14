//
#pragma once
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#ifndef __GPU_TIMER_H__
#define __GPU_TIMER_H__

#include <optix.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "RTNumericPrecision.hpp"

#ifndef __CUDA_ARCH__
struct GpuTimer
{
      cudaEvent_t start;
      cudaEvent_t stop;

      GpuTimer()
      {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
      }

      ~GpuTimer()
      {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
      }

      void Start()
      {
            cudaEventRecord(start, 0);
      }

      void Stop()
      {
            cudaEventRecord(stop, 0);
      }

      float Elapsed()
      {
            float elapsed;
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed, start, stop);
            return elapsed;
      }
};
#endif
#endif  /* __GPU_TIMER_H__ */

struct Params
{
    uchar4*                image;
    unsigned int           image_width;
    unsigned int           image_height;
    int                    origin_x;
    int                    origin_y;
    OptixTraversableHandle handle;
    int                    mode;
};


struct RayGenData
{
    float3 cam_eye;
    float3 camera_u, camera_v, camera_w;
};


struct MissData
{
    float r, g, b;
};


struct HitGroupData
{
    // No data needed
};


template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct RayData {
    int*            rows;
    int*            cols;
    bqsim_rt::Complex* values;
    uint64_t        size;
};

struct SphereData {
    OptixAabb*      aabbs;
    bqsim_rt::Complex* sphereColor;
    bqsim_rt::Complex* rayValues;
    bqsim_rt::Complex* result;
    int              resultNumRow;
    int              resultNumCol;
    uint64_t         matrix1size;
    uint64_t         matrix2size;
    int*             rayRows;
    int*             rayCols;
    int*             rayCounts;
    int*             rowCounts;
    int*             rayOffsets;
    int*             rayWritePos;
    int*             outRows;
    int*             outCols;
    bqsim_rt::Complex* outVals;
    uint64_t         outCapacity;
    int              mode;
};

struct optixState {
    int                         width                    = 0;
    int                         height                   = 0;
    int*                        d_ray_rows                = nullptr;
    int*                        d_ray_cols                = nullptr;
    bqsim_rt::Complex*            d_ray_vals                = nullptr;
    uint64_t                    d_size                   = 0;       // ray_size
    uint64_t                    sphere_size              = 0;
    OptixAabb*                  aabbs                    = nullptr;
    uint64_t                    aabb_size                = 0;
    float3*                     spherePoints             = nullptr;
    float*                      sphereRadius             = nullptr;
    bqsim_rt::Complex*            sphereValues             = nullptr;
    bqsim_rt::Complex*            d_result                 = nullptr;
    uint64_t                    d_result_buf_size        = 0;
    int2                        m_result_dim;
    int*                        d_ray_counts             = nullptr;
    int*                        d_row_counts             = nullptr;
    int*                        d_ray_offsets            = nullptr;
    int*                        d_ray_write_pos          = nullptr;
    int*                        d_out_rows               = nullptr;
    int*                        d_out_cols               = nullptr;
    bqsim_rt::Complex*            d_out_vals               = nullptr;
    uint64_t                    out_capacity             = 0;
    int                         rt_mode                  = 0;

    CUdeviceptr                 devicePoints                ;
    CUdeviceptr                 deviceRadius                ;

    OptixDeviceContext          context                  = 0;

    OptixTraversableHandle      gas_handle               = 0;
    CUdeviceptr                 d_gas_output_buffer      = 0;

    OptixPipelineCompileOptions pipeline_compile_options = {};
    OptixPipelineLinkOptions    pipeline_link_options    = {};
    OptixModule                 module                   = 0;
    OptixModule                 sphere_module            = 0;
    OptixPipeline               pipeline                 = 0;
    OptixPipeline               pipeline_2               = 0;

    OptixProgramGroup           raygen_prog_group        = 0;
    OptixProgramGroup           miss_prog_group          = 0;
    OptixProgramGroup           hit_prog_group           = 0;

    Params                      params                   = {};
    Params                      params_translated        = {};
    OptixShaderBindingTable     sbt                      = {};
};
