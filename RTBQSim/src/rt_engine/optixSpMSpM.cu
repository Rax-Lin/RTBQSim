//
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

#include <optix.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdint>
#include "optixSpMSpM.h"
// #include <cuda/helpers.h>

#include <sutil/vec_math.h>
#include <nvtx3/nvToolsExt.h>
// #include <stdgpu/unordered_map.cuh>
// #include <stdgpu/iterator.h>        // device_begin, device_end
// #include <cuco/static_map.cuh>

extern "C" {
__constant__ Params params;
}


static __forceinline__ __device__ void trace(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax,
        unsigned int           payload0,
        unsigned int           payload1)
{
    unsigned int p0 = payload0;
    unsigned int p1 = payload1;
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset
            0,                   // SBT stride
            0,                   // missSBTIndex
            p0, p1 );
}

static __forceinline__ __device__ bool controlsSatisfied(const qc::GatePrimitive& gate, int row) {
    for (int c = 0; c < gate.control_count; ++c) {
        const int qb = gate.controls[c];
        if (((row >> qb) & 1) == 0) {
            return false;
        }
    }
    return true;
}

static __forceinline__ __device__ bqsim_rt::Complex gateValueAt(const qc::GatePrimitive& gate, int row, int col) {
    if (gate.target_count != 1 || gate.matrix_dim != 2) {
        return bqsim_rt::make_complex(0.0, 0.0);
    }
    const int target = gate.targets[0];
    const bool controls_ok = controlsSatisfied(gate, row);
    if (!controls_ok) {
        return (col == row) ? bqsim_rt::make_complex(1.0, 0.0) : bqsim_rt::make_complex(0.0, 0.0);
    }
    const int bit = (row >> target) & 1;
    const int base = row & ~(1 << target);
    const int col0 = base;
    const int col1 = base | (1 << target);
    const int m = bit * 2;
    if (col == col0) {
        const bqsim_rt::MatrixElem a0 = gate.matrix[m];
        return bqsim_rt::make_complex(a0.x, a0.y);
    }
    if (col == col1) {
        const bqsim_rt::MatrixElem a1 = gate.matrix[m + 1];
        return bqsim_rt::make_complex(a1.x, a1.y);
    }
    return bqsim_rt::make_complex(0.0, 0.0);
}

// static "C"  void checkSphere()
// { (is false)
//     {
//         return false;
//     }
    
//     return true;
    
// }

#if defined(NOTHING)
extern "C" __global__ void __raygen__rg()
{
    int dx = 1;
    // printf("NOTHING%d\n", dx);
}
#else
extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    unsigned int ray_idx = idx.x;
    RayData* ray_data = reinterpret_cast<RayData*>(optixGetSbtDataPointer());
    if (ray_idx >= ray_data->size) {
        return;
    }

    if (ray_data->proceduralMode != 0) {
        const qc::GatePrimitive& gate = ray_data->gate;
        if (gate.target_count != 1 || gate.matrix_dim != 2) {
            return;
        }
        int row = 0;
        int lane = 0;
        if (ray_data->proceduralMode == 2) {
            row = static_cast<int>(ray_idx);
            lane = -1;
        } else {
            row = static_cast<int>(ray_idx >> 1);
            lane = static_cast<int>(ray_idx & 1u);
        }
        if (row >= ray_data->nDim) {
            return;
        }
        const int target = gate.targets[0];
        const bool controls_ok = controlsSatisfied(gate, row);
        const int bit = (row >> target) & 1;
        const int base = row & ~(1 << target);
        int col = row;
        bqsim_rt::Complex v = bqsim_rt::make_complex(0.0, 0.0);
        if (ray_data->proceduralMode == 2) {
            if (!controls_ok) {
                col = row;
                v = bqsim_rt::make_complex(1.0, 0.0);
            } else {
                const int m = bit * 2;
                const bqsim_rt::MatrixElem a0 = gate.matrix[m];
                const bqsim_rt::MatrixElem a1 = gate.matrix[m + 1];
                const bool nz0 = (a0.x != 0.0 || a0.y != 0.0);
                if (nz0) {
                    col = base;
                    v = bqsim_rt::make_complex(a0.x, a0.y);
                } else {
                    col = base | (1 << target);
                    v = bqsim_rt::make_complex(a1.x, a1.y);
                }
            }
        } else {
            const int col0 = controls_ok ? base : row;
            const int col1 = controls_ok ? (base | (1 << target)) : row;
            col = (lane == 0) ? col0 : col1;
            if (!controls_ok) {
                v = (lane == 0) ? bqsim_rt::make_complex(1.0, 0.0) : bqsim_rt::make_complex(0.0, 0.0);
            } else {
                const int m = bit * 2;
                const bqsim_rt::MatrixElem a = gate.matrix[m + lane];
                v = bqsim_rt::make_complex(a.x, a.y);
            }
        }
        if (v.x == 0.0 && v.y == 0.0) {
            return;
        }
        float3 origin = make_float3(-1.0f,
                                    static_cast<float>(col) + 0.5f,
                                    0.5f);
        float3 direction = make_float3(1.0f, 0.0f, 0.0f);
        trace(params.handle,
              origin,
              direction,
              0.0f,
              1e16f,
              static_cast<unsigned int>(row),
              static_cast<unsigned int>(col));
        return;
    }

    int row = ray_data->rows[ray_idx];
    int col = ray_data->cols[ray_idx];
    const bqsim_rt::Complex v = ray_data->values[ray_idx];
    if (v.x == 0.0 && v.y == 0.0) {
        return;
    }
    float3 origin = make_float3(-1.0f,
                                static_cast<float>(col) + 0.5f,
                                0.5f);
    float3 direction = make_float3(1.0f, 0.0f, 0.0f);

    trace(params.handle,
          origin,
          direction,
          0.0f,
          1e16f,
          ray_idx,
          static_cast<unsigned int>(row));
}
#endif



extern "C" __global__ void __miss__ms()
{
    // MissData* rt_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    // float3    payload = getPayload();
    // setPayload( make_float3( rt_data->r, rt_data->g, rt_data->b ) );
}


#if defined(NOTHING)
extern "C" __global__ void __anyhit__ch()
{
    int idx = 1;
}
#else
extern "C" __global__ void __anyhit__ch()
{
    const unsigned int payload0 = optixGetPayload_0();
    const unsigned int payload1 = optixGetPayload_1();
    const unsigned int ray_idx = payload0;
    const unsigned int sphere_idx = optixGetPrimitiveIndex();
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    const unsigned int sbtGASIndex = optixGetSbtGASIndex();
    float4 sphere;
    optixGetSphereData(gas, sphere_idx, sbtGASIndex, 0.f, &sphere);

    SphereData* hit_data = reinterpret_cast<SphereData*>(optixGetSbtDataPointer());
    if (hit_data->mode == 1) {
        if (!hit_data->outCount) {
            optixIgnoreIntersection();
            return;
        }
        const int out_idx = atomicAdd(hit_data->outCount, 1);
        if (static_cast<uint64_t>(out_idx) >= hit_data->outCapacity) {
            optixIgnoreIntersection();
            return;
        }
        const bool procedural = hit_data->proceduralMode != 0;
        const int row = procedural ? static_cast<int>(payload0) : hit_data->rayRows[ray_idx];
        const int gate_col = procedural ? static_cast<int>(payload1) : hit_data->rayCols[ray_idx];
        const int col = static_cast<int>(sphere.x);
        hit_data->outRows[out_idx] = row;
        hit_data->outCols[out_idx] = col;
        const bqsim_rt::Complex a = procedural ? gateValueAt(hit_data->gate, row, gate_col)
                                               : hit_data->rayValues[ray_idx];
        if (a.x == 0.0 && a.y == 0.0) {
            optixIgnoreIntersection();
            return;
        }
        const bqsim_rt::Complex b = hit_data->sphereColor[sphere_idx];
        hit_data->outVals[out_idx] = bqsim_rt::cmul(a, b);
        optixIgnoreIntersection();
        return;
    }
    if (hit_data->mode == 0) {
        // Diagonal-gate RT path: update accumulated primitive values directly
        // while preserving topology (rows/cols) and primitive ordering.
        const bool procedural = hit_data->proceduralMode != 0;
        const int row = procedural ? static_cast<int>(payload0) : hit_data->rayRows[ray_idx];
        const int col = procedural ? static_cast<int>(payload1) : static_cast<int>(sphere.x);
        const bqsim_rt::Complex a = procedural ? gateValueAt(hit_data->gate, row, col)
                                               : hit_data->rayValues[ray_idx];
        if (a.x == 0.0 && a.y == 0.0) {
            optixIgnoreIntersection();
            return;
        }
        const bqsim_rt::Complex b = hit_data->sphereColor[sphere_idx];
        hit_data->result[sphere_idx] = bqsim_rt::cmul(a, b);
        optixIgnoreIntersection();
        return;
    } // not optimized
    // CSV formatted printf
    // Format: ray_idx.x,ray_idx.y,ray_idx.z,
    //         payload.x,payload.y,payload.z,
    //         sphere_idx,sphere.x,sphere.y,sphere.z,
    //         sphereData,resultFloat,
    // Meaning: ray_idx, 0, 0, 
    //         mat_1_x (result_row), mat_1_y, mat_1_value,
    //         sphere_idx, mat_2_x, mat_2_y (result_col), 0,
    //         mat_2_value, mat_1 val * mat_2 val
    // printf("%d,%d,%d,%.0f,%.0f,%.3f,%d,%.3f,%.3f,%.3f,%.3f,%.3f\n",
    //            (int)ray_idx.x, (int)ray_idx.y, (int)ray_idx.z, 
    //            payload.x, payload.y, payload.z, 
    //            (int)sphere_idx, sphere.x, sphere.y, sphere.z, 
    //            sphereData, resultFloat);

    // printf("%d\n", (int)ray_idx.x );
    // printf("%.3f\n", payload.z * sphereData);
    // printf("Mat_1 [%.0f][%.0f](%.3f) * Mat_2 [%.0f][%.0f](%.3f) -> Mat_result[%.0f][%.0f](%.3f)\n", payload.x, payload.y, payload.z, sphere.x, sphere.y, sphereData, payload.x, sphere.y, resultFloat);
    // printf("Current Result[%.0f][%.0f],(%.3f)\n", payload.x, sphere.y, hit_data->result[idx]);
}
#endif
