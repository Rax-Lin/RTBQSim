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

static __forceinline__ __device__ void atomicAddComplex(cuDoubleComplex* addr, cuDoubleComplex val)
{
    atomicAdd(&(addr->x), val.x);
    atomicAdd(&(addr->y), val.y);
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

    int row = ray_data->rows[ray_idx];
    int col = ray_data->cols[ray_idx];
    const cuDoubleComplex v = ray_data->values[ray_idx];
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
extern "C" __global__ void __intersection__is()
{
    int idx = 1;
}
#else
extern "C" __global__ void __intersection__is()
{
    const SphereData* hit_data = reinterpret_cast<SphereData*>(optixGetSbtDataPointer());
    const unsigned int prim_idx = optixGetPrimitiveIndex();
    const OptixAabb aabb = hit_data->aabbs[prim_idx];

    const float3 ray_o = optixGetObjectRayOrigin();
    const float3 ray_d = optixGetObjectRayDirection();

    float tmin = optixGetRayTmin();
    float tmax = optixGetRayTmax();

    if (ray_d.x == 0.0f && (ray_o.x < aabb.minX || ray_o.x > aabb.maxX)) {
        return;
    }
    if (ray_d.y == 0.0f && (ray_o.y < aabb.minY || ray_o.y > aabb.maxY)) {
        return;
    }

    if (ray_d.x != 0.0f) {
        const float inv = 1.0f / ray_d.x;
        const float t0 = (aabb.minX - ray_o.x) * inv;
        const float t1 = (aabb.maxX - ray_o.x) * inv;
        tmin = fmaxf(tmin, fminf(t0, t1));
        tmax = fminf(tmax, fmaxf(t0, t1));
    }
    if (ray_d.y != 0.0f) {
        const float inv = 1.0f / ray_d.y;
        const float t0 = (aabb.minY - ray_o.y) * inv;
        const float t1 = (aabb.maxY - ray_o.y) * inv;
        tmin = fmaxf(tmin, fminf(t0, t1));
        tmax = fminf(tmax, fmaxf(t0, t1));
    }
    if (ray_d.z != 0.0f) {
        const float inv = 1.0f / ray_d.z;
        const float t0 = (aabb.minZ - ray_o.z) * inv;
        const float t1 = (aabb.maxZ - ray_o.z) * inv;
        tmin = fmaxf(tmin, fminf(t0, t1));
        tmax = fminf(tmax, fmaxf(t0, t1));
    }

    if (tmax >= tmin) {
        optixReportIntersection(tmin, 0);
    }
}
#endif

#if defined(NOTHING)
extern "C" __global__ void __anyhit__ch()
{
    int idx = 1;
}
#else
extern "C" __global__ void __anyhit__ch()
{
    const unsigned int ray_idx = optixGetPayload_0();
    const unsigned int sphere_idx = optixGetPrimitiveIndex();

    SphereData* hit_data = reinterpret_cast<SphereData*>(optixGetSbtDataPointer());
    if (hit_data->mode == 0) {
        const int row = hit_data->rayRows[ray_idx];
        atomicAdd(&hit_data->rayCounts[ray_idx], 1);
        atomicAdd(&hit_data->rowCounts[row], 1);
        optixIgnoreIntersection();
        return;
    }
    if (hit_data->mode == 1) {
        const int slot = atomicAdd(&hit_data->rayWritePos[ray_idx], 1);
        const int out_idx = hit_data->rayOffsets[ray_idx] + slot;
        if (static_cast<uint64_t>(out_idx) >= hit_data->outCapacity) {
            optixIgnoreIntersection();
            return;
        }
        const int row = hit_data->rayRows[ray_idx];
        const OptixAabb aabb = hit_data->aabbs[sphere_idx];
        const int col = static_cast<int>(aabb.minX);
        hit_data->outRows[out_idx] = row;
        hit_data->outCols[out_idx] = col;
        const cuDoubleComplex a = hit_data->rayValues[ray_idx];
        const cuDoubleComplex b = hit_data->sphereColor[sphere_idx];
        hit_data->outVals[out_idx] = cuCmul(a, b);
        optixIgnoreIntersection();
        return;
    }
    cuDoubleComplex a = hit_data->rayValues[ray_idx];
    cuDoubleComplex b = hit_data->sphereColor[sphere_idx];
    cuDoubleComplex prod = cuCmul(a, b);
    atomicAddComplex(&hit_data->result[ray_idx], prod);
    optixIgnoreIntersection();
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
