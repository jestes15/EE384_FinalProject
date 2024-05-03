//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: diff_unwrap_phase.cu
//
// GPU Coder version                    : 23.2
// CUDA/C/C++ source code generated on  : 03-May-2024 13:01:26
//

// Include Files
#include "MWCudaDimUtility.hpp"
#include "MWLaunchParametersUtilities.hpp"
#include "diff_unwrap_phase.h"
#include "diff_unwrap_phase_data.h"
#include "diff_unwrap_phase_emxutil.h"
#include "diff_unwrap_phase_initialize.h"
#include "diff_unwrap_phase_types.h"
#include "rt_nonfinite.h"
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>

// Function Declarations
static __global__ void diff_unwrap_phase_kernel1(const emxArray_creal32_T yq, const int nx, emxArray_real32_T b_y);

static __global__ void diff_unwrap_phase_kernel2(const emxArray_real32_T b_y, const int vstride, const int i1,
                                                 const int vlen, emxArray_real32_T vwork);

static void gpuEmxEnsureCapacity_real32_T(const emxArray_real32_T *cpu, emxArray_real32_T *gpu);

static void gpuEmxFree_creal32_T(emxArray_creal32_T *gpu);

static void gpuEmxFree_real32_T(emxArray_real32_T *gpu);

static void gpuEmxMemcpyCpuToGpu_creal32_T(emxArray_creal32_T *gpu, const emxArray_creal32_T *cpu);

static void gpuEmxMemcpyCpuToGpu_real32_T(emxArray_real32_T *gpu, const emxArray_real32_T *cpu);

static void gpuEmxMemcpyGpuToCpu_real32_T(emxArray_real32_T *cpu, emxArray_real32_T *gpu);

static void gpuEmxReset_creal32_T(emxArray_creal32_T *gpu);

static void gpuEmxReset_real32_T(emxArray_real32_T *gpu);

static float rt_remf_snf(float u0, float u1);

// Function Definitions
//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_creal32_T yq
//                const int nx
//                emxArray_real32_T b_y
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void diff_unwrap_phase_kernel1(const emxArray_creal32_T yq, const int nx,
                                                                            emxArray_real32_T b_y)
{
    unsigned long long loopEnd;
    unsigned long long threadId;
    unsigned long long threadStride;
    threadId = static_cast<unsigned long long>(mwGetGlobalThreadIndexInXDimension());
    threadStride = mwGetTotalThreadsLaunched();
    loopEnd = static_cast<unsigned long long>(nx) - 1ULL;
    for (unsigned long long idx{threadId}; idx <= loopEnd; idx += threadStride)
    {
        int k;
        k = static_cast<int>(idx);
        b_y.data[k] = atan2f(yq.data[k].im, yq.data[k].re);
    }
}

//
// Arguments    : dim3 blockArg
//                dim3 gridArg
//                const emxArray_real32_T b_y
//                const int vstride
//                const int i1
//                const int vlen
//                emxArray_real32_T vwork
// Return Type  : void
//
static __global__ __launch_bounds__(1024, 1) void diff_unwrap_phase_kernel2(const emxArray_real32_T b_y,
                                                                            const int vstride, const int i1,
                                                                            const int vlen, emxArray_real32_T vwork)
{
    unsigned long long loopEnd;
    unsigned long long threadId;
    unsigned long long threadStride;
    threadId = static_cast<unsigned long long>(mwGetGlobalThreadIndexInXDimension());
    threadStride = mwGetTotalThreadsLaunched();
    loopEnd = static_cast<unsigned long long>(vlen);
    for (unsigned long long idx{threadId}; idx <= loopEnd; idx += threadStride)
    {
        int k;
        k = static_cast<int>(idx);
        vwork.data[k] = b_y.data[i1 + k * vstride];
    }
}

//
// Arguments    : const emxArray_real32_T *cpu
//                emxArray_real32_T *gpu
// Return Type  : void
//
static void gpuEmxEnsureCapacity_real32_T(const emxArray_real32_T *cpu, emxArray_real32_T *gpu)
{
    float *newData;
    if (gpu->data == 0)
    {
        newData = 0ULL;
        cudaMalloc(&newData, static_cast<unsigned int>(cpu->allocatedSize) * sizeof(float));
        gpu->numDimensions = cpu->numDimensions;
        gpu->size = static_cast<int *>(calloc(gpu->numDimensions, sizeof(int)));
        for (int i{0}; i < cpu->numDimensions; i++)
        {
            gpu->size[i] = cpu->size[i];
        }
        gpu->allocatedSize = cpu->allocatedSize;
        gpu->canFreeData = true;
        gpu->data = newData;
    }
    else
    {
        int actualSizeCpu;
        int actualSizeGpu;
        actualSizeCpu = 1;
        actualSizeGpu = 1;
        for (int i{0}; i < cpu->numDimensions; i++)
        {
            actualSizeGpu *= gpu->size[i];
            actualSizeCpu *= cpu->size[i];
            gpu->size[i] = cpu->size[i];
        }
        if (gpu->allocatedSize < actualSizeCpu)
        {
            newData = 0ULL;
            cudaMalloc(&newData, static_cast<unsigned int>(cpu->allocatedSize) * sizeof(float));
            cudaMemcpy(newData, gpu->data, static_cast<unsigned int>(actualSizeGpu) * sizeof(float),
                       cudaMemcpyDeviceToDevice);
            gpu->allocatedSize = cpu->allocatedSize;
            if (gpu->canFreeData)
            {
                cudaFree(gpu->data);
            }
            gpu->canFreeData = true;
            gpu->data = newData;
        }
    }
}

//
// Arguments    : emxArray_creal32_T *gpu
// Return Type  : void
//
static void gpuEmxFree_creal32_T(emxArray_creal32_T *gpu)
{
    if (gpu->data != (void *)4207599121ULL)
    {
        cudaFree(gpu->data);
    }
    std::free(gpu->size);
}

//
// Arguments    : emxArray_real32_T *gpu
// Return Type  : void
//
static void gpuEmxFree_real32_T(emxArray_real32_T *gpu)
{
    if (gpu->data != (void *)4207599121ULL)
    {
        cudaFree(gpu->data);
    }
    std::free(gpu->size);
}

//
// Arguments    : emxArray_creal32_T *gpu
//                const emxArray_creal32_T *cpu
// Return Type  : void
//
static void gpuEmxMemcpyCpuToGpu_creal32_T(emxArray_creal32_T *gpu, const emxArray_creal32_T *cpu)
{
    int actualSize;
    int i;
    if (gpu->numDimensions < cpu->numDimensions)
    {
        gpu->numDimensions = cpu->numDimensions;
        free(gpu->size);
        gpu->size = static_cast<int *>(calloc(gpu->numDimensions, sizeof(int)));
    }
    else
    {
        gpu->numDimensions = cpu->numDimensions;
    }
    actualSize = 1;
    for (i = 0; i < cpu->numDimensions; i++)
    {
        actualSize *= cpu->size[i];
        gpu->size[i] = cpu->size[i];
    }
    if (gpu->allocatedSize < actualSize)
    {
        if (gpu->canFreeData)
        {
            cudaFree(gpu->data);
        }
        i = cpu->allocatedSize;
        if (i < actualSize)
        {
            i = actualSize;
        }
        gpu->allocatedSize = i;
        gpu->canFreeData = true;
        cudaMalloc(&gpu->data, static_cast<unsigned int>(gpu->allocatedSize) * sizeof(creal32_T));
    }
    cudaMemcpy(gpu->data, cpu->data, static_cast<unsigned int>(actualSize) * sizeof(creal32_T), cudaMemcpyHostToDevice);
}

//
// Arguments    : emxArray_real32_T *gpu
//                const emxArray_real32_T *cpu
// Return Type  : void
//
static void gpuEmxMemcpyCpuToGpu_real32_T(emxArray_real32_T *gpu, const emxArray_real32_T *cpu)
{
    int actualSize;
    int i;
    if (gpu->numDimensions < cpu->numDimensions)
    {
        gpu->numDimensions = cpu->numDimensions;
        free(gpu->size);
        gpu->size = static_cast<int *>(calloc(gpu->numDimensions, sizeof(int)));
    }
    else
    {
        gpu->numDimensions = cpu->numDimensions;
    }
    actualSize = 1;
    for (i = 0; i < cpu->numDimensions; i++)
    {
        actualSize *= cpu->size[i];
        gpu->size[i] = cpu->size[i];
    }
    if (gpu->allocatedSize < actualSize)
    {
        if (gpu->canFreeData)
        {
            cudaFree(gpu->data);
        }
        i = cpu->allocatedSize;
        if (i < actualSize)
        {
            i = actualSize;
        }
        gpu->allocatedSize = i;
        gpu->canFreeData = true;
        cudaMalloc(&gpu->data, static_cast<unsigned int>(gpu->allocatedSize) * sizeof(float));
    }
    cudaMemcpy(gpu->data, cpu->data, static_cast<unsigned int>(actualSize) * sizeof(float), cudaMemcpyHostToDevice);
}

//
// Arguments    : emxArray_real32_T *cpu
//                emxArray_real32_T *gpu
// Return Type  : void
//
static void gpuEmxMemcpyGpuToCpu_real32_T(emxArray_real32_T *cpu, emxArray_real32_T *gpu)
{
    int actualSize;
    actualSize = 1;
    for (int i{0}; i < cpu->numDimensions; i++)
    {
        actualSize *= cpu->size[i];
    }
    cudaMemcpy(cpu->data, gpu->data, static_cast<unsigned int>(actualSize) * sizeof(float), cudaMemcpyDeviceToHost);
}

//
// Arguments    : emxArray_creal32_T *gpu
// Return Type  : void
//
static void gpuEmxReset_creal32_T(emxArray_creal32_T *gpu)
{
    std::memset(gpu, 0, sizeof(emxArray_creal32_T));
}

//
// Arguments    : emxArray_real32_T *gpu
// Return Type  : void
//
static void gpuEmxReset_real32_T(emxArray_real32_T *gpu)
{
    std::memset(gpu, 0, sizeof(emxArray_real32_T));
}

//
// Arguments    : float u0
//                float u1
// Return Type  : float
//
static float rt_remf_snf(float u0, float u1)
{
    float b_y;
    if (std::isnan(u0) || std::isnan(u1) || std::isinf(u0))
    {
        b_y = rtNaNF;
    }
    else if (std::isinf(u1))
    {
        b_y = u0;
    }
    else if ((u1 != 0.0F) && (u1 != std::trunc(u1)))
    {
        float q;
        q = std::abs(u0 / u1);
        if (!(std::abs(q - std::floor(q + 0.5F)) > FLT_EPSILON * q))
        {
            b_y = 0.0F * u0;
        }
        else
        {
            b_y = std::fmod(u0, u1);
        }
    }
    else
    {
        b_y = std::fmod(u0, u1);
    }
    return b_y;
}

//
// codegen
//
// Arguments    : const emxArray_creal32_T *cpu_yq
//                emxArray_real32_T *b_z
// Return Type  : void
//
void diff_unwrap_phase(const emxArray_creal32_T *cpu_yq, emxArray_real32_T *b_z)
{
    dim3 block;
    dim3 grid;
    emxArray_creal32_T gpu_yq;
    emxArray_real32_T gpu_vwork;
    emxArray_real32_T gpu_y;
    emxArray_real32_T *cpu_vwork;
    emxArray_real32_T *cpu_y;
    float dp_corr;
    float pkm1;
    int i1;
    int i2;
    int m;
    int nx;
    int vstride;
    boolean_T validLaunchParams;
    boolean_T vwork_outdatedOnCpu;
    boolean_T vwork_outdatedOnGpu;
    boolean_T y_outdatedOnCpu;
    boolean_T y_outdatedOnGpu;
    if (!isInitialized_diff_unwrap_phase)
    {
        diff_unwrap_phase_initialize();
    }
    gpuEmxReset_real32_T(&gpu_vwork);
    gpuEmxReset_real32_T(&gpu_y);
    gpuEmxReset_creal32_T(&gpu_yq);
    vwork_outdatedOnCpu = false;
    vwork_outdatedOnGpu = false;
    y_outdatedOnCpu = false;
    y_outdatedOnGpu = false;
    nx = cpu_yq->size[0];
    emxInit_real32_T(&cpu_y, 1);
    i1 = cpu_y->size[0];
    cpu_y->size[0] = cpu_yq->size[0];
    emxEnsureCapacity_real32_T(cpu_y, i1);
    gpuEmxEnsureCapacity_real32_T(cpu_y, &gpu_y);
    validLaunchParams = mwGetLaunchParameters1D(static_cast<double>(nx), &grid, &block, 1024U, 65535U);
    if (validLaunchParams)
    {
        gpuEmxMemcpyCpuToGpu_creal32_T(&gpu_yq, cpu_yq);
        diff_unwrap_phase_kernel1<<<grid, block>>>(gpu_yq, nx, gpu_y);
        y_outdatedOnCpu = true;
    }
    nx = 0;
    if (cpu_y->size[0] != 1)
    {
        nx = -1;
    }
    if (nx + 2 <= 1)
    {
        i2 = cpu_y->size[0] - 1;
    }
    else
    {
        i2 = 0;
    }
    emxInit_real32_T(&cpu_vwork, 1);
    i1 = cpu_vwork->size[0];
    cpu_vwork->size[0] = i2 + 1;
    emxEnsureCapacity_real32_T(cpu_vwork, i1);
    gpuEmxEnsureCapacity_real32_T(cpu_vwork, &gpu_vwork);
    vstride = 1;
    for (m = 0; m <= nx; m++)
    {
        vstride *= cpu_y->size[0];
    }
    i1 = -1;
    for (nx = 0; nx < vstride; nx++)
    {
        float cumsum_dp_corr;
        unsigned int k;
        boolean_T exitg1;
        i1++;
        validLaunchParams = mwGetLaunchParameters1D(static_cast<double>(i2 + 1LL), &grid, &block, 1024U, 65535U);
        if (validLaunchParams)
        {
            if (y_outdatedOnGpu)
            {
                gpuEmxMemcpyCpuToGpu_real32_T(&gpu_y, cpu_y);
            }
            y_outdatedOnGpu = false;
            if (vwork_outdatedOnGpu)
            {
                gpuEmxMemcpyCpuToGpu_real32_T(&gpu_vwork, cpu_vwork);
            }
            diff_unwrap_phase_kernel2<<<grid, block>>>(gpu_y, vstride, i1, i2, gpu_vwork);
            vwork_outdatedOnGpu = false;
            vwork_outdatedOnCpu = true;
        }
        m = cpu_vwork->size[0];
        cumsum_dp_corr = 0.0F;
        k = 1U;
        exitg1 = false;
        while ((!exitg1) && (static_cast<int>(k) < m))
        {
            if (vwork_outdatedOnCpu)
            {
                gpuEmxMemcpyGpuToCpu_real32_T(cpu_vwork, &gpu_vwork);
            }
            vwork_outdatedOnCpu = false;
            if (std::isinf(cpu_vwork->data[static_cast<int>(k) - 1]) ||
                std::isnan(cpu_vwork->data[static_cast<int>(k) - 1]))
            {
                k = static_cast<unsigned int>(static_cast<int>(k) + 1);
            }
            else
            {
                exitg1 = true;
            }
        }
        if (static_cast<int>(k) < cpu_vwork->size[0])
        {
            if (vwork_outdatedOnCpu)
            {
                gpuEmxMemcpyGpuToCpu_real32_T(cpu_vwork, &gpu_vwork);
            }
            vwork_outdatedOnCpu = false;
            pkm1 = cpu_vwork->data[static_cast<int>(k) - 1];
            int exitg2;
            do
            {
                exitg2 = 0;
                k++;
                while ((k <= static_cast<unsigned int>(m)) && (std::isinf(cpu_vwork->data[static_cast<int>(k) - 1]) ||
                                                               std::isnan(cpu_vwork->data[static_cast<int>(k) - 1])))
                {
                    k++;
                }
                if (k > static_cast<unsigned int>(m))
                {
                    exitg2 = 1;
                }
                else
                {
                    pkm1 = cpu_vwork->data[static_cast<int>(k) - 1] - pkm1;
                    dp_corr = pkm1 / 6.28318548F;
                    if (std::abs(rt_remf_snf(dp_corr, 1.0F)) <= 0.5F)
                    {
                        dp_corr = std::trunc(dp_corr);
                    }
                    else
                    {
                        dp_corr = std::round(dp_corr);
                    }
                    if (std::abs(pkm1) >= 3.14159274F)
                    {
                        cumsum_dp_corr += dp_corr;
                    }
                    pkm1 = cpu_vwork->data[static_cast<int>(k) - 1];
                    cpu_vwork->data[static_cast<int>(k) - 1] -= 6.28318548F * cumsum_dp_corr;
                    vwork_outdatedOnGpu = true;
                }
            } while (exitg2 == 0);
        }
        for (m = 0; m <= i2; m++)
        {
            if (y_outdatedOnCpu)
            {
                gpuEmxMemcpyGpuToCpu_real32_T(cpu_y, &gpu_y);
            }
            if (vwork_outdatedOnCpu)
            {
                gpuEmxMemcpyGpuToCpu_real32_T(cpu_vwork, &gpu_vwork);
            }
            vwork_outdatedOnCpu = false;
            cpu_y->data[i1 + m * vstride] = cpu_vwork->data[m];
            y_outdatedOnCpu = false;
            y_outdatedOnGpu = true;
        }
    }
    emxFree_real32_T(&cpu_vwork);
    i2 = cpu_y->size[0];
    if (cpu_y->size[0] == 0)
    {
        b_z->size[0] = 0;
    }
    else
    {
        nx = cpu_y->size[0] - 1;
        if (nx > 1)
        {
            nx = 1;
        }
        if (nx < 1)
        {
            b_z->size[0] = 0;
        }
        else
        {
            i1 = b_z->size[0];
            b_z->size[0] = cpu_y->size[0] - 1;
            emxEnsureCapacity_real32_T(b_z, i1);
            if (cpu_y->size[0] - 1 != 0)
            {
                float work_data[1];
                if (y_outdatedOnCpu)
                {
                    gpuEmxMemcpyGpuToCpu_real32_T(cpu_y, &gpu_y);
                }
                work_data[0] = cpu_y->data[0];
                for (m = 0; m <= i2 - 2; m++)
                {
                    pkm1 = cpu_y->data[m + 1];
                    dp_corr = pkm1;
                    pkm1 -= work_data[0];
                    work_data[0] = dp_corr;
                    b_z->data[m] = pkm1;
                }
            }
        }
    }
    emxFree_real32_T(&cpu_y);
    gpuEmxFree_creal32_T(&gpu_yq);
    gpuEmxFree_real32_T(&gpu_y);
    gpuEmxFree_real32_T(&gpu_vwork);
}

//
// File trailer for diff_unwrap_phase.cu
//
// [EOF]
//