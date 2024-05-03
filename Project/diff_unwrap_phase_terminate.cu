//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: diff_unwrap_phase_terminate.cu
//
// GPU Coder version                    : 23.2
// CUDA/C/C++ source code generated on  : 03-May-2024 13:01:26
//

// Include Files
#include "diff_unwrap_phase_terminate.h"
#include "diff_unwrap_phase_data.h"
#include "rt_nonfinite.h"
#include "stdio.h"

// Function Definitions
//
// Arguments    : void
// Return Type  : void
//
void diff_unwrap_phase_terminate()
{
  cudaError_t errCode;
  errCode = cudaGetLastError();
  if (errCode != cudaSuccess) {
    fprintf(stderr, "ERR[%d] %s:%s\n", errCode, cudaGetErrorName(errCode),
            cudaGetErrorString(errCode));
    exit(errCode);
  }
  isInitialized_diff_unwrap_phase = false;
}

//
// File trailer for diff_unwrap_phase_terminate.cu
//
// [EOF]
//
