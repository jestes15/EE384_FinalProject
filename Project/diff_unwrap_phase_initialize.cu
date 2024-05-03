//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: diff_unwrap_phase_initialize.cu
//
// GPU Coder version                    : 23.2
// CUDA/C/C++ source code generated on  : 03-May-2024 13:01:26
//

// Include Files
#include "diff_unwrap_phase_initialize.h"
#include "diff_unwrap_phase_data.h"
#include "rt_nonfinite.h"

// Function Definitions
//
// Arguments    : void
// Return Type  : void
//
void diff_unwrap_phase_initialize()
{
  cudaGetLastError();
  isInitialized_diff_unwrap_phase = true;
}

//
// File trailer for diff_unwrap_phase_initialize.cu
//
// [EOF]
//
