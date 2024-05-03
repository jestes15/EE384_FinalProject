//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: diff_unwrap_phase_types.h
//
// GPU Coder version                    : 23.2
// CUDA/C/C++ source code generated on  : 03-May-2024 13:01:26
//

#ifndef DIFF_UNWRAP_PHASE_TYPES_H
#define DIFF_UNWRAP_PHASE_TYPES_H

// Include Files
#include "rtwtypes.h"

// Type Definitions
struct emxArray_creal32_T {
  creal32_T *data;
  int *size;
  int allocatedSize;
  int numDimensions;
  boolean_T canFreeData;
};

struct emxArray_real32_T {
  float *data;
  int *size;
  int allocatedSize;
  int numDimensions;
  boolean_T canFreeData;
};

#endif
//
// File trailer for diff_unwrap_phase_types.h
//
// [EOF]
//
