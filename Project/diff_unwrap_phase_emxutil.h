//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: diff_unwrap_phase_emxutil.h
//
// GPU Coder version                    : 23.2
// CUDA/C/C++ source code generated on  : 03-May-2024 13:01:26
//

#ifndef DIFF_UNWRAP_PHASE_EMXUTIL_H
#define DIFF_UNWRAP_PHASE_EMXUTIL_H

// Include Files
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>

struct emxArray_real32_T;

struct emxArray_creal32_T;

// Function Declarations
extern void emxEnsureCapacity_real32_T(emxArray_real32_T *emxArray,
                                       int oldNumel);

extern void emxFree_creal32_T(emxArray_creal32_T **pEmxArray);

extern void emxFree_real32_T(emxArray_real32_T **pEmxArray);

extern void emxInit_creal32_T(emxArray_creal32_T **pEmxArray,
                              int b_numDimensions);

extern void emxInit_real32_T(emxArray_real32_T **pEmxArray,
                             int b_numDimensions);

#endif
//
// File trailer for diff_unwrap_phase_emxutil.h
//
// [EOF]
//
