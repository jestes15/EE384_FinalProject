//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: diff_unwrap_phase_emxAPI.h
//
// GPU Coder version                    : 23.2
// CUDA/C/C++ source code generated on  : 03-May-2024 13:01:26
//

#ifndef DIFF_UNWRAP_PHASE_EMXAPI_H
#define DIFF_UNWRAP_PHASE_EMXAPI_H

// Include Files
#include "diff_unwrap_phase_types.h"
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
extern emxArray_creal32_T *emxCreateND_creal32_T(int b_numDimensions,
                                                 const int *b_size);

extern emxArray_real32_T *emxCreateND_real32_T(int b_numDimensions,
                                               const int *b_size);

extern emxArray_creal32_T *emxCreateWrapperND_creal32_T(creal32_T *b_data,
                                                        int b_numDimensions,
                                                        const int *b_size);

extern emxArray_real32_T *emxCreateWrapperND_real32_T(float *b_data,
                                                      int b_numDimensions,
                                                      const int *b_size);

extern emxArray_creal32_T *emxCreateWrapper_creal32_T(creal32_T *b_data,
                                                      int rows, int cols);

extern emxArray_real32_T *emxCreateWrapper_real32_T(float *b_data, int rows,
                                                    int cols);

extern emxArray_creal32_T *emxCreate_creal32_T(int rows, int cols);

extern emxArray_real32_T *emxCreate_real32_T(int rows, int cols);

extern void emxDestroyArray_creal32_T(emxArray_creal32_T *emxArray);

extern void emxDestroyArray_real32_T(emxArray_real32_T *emxArray);

extern void emxInitArray_creal32_T(emxArray_creal32_T **pEmxArray,
                                   int b_numDimensions);

extern void emxInitArray_real32_T(emxArray_real32_T **pEmxArray,
                                  int b_numDimensions);

#endif
//
// File trailer for diff_unwrap_phase_emxAPI.h
//
// [EOF]
//
