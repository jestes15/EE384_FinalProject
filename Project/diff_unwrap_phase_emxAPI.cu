//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: diff_unwrap_phase_emxAPI.cu
//
// GPU Coder version                    : 23.2
// CUDA/C/C++ source code generated on  : 03-May-2024 13:01:26
//

// Include Files
#include "diff_unwrap_phase_emxAPI.h"
#include "diff_unwrap_phase_emxutil.h"
#include "diff_unwrap_phase_types.h"
#include "rt_nonfinite.h"
#include <cstdlib>

// Function Definitions
//
// Arguments    : int b_numDimensions
//                const int *b_size
// Return Type  : emxArray_creal32_T *
//
emxArray_creal32_T *emxCreateND_creal32_T(int b_numDimensions,
                                          const int *b_size)
{
  emxArray_creal32_T *emx;
  int numEl;
  emxInit_creal32_T(&emx, b_numDimensions);
  numEl = 1;
  for (int i{0}; i < b_numDimensions; i++) {
    numEl *= b_size[i];
    emx->size[i] = b_size[i];
  }
  emx->data = static_cast<creal32_T *>(
      std::malloc(static_cast<unsigned int>(numEl) * sizeof(creal32_T)));
  emx->numDimensions = b_numDimensions;
  emx->allocatedSize = numEl;
  return emx;
}

//
// Arguments    : int b_numDimensions
//                const int *b_size
// Return Type  : emxArray_real32_T *
//
emxArray_real32_T *emxCreateND_real32_T(int b_numDimensions, const int *b_size)
{
  emxArray_real32_T *emx;
  int numEl;
  emxInit_real32_T(&emx, b_numDimensions);
  numEl = 1;
  for (int i{0}; i < b_numDimensions; i++) {
    numEl *= b_size[i];
    emx->size[i] = b_size[i];
  }
  emx->data = static_cast<float *>(
      std::malloc(static_cast<unsigned int>(numEl) * sizeof(float)));
  emx->numDimensions = b_numDimensions;
  emx->allocatedSize = numEl;
  return emx;
}

//
// Arguments    : creal32_T *b_data
//                int b_numDimensions
//                const int *b_size
// Return Type  : emxArray_creal32_T *
//
emxArray_creal32_T *emxCreateWrapperND_creal32_T(creal32_T *b_data,
                                                 int b_numDimensions,
                                                 const int *b_size)
{
  emxArray_creal32_T *emx;
  int numEl;
  emxInit_creal32_T(&emx, b_numDimensions);
  numEl = 1;
  for (int i{0}; i < b_numDimensions; i++) {
    numEl *= b_size[i];
    emx->size[i] = b_size[i];
  }
  emx->data = b_data;
  emx->numDimensions = b_numDimensions;
  emx->allocatedSize = numEl;
  emx->canFreeData = false;
  return emx;
}

//
// Arguments    : float *b_data
//                int b_numDimensions
//                const int *b_size
// Return Type  : emxArray_real32_T *
//
emxArray_real32_T *emxCreateWrapperND_real32_T(float *b_data,
                                               int b_numDimensions,
                                               const int *b_size)
{
  emxArray_real32_T *emx;
  int numEl;
  emxInit_real32_T(&emx, b_numDimensions);
  numEl = 1;
  for (int i{0}; i < b_numDimensions; i++) {
    numEl *= b_size[i];
    emx->size[i] = b_size[i];
  }
  emx->data = b_data;
  emx->numDimensions = b_numDimensions;
  emx->allocatedSize = numEl;
  emx->canFreeData = false;
  return emx;
}

//
// Arguments    : creal32_T *b_data
//                int rows
//                int cols
// Return Type  : emxArray_creal32_T *
//
emxArray_creal32_T *emxCreateWrapper_creal32_T(creal32_T *b_data, int rows,
                                               int cols)
{
  emxArray_creal32_T *emx;
  emxInit_creal32_T(&emx, 2);
  emx->size[0] = rows;
  emx->size[1] = cols;
  emx->data = b_data;
  emx->numDimensions = 2;
  emx->allocatedSize = rows * cols;
  emx->canFreeData = false;
  return emx;
}

//
// Arguments    : float *b_data
//                int rows
//                int cols
// Return Type  : emxArray_real32_T *
//
emxArray_real32_T *emxCreateWrapper_real32_T(float *b_data, int rows, int cols)
{
  emxArray_real32_T *emx;
  emxInit_real32_T(&emx, 2);
  emx->size[0] = rows;
  emx->size[1] = cols;
  emx->data = b_data;
  emx->numDimensions = 2;
  emx->allocatedSize = rows * cols;
  emx->canFreeData = false;
  return emx;
}

//
// Arguments    : int rows
//                int cols
// Return Type  : emxArray_creal32_T *
//
emxArray_creal32_T *emxCreate_creal32_T(int rows, int cols)
{
  emxArray_creal32_T *emx;
  int numEl;
  emxInit_creal32_T(&emx, 2);
  emx->size[0] = rows;
  numEl = rows * cols;
  emx->size[1] = cols;
  emx->data = static_cast<creal32_T *>(
      std::malloc(static_cast<unsigned int>(numEl) * sizeof(creal32_T)));
  emx->numDimensions = 2;
  emx->allocatedSize = numEl;
  return emx;
}

//
// Arguments    : int rows
//                int cols
// Return Type  : emxArray_real32_T *
//
emxArray_real32_T *emxCreate_real32_T(int rows, int cols)
{
  emxArray_real32_T *emx;
  int numEl;
  emxInit_real32_T(&emx, 2);
  emx->size[0] = rows;
  numEl = rows * cols;
  emx->size[1] = cols;
  emx->data = static_cast<float *>(
      std::malloc(static_cast<unsigned int>(numEl) * sizeof(float)));
  emx->numDimensions = 2;
  emx->allocatedSize = numEl;
  return emx;
}

//
// Arguments    : emxArray_creal32_T *emxArray
// Return Type  : void
//
void emxDestroyArray_creal32_T(emxArray_creal32_T *emxArray)
{
  emxFree_creal32_T(&emxArray);
}

//
// Arguments    : emxArray_real32_T *emxArray
// Return Type  : void
//
void emxDestroyArray_real32_T(emxArray_real32_T *emxArray)
{
  emxFree_real32_T(&emxArray);
}

//
// Arguments    : emxArray_creal32_T **pEmxArray
//                int b_numDimensions
// Return Type  : void
//
void emxInitArray_creal32_T(emxArray_creal32_T **pEmxArray, int b_numDimensions)
{
  emxInit_creal32_T(pEmxArray, b_numDimensions);
}

//
// Arguments    : emxArray_real32_T **pEmxArray
//                int b_numDimensions
// Return Type  : void
//
void emxInitArray_real32_T(emxArray_real32_T **pEmxArray, int b_numDimensions)
{
  emxInit_real32_T(pEmxArray, b_numDimensions);
}

//
// File trailer for diff_unwrap_phase_emxAPI.cu
//
// [EOF]
//
