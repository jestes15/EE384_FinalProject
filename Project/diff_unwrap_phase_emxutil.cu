//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: diff_unwrap_phase_emxutil.cu
//
// GPU Coder version                    : 23.2
// CUDA/C/C++ source code generated on  : 03-May-2024 13:01:26
//

// Include Files
#include "diff_unwrap_phase_emxutil.h"
#include "diff_unwrap_phase_types.h"
#include "rt_nonfinite.h"
#include <algorithm>
#include <cstdlib>

// Function Definitions
//
// Arguments    : emxArray_real32_T *emxArray
//                int oldNumel
// Return Type  : void
//
void emxEnsureCapacity_real32_T(emxArray_real32_T *emxArray, int oldNumel)
{
  int i;
  int newNumel;
  void *newData;
  if (oldNumel < 0) {
    oldNumel = 0;
  }
  newNumel = 1;
  for (i = 0; i < emxArray->numDimensions; i++) {
    newNumel *= emxArray->size[i];
  }
  if (newNumel > emxArray->allocatedSize) {
    i = emxArray->allocatedSize;
    if (i < 16) {
      i = 16;
    }
    while (i < newNumel) {
      if (i > 1073741823) {
        i = MAX_int32_T;
      } else {
        i *= 2;
      }
    }
    newData = std::malloc(static_cast<unsigned int>(i) * sizeof(float));
    if (emxArray->data != nullptr) {
      std::copy(emxArray->data,
                emxArray->data + static_cast<unsigned int>(oldNumel),
                static_cast<float *>(newData));
      if (emxArray->canFreeData) {
        std::free(emxArray->data);
      }
    }
    emxArray->data = static_cast<float *>(newData);
    emxArray->allocatedSize = i;
    emxArray->canFreeData = true;
  }
}

//
// Arguments    : emxArray_creal32_T **pEmxArray
// Return Type  : void
//
void emxFree_creal32_T(emxArray_creal32_T **pEmxArray)
{
  if (*pEmxArray != static_cast<emxArray_creal32_T *>(nullptr)) {
    if (((*pEmxArray)->data != static_cast<creal32_T *>(nullptr)) &&
        (*pEmxArray)->canFreeData) {
      std::free((*pEmxArray)->data);
    }
    std::free((*pEmxArray)->size);
    std::free(*pEmxArray);
    *pEmxArray = static_cast<emxArray_creal32_T *>(nullptr);
  }
}

//
// Arguments    : emxArray_real32_T **pEmxArray
// Return Type  : void
//
void emxFree_real32_T(emxArray_real32_T **pEmxArray)
{
  if (*pEmxArray != static_cast<emxArray_real32_T *>(nullptr)) {
    if (((*pEmxArray)->data != static_cast<float *>(nullptr)) &&
        (*pEmxArray)->canFreeData) {
      std::free((*pEmxArray)->data);
    }
    std::free((*pEmxArray)->size);
    std::free(*pEmxArray);
    *pEmxArray = static_cast<emxArray_real32_T *>(nullptr);
  }
}

//
// Arguments    : emxArray_creal32_T **pEmxArray
//                int b_numDimensions
// Return Type  : void
//
void emxInit_creal32_T(emxArray_creal32_T **pEmxArray, int b_numDimensions)
{
  emxArray_creal32_T *emxArray;
  *pEmxArray = static_cast<emxArray_creal32_T *>(
      std::malloc(sizeof(emxArray_creal32_T)));
  emxArray = *pEmxArray;
  emxArray->data = static_cast<creal32_T *>(nullptr);
  emxArray->numDimensions = b_numDimensions;
  emxArray->size = static_cast<int *>(
      std::malloc(sizeof(int) * static_cast<unsigned int>(b_numDimensions)));
  emxArray->allocatedSize = 0;
  emxArray->canFreeData = true;
  for (int i{0}; i < b_numDimensions; i++) {
    emxArray->size[i] = 0;
  }
}

//
// Arguments    : emxArray_real32_T **pEmxArray
//                int b_numDimensions
// Return Type  : void
//
void emxInit_real32_T(emxArray_real32_T **pEmxArray, int b_numDimensions)
{
  emxArray_real32_T *emxArray;
  *pEmxArray =
      static_cast<emxArray_real32_T *>(std::malloc(sizeof(emxArray_real32_T)));
  emxArray = *pEmxArray;
  emxArray->data = static_cast<float *>(nullptr);
  emxArray->numDimensions = b_numDimensions;
  emxArray->size = static_cast<int *>(
      std::malloc(sizeof(int) * static_cast<unsigned int>(b_numDimensions)));
  emxArray->allocatedSize = 0;
  emxArray->canFreeData = true;
  for (int i{0}; i < b_numDimensions; i++) {
    emxArray->size[i] = 0;
  }
}

//
// File trailer for diff_unwrap_phase_emxutil.cu
//
// [EOF]
//
