/**
 * _____________________________________________________________________
 *
 * @file
 *
 * @brief <brief description>
 *
 * @details <detailed description>
 *
 * @sddtag{<sdd_tag_id>}
 *
 * _____________________________________________________________________
 */
/*
 *                    Copyright 2012, GE Healthcare.
 *
 *     This Program is an unpublished work that is fully protected
 *     by the United States Copyright Laws and is fully considered
 *     to be a TRADE SECRET belonging to GE Healthcare.
 *
 *     This document shall not be copied, in whole or in part, nor
 *     shall the algorithms or code defined within this document be
 *     disclosed to others without the express written permission
 *     of GE Healthcare.
 * _____________________________________________________________________
 */

#ifndef ITKCUDAUTILITY_H_
#define ITKCUDAUTILITY_H_

#include <cuda_runtime.h>
#include <itkMacro.h>

#define itkExceptionOnCudaErrorMacro(x) \
   { \
      const cudaError_t lastCudaError = cudaGetLastError(); \
      if (lastCudaError != cudaSuccess) \
      { \
         itkExceptionMacro(<< "CUDA error " << lastCudaError << ": " \
                           << cudaGetErrorString(lastCudaError) << std::endl x); \
      } \
   }

#define itkGenericExceptionOnCudaErrorMacro(x) \
   { \
      const cudaError_t lastCudaError = cudaGetLastError(); \
      if (lastCudaError != cudaSuccess) \
      { \
         itkGenericExceptionMacro(<< "CUDA error " << lastCudaError << ": " \
                                  << cudaGetErrorString(lastCudaError) << std::endl x); \
      } \
   }


#endif /* #ifndef ITKCUDAUTILITY_H_ */
