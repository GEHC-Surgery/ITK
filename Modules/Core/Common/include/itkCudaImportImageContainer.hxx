/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef __itkCudaImportImageContainer_hxx
#define __itkCudaImportImageContainer_hxx

#include "itkCudaImportImageContainer.h"
#include "cuda.h"
#include "itkCudaUtility.h"
#include <cstring>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef CITK_DEBUG_ENABLED
#define CITK_DEBUG(x) std::cout << "CITK: " << m_Serial << ": " x << std::endl
#else
#define CITK_DEBUG(x)
#endif

namespace itk
{

template <typename TElementIdentifier, typename TElement>
CudaImportImageContainer<TElementIdentifier , TElement>
::CudaImportImageContainer()
{
#ifdef CITK_DEBUG_ENABLED
  m_Serial = (int)(rand()/10000000);
#endif
  m_ImageLocation = UNKNOWN;
  m_DevicePointer = 0;
  m_ImportPointer = 0;
  m_ContainerManageMemory = true;
  m_ContainerManageDevice = true;
  m_Capacity = 0;
  m_Size = 0;
}


template <typename TElementIdentifier, typename TElement>
CudaImportImageContainer< TElementIdentifier , TElement >
::~CudaImportImageContainer()
{
//  CITK_DEBUG(<< "destructor");

  // Do our best to deallocate memory without letting exceptions out of the
  // destructor (just log an error message and swallow the exception)
  try
    {
    this->DeallocateManagedCPUMemory();
    }
  catch(...)
    {
    itkGenericOutputMacro(<< "Failed to deallocate managed CPU memory")
    }
  try
    {
    this->DeallocateManagedGPUMemory();
    }
  catch(...)
    {
    itkGenericOutputMacro(<< "Failed to deallocate managed GPU memory")
    }
}


/**
    * Tell the container to allocate enough memory to allow at least
    * as many elements as the size given to be stored.
    */
template <typename TElementIdentifier, typename TElement>
void
CudaImportImageContainer< TElementIdentifier , TElement >
::Reserve(ElementIdentifier size)
{
  CITK_DEBUG(<< "Reserving CPU");

  /* Parent Class */
  if (m_ImportPointer)
    {
    if (size > m_Capacity)
      {
      TElement* temp = this->AllocateElements(size);
      // only copy the portion of the data used in the old buffer

      memcpy(temp, m_ImportPointer, m_Size*sizeof(TElement));

      DeallocateManagedMemory();

      m_ImportPointer = temp;
      m_ContainerManageMemory = true;
      m_Capacity = size;
      m_Size = size;
      this->Modified();
      }
    else
      {
      m_Size = size;
      this->Modified();
      }
    }
  else
    {
    m_ImportPointer = this->AllocateElements(size);
    m_Capacity = size;
    m_Size = size;
    m_ContainerManageMemory = true;
    this->Modified();
    }
  m_ImageLocation = CPU;
  CITK_DEBUG(<< "Reserved CPU ");
}

/**
 * Tell the container to allocate enough memory to allow at least
 * as many elements as the size given to be stored.
 */
template <typename TElementIdentifier, typename TElement>
void
CudaImportImageContainer< TElementIdentifier , TElement >
::ReserveGPU(ElementIdentifier size)
{
  CITK_DEBUG(<< "Reserving GPU ");

  /* Parent Class */
  if (m_DevicePointer)
    {
    if (size > m_Capacity)
      {
      CITK_DEBUG(<< "Reserving in new GPU buffer because " << size << " > " << m_Capacity);
      TElement* temp = this->AllocateGPUElements(size);
      // only copy the portion of the data used in the old buffer

      cudaMemcpy(temp, m_DevicePointer, m_Size*sizeof(TElement), cudaMemcpyDeviceToDevice);
      itkExceptionOnCudaErrorMacro(<< "Failed to reserve GPU memory (size: " << m_Size*sizeof(TElement) << " bytes)");

      DeallocateManagedMemory();

      m_DevicePointer = temp;
      m_ContainerManageMemory = true;
      m_Capacity = size;
      m_Size = size;
      this->Modified();
      }
    else
      {
      CITK_DEBUG(<< "GPU buffer already big enough: " << size << " <= " << m_Capacity);
      m_Size = size;
      this->Modified();
      }
    }
  else
    {
    m_DevicePointer = this->AllocateGPUElements(size);
    m_Capacity = size;
    m_Size = size;
    m_ContainerManageMemory = true;
    this->Modified();
    }
  m_ImageLocation = GPU;
  CITK_DEBUG(<< "Reserved GPU ");
}


/**
 * Tell the container to try to minimize its memory usage for storage of
 * the current number of elements.
 */
template <typename TElementIdentifier, typename TElement>
void
CudaImportImageContainer< TElementIdentifier , TElement >
::Squeeze(void)
{
  CITK_DEBUG(<< "Squeezing CPU");

  /* Parent Code */
  if (m_ImportPointer)
    {
    if (m_Size < m_Capacity)
      {
      const TElementIdentifier size = m_Size;
      TElement* temp = this->AllocateElements(size);
      memcpy(temp, m_ImportPointer, size*sizeof(TElement));

      DeallocateManagedMemory();

      m_ImportPointer = temp;
      m_ContainerManageMemory = true;
      m_Capacity = size;
      m_Size = size;

      this->Modified();
      }
    }

  CITK_DEBUG(<< "Squeezed CPU");
}


/**
 * Tell the container to try to minimize its memory usage for storage of
 * the current number of elements.
 */
template <typename TElementIdentifier, typename TElement>
void
CudaImportImageContainer< TElementIdentifier , TElement >
::Initialize(void)
{
  CITK_DEBUG(<< "Initializing CPU");

  /* Parent code */
  if (m_ImportPointer)
    {
    DeallocateManagedMemory();

    m_ContainerManageMemory = true;

    this->Modified();
    }

  CITK_DEBUG(<< "Initialized CPU");
}


/**
 * Set the pointer from which the image data is imported.  "num" is
 * the number of pixels in the block of memory. If
 * "LetContainerManageMemory" is false, then the application retains
 * the responsibility of freeing the memory for this image data.  If
 * "LetContainerManageMemory" is true, then this class will free the
 * memory when this object is destroyed.
 */
template <typename TElementIdentifier, typename TElement>
void
CudaImportImageContainer< TElementIdentifier , TElement >
::SetImportPointer(TElement *ptr, TElementIdentifier num,
                   bool LetContainerManageMemory)
{
  DeallocateManagedMemory();
  m_ImportPointer = ptr;
  m_ContainerManageMemory = LetContainerManageMemory;
  m_Capacity = num;
  m_Size = num;
  //AllocateGPU();

  this->Modified();
  m_ImageLocation = CPU;
}

template <typename TElementIdentifier, typename TElement>
void
CudaImportImageContainer< TElementIdentifier, TElement >
::SetDevicePointer(TElement *ptr, TElementIdentifier num,
                   bool LetContainerManageDevice)
{
  DeallocateManagedMemory();
  m_DevicePointer = ptr;
  m_ContainerManageDevice = LetContainerManageDevice;
  m_Capacity = num;
  m_Size = num;
  this->Modified();
  m_ImageLocation = GPU;
}

template <typename TElementIdentifier, typename TElement>
TElement* CudaImportImageContainer< TElementIdentifier , TElement >
::AllocateElements(ElementIdentifier size) const
{
  // Encapsulate all image memory allocation here to throw an
  // exception when memory allocation fails even when the compiler
  // does not do this by default.

  /* Parent code */
  TElement* data;
  try
    {
    data = new TElement[size];
    }
  catch(...)
    {
    data = 0;
    }
  if(!data)
    {
    // We cannot construct an error string here because we may be out
    // of memory.  Do not use the exception macro.
    throw MemoryAllocationError(__FILE__, __LINE__,
                                "Failed to allocate memory for image.",
                                ITK_LOCATION);
    }
  return data;

}

template <typename TElementIdentifier, typename TElement>
TElement* CudaImportImageContainer< TElementIdentifier , TElement >
::AllocateGPUElements(ElementIdentifier size) const
{
  // Encapsulate all image GPU memory allocation here to throw an
  // exception when memory allocation fails.
  TElement* data;
  cudaMalloc( &data, sizeof(TElement)*size);
  itkExceptionOnCudaErrorMacro(<< "Failed to allocate GPU memory for image.");

  CITK_DEBUG(<< "AllocateGPUElements: " << static_cast<void *>(data));
  return data;

}

template <typename TElementIdentifier, typename TElement>
void CudaImportImageContainer< TElementIdentifier , TElement >
::DeallocateManagedMemory()
{
  this->DeallocateManagedCPUMemory();
  this->DeallocateManagedGPUMemory();

  m_Capacity = 0;
  m_Size = 0;
}

template <typename TElementIdentifier, typename TElement>
void CudaImportImageContainer< TElementIdentifier , TElement >
::DeallocateManagedCPUMemory() const
{
  // CPU Deallocate
  if (m_ImportPointer && m_ContainerManageMemory)
    {
    delete [] m_ImportPointer;
    }

  m_ImportPointer = 0;
}

template <typename TElementIdentifier, typename TElement>
void CudaImportImageContainer< TElementIdentifier , TElement >
::DeallocateManagedGPUMemory() const
{
  //CITK_DEBUG(<< "DeallocateManagedGPUMemory (" << static_cast<void *>(m_DevicePointer) << ")");
  // GPU Deallocate
  if (m_DevicePointer && m_ContainerManageDevice)
    {
    CITK_DEBUG(<< "cudaFree(" << static_cast<void *>(m_DevicePointer) << ")");
    cudaFree(m_DevicePointer);
    itkExceptionOnCudaErrorMacro(<< "Failed to deallocate GPU memory for image (m_DevicePointer: "
                                 << static_cast<void *>(m_DevicePointer) << ")");
    }

  m_DevicePointer = 0;
}

template <typename TElementIdentifier, typename TElement>
void
CudaImportImageContainer< TElementIdentifier , TElement >
::PrintSelf(std::ostream& os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  os << indent << "Pointer: " << static_cast<void *>(m_ImportPointer) << std::endl;
  os << indent << "DevPointer: " << static_cast<void *>(m_DevicePointer) << std::endl;
  os << indent << "Image location: " << m_ImageLocation << std::endl;
  os << indent << "Container manages memory: "
     << (m_ContainerManageMemory ? "true" : "false") << std::endl;
  os << indent << "Container manages device memory: "
     << (m_ContainerManageDevice ? "true" : "false") << std::endl;
  os << indent << "Size: " << m_Size << std::endl;
  os << indent << "Capacity: " << m_Capacity << std::endl;
}

template <typename TElementIdentifier, typename TElement>
void
CudaImportImageContainer< TElementIdentifier , TElement >
::CopyToGPU() const
{

  AllocateGPU();
  CITK_DEBUG(<< "Copying to GPU (" << static_cast<void *>(m_DevicePointer)
             << " <-- " << static_cast<void *>(m_ImportPointer) << ")");
  cudaMemcpy(m_DevicePointer, m_ImportPointer,
             sizeof(TElement)*m_Size, cudaMemcpyHostToDevice);
  itkExceptionOnCudaErrorMacro(<< "Failed to copy CPU buffer to GPU");

  m_ImageLocation = GPU;
}

template <typename TElementIdentifier, typename TElement>
void
CudaImportImageContainer< TElementIdentifier , TElement >
::CopyToCPU() const
{
  AllocateCPU();
  CITK_DEBUG(<< "Copying to CPU (" << static_cast<void *>(m_ImportPointer)
             << " <-- " << static_cast<void *>(m_DevicePointer) << ")");
  cudaMemcpy(m_ImportPointer, m_DevicePointer,
             sizeof(TElement)*m_Size, cudaMemcpyDeviceToHost);
  itkExceptionOnCudaErrorMacro(<< "Failed to copy GPU buffer to CPU");

  m_ImageLocation = CPU;
}

template <typename TElementIdentifier, typename TElement>
void
CudaImportImageContainer< TElementIdentifier, TElement >
::AllocateGPU() const
{
  // should only need to allocate GPU memory if the device
  // pointer is null
  if (!m_DevicePointer)
    m_DevicePointer = this->AllocateGPUElements(m_Size);
}

template <typename TElementIdentifier, typename TElement>
void
CudaImportImageContainer< TElementIdentifier, TElement >
::AllocateCPU() const
{
  // should only need to allocate CPU memory if the import
  // pointer is null
  if (!m_ImportPointer)
    m_ImportPointer = this->AllocateElements(m_Size);
}

} // end namespace itk

#endif
