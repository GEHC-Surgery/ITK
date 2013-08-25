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
#ifndef __itkCudaImportImageContainer_h
#define __itkCudaImportImageContainer_h

#include "itkObject.h"
#include "itkObjectFactory.h"
#include "itkImportImageContainer.h"
#include <cuda_runtime.h>
#include <utility>

//#define CITK_DEBUG_ENABLED 1

namespace itk
{

/** \class ImportImageContainer
 * Defines an itk::Image front-end to a standard C-array. This container
 * conforms to the ImageContainerInterface. This is a full-fleged Object,
 * so there is modification time, debug, and reference count information.
 *
 * Template parameters for ImportImageContainer:
 *
 * TElementIdentifier =
 *     An INTEGRAL type for use in indexing the imported buffer.
 *
 * TElement =
 *    The element type stored in the container.
 *
 * \ingroup ImageObjects
 * \ingroup IOFilters
 */

template <typename TElementIdentifier, typename TElement>
class CudaImportImageContainer:
    public ImportImageContainer <TElementIdentifier, TElement>
{
public:
  /** Standard class typedefs. */
  typedef CudaImportImageContainer                            Self;
  typedef ImportImageContainer<TElementIdentifier, TElement>  Superclass;
  typedef SmartPointer<Self>                                  Pointer;
  typedef SmartPointer<const Self>                            ConstPointer;

  /** Save the template parameters. */
  typedef TElementIdentifier  ElementIdentifier;
  typedef TElement            Element;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Standard part of every itk Object. */
  itkTypeMacro(CudaImportImageContainer, ImportImageContainer);

  /** Get the pointer from which the image data is imported. */
  TElement *GetImportPointer()
  {
    if (m_ImageLocation==GPU) { CopyToCPU(); }
    return m_ImportPointer;
  };

  TElement *GetImportPointer() const
  {
    if (m_ImageLocation==GPU) { CopyToCPU();}
    return m_ImportPointer;
  }

  TElement *GetDevicePointer()
  {
    if (m_ImageLocation==CPU) { CopyToGPU(); }
    return m_DevicePointer;
  }

// TElement *GetDevicePointer() const
//          {
//             if (m_ImageLocation==CPU) { CopyToGPU(); m_ImageLocation=BOTH; }
//             return m_DevicePointer;
//          };


  /** Set the pointer from which the image data is imported.  "num" is
   * the number of pixels in the block of memory. If
   * "LetContainerManageMemory" is false, then the application retains
   * the responsibility of freeing the memory for this image data.  If
   * "LetContainerManageMemory" is true, then this class will free the
   * memory when this object is destroyed. */
  void SetImportPointer(TElement *ptr, TElementIdentifier num,
                        bool LetContainerManageMemory = false);
  void SetDevicePointer(TElement *ptr, TElementIdentifier
                        num, bool LetContainerManageMemory = false);

  /** Index operator. This version can be an lvalue. */
  TElement & operator[](const ElementIdentifier id)
  {
    if (m_ImageLocation==GPU) { CopyToCPU(); }
    return m_ImportPointer[id];
  }

  /** Index operator. This version can only be an rvalue */
  const TElement & operator[](const ElementIdentifier id) const
  {
    if (m_ImageLocation==GPU) { CopyToCPU();}
    return m_ImportPointer[id];
  }

  /** Return a pointer to the beginning of the buffer.  This is used by
   * the image iterator class. */
  TElement *GetBufferPointer()
  {
    if (m_ImageLocation == GPU)
      {
      CopyToCPU();
      }
    return m_ImportPointer;
  }

  /** Get the capacity of the container. */
  ElementIdentifier Capacity(void) const
  { return (ElementIdentifier) m_Capacity; }

  /** Get the number of elements currently stored in the container. */
  ElementIdentifier Size(void) const
  { return (ElementIdentifier) m_Size; }

  /** Tell the container to allocate enough memory to allow at least
   * as many elements as the size given to be stored.  If new memory
   * needs to be allocated, the contents of the old buffer are copied
   * to the new area.  The old buffer is deleted if the original pointer
   * was passed in using "LetContainerManageMemory"=true. The new buffer's
   * memory management will be handled by the container from that point on.
   *
   * In general, Reserve should not change the usable elements of the
   * container. However, in this particular case, Reserve as a Resize
   * semantics that is kept for backward compatibility reasons.
   *
   * \sa SetImportPointer() */
  void Reserve(ElementIdentifier num);
  void ReserveGPU(ElementIdentifier num);

  /** Tell the container to try to minimize its memory usage for
   * storage of the current number of elements.  If new memory is
   * allocated, the contents of old buffer are copied to the new area.
   * The previous buffer is deleted if the original pointer was in
   * using "LetContainerManageMemory"=true.  The new buffer's memory
   * management will be handled by the container from that point on. */
  void Squeeze(void);

  /** Tell the container to release any of its allocated memory. */
  void Initialize(void);


  /** These methods allow to define whether upon destruction of this class
   *  the memory buffer should be released or not.  Setting it to true
   *  (or ON) makes that this class will take care of memory release.
   *  Setting it to false (or OFF) will prevent the destructor from
   *  deleting the memory buffer. This is desirable only when the data
   *  is intended to be used by external applications.
   *  Note that the normal logic of this class set the value of the boolean
   *  flag. This may override your setting if you call this methods prematurely.
   *  \warning Improper use of these methods will result in memory leaks */
  itkSetMacro(ContainerManageMemory,bool);
  itkGetConstMacro(ContainerManageMemory,bool);
  itkBooleanMacro(ContainerManageMemory);
  itkSetMacro(ContainerManageDevice,bool);
  itkGetConstMacro(ContainerManageDevice,bool);
  itkBooleanMacro(ContainerManageDevice);

protected:
  CudaImportImageContainer();
  virtual ~CudaImportImageContainer();

  /** PrintSelf routine. Normally this is a protected internal method. It is
   * made public here so that Image can call this method.  Users should not
   * call this method but should call Print() instead. */
  void PrintSelf(std::ostream& os, Indent indent) const;

  virtual TElement* AllocateElements(ElementIdentifier size) const;
  virtual TElement* AllocateGPUElements(ElementIdentifier size) const;
  virtual void DeallocateManagedMemory();
  virtual void DeallocateManagedCPUMemory() const;
  virtual void DeallocateManagedGPUMemory() const;

  /* Set the m_Size member that represents the number of elements
   * currently stored in the container. Use this function with great
   * care since it only changes the m_Size member and not the actual size
   * of the import pointer m_ImportPointer. It should typically
   * be used only to override AllocateElements and
   * DeallocateManagedMemory. */
  itkSetMacro(Size,TElementIdentifier);

  /* Set the m_Capacity member that represents the capacity of
   * the current container. Use this function with great care
   * since it only changes the m_Capacity member and not the actual
   * capacity of the import pointer m_ImportPointer. It should typically
   * be used only to override AllocateElements and
   * DeallocateManagedMemory. */
  itkSetMacro(Capacity,TElementIdentifier);


  /* Set the m_ImportPointer member. Use this function with great care
   * since it only changes the m_ImportPointer member but not the m_Size
   * and m_Capacity members. It should typically be used only to override
   * AllocateElements and DeallocateManagedMemory. */
  void SetImportPointer(TElement *ptr)
  { m_ImportPointer=ptr; m_ImageLocation=CPU; }
  void SetDevicePointer(TElement *ptr)
  { m_DevicePointer=ptr; m_ImageLocation=GPU; }

private:
  CudaImportImageContainer(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  void CopyToGPU() const;
  void CopyToCPU() const;
  void AllocateGPU() const;
  void AllocateCPU() const;

  mutable TElement    *m_ImportPointer;
  mutable TElement    *m_DevicePointer;
  TElementIdentifier   m_Size;
  TElementIdentifier   m_Capacity;
  bool                 m_ContainerManageMemory;
  bool                 m_ContainerManageDevice;

#ifdef CITK_DEBUG_ENABLED
  int                  m_Serial;
#endif

  mutable enum memoryStatus{
    UNKNOWN,
    BOTH,
    CPU,
    GPU
  } m_ImageLocation;
};

} // end namespace itk

// Define instantiation macro for this template.
#define ITK_TEMPLATE_CudaImportImageContainer(_, EXPORT, x, y) namespace itk { \
    _(2(class EXPORT CudaImportImageContainer< ITK_TEMPLATE_2 x >)) \
      namespace Templates { typedef CudaImportImageContainer<ITK_TEMPLATE_2 x > CudaImportImageContainer##y; } \
  }

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkCudaImportImageContainer.hxx"
#endif

#endif
