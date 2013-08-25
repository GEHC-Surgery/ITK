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
#ifndef __CudaImageToImageFilter_hxx
#define __CudaImageToImageFilter_hxx
#include "CudaImageToImageFilter.h"

namespace itk
{

/**
 *
 */
template <class TInputImage, class TOutputImage>
CudaImageToImageFilter<TInputImage,TOutputImage>
::CudaImageToImageFilter()
{
  // Modify superclass default values, can be overridden by subclasses
  this->SetNumberOfRequiredInputs(1);
}

/**
 *
 */
template <class TInputImage, class TOutputImage>
CudaImageToImageFilter<TInputImage,TOutputImage>
::~CudaImageToImageFilter()
{
}

template <class TInputImage, class TOutputImage>
void
CudaImageToImageFilter<TInputImage,TOutputImage>
::AllocateOutputs()
{
  typedef ImageBase<OutputImageDimension> ImageBaseType;
  typename ImageBaseType::Pointer outputPtr;

  // Allocate the output memory
  for (unsigned int i=0; i < this->GetNumberOfOutputs(); i++)
    {

    // Check whether the output is an image of the appropriate
    // dimension (use ProcessObject's version of the GetInput()
    // method since it returns the input as a pointer to a
    // DataObject as opposed to the subclass version which
    // static_casts the input to an TInputImage).
    outputPtr = dynamic_cast< ImageBaseType *>( this->ProcessObject::GetOutput(i) );

    if ( outputPtr )
      {
//       outputPtr->SetBufferedRegion( outputPtr->GetRequestedRegion() );
//       outputPtr->AllocateGPU();
      // may not be the best way of doing this. Casting loses the
      // AllocateGPU method
      OutputImagePointer op2 = this->GetOutput(i);
      op2->SetBufferedRegion( outputPtr->GetRequestedRegion() );
      op2->AllocateGPU();
      }
    }
}


} // end namespace itk

#endif
