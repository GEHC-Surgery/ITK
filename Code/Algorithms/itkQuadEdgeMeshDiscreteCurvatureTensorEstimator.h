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
#ifndef __itkQuadEdgeMeshDiscreteCurvatureTensorEstimator_h
#define __itkQuadEdgeMeshDiscreteCurvatureTensorEstimator_h

namespace itk
{
/**
 * \class QuadEdgeMeshDiscreteCurvatureTensorEstimator
 *
 * \brief FIXME Add documentation here
 *
 */
template< class TInputMesh, class TOutputMesh >
class ITK_EXPORT QuadEdgeMeshDiscreteCurvatureTensorEstimator:
  public QuadEdgeMeshToQuadEdgeMeshFilter< TInputMesh, TOutputMesh >
{
public:
  typedef QuadEdgeMeshDiscreteCurvatureTensorEstimator Self;
  typedef SmartPointer< Self >                         Pointer;
  typedef SmartPointer< const Self >                   ConstPointer;
  typedef QuadEdgeMeshToQuadEdgeMeshFilter             Superclass;

  /** Run-time type information (and related methods).   */
  itkTypeMacro(QuadEdgeMeshDiscreteCurvatureTensorEstimator, QuadEdgeMeshToQuadEdgeMeshFilter);

  /** New macro for creation of through a Smart Pointer   */
  itkNewMacro(Self);

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro( OutputIsFloatingPointCheck,
                   ( Concept::IsFloatingPoint< OutputCurvatureType > ) );
  /** End concept checking */
#endif

protected:
  QuadEdgeMeshDiscreteCurvatureTensorEstimator() {}
  ~QuadEdgeMeshDiscreteCurvatureTensorEstimator() {}

  ///TODO to be implemented
  virtual void GenerateData()
  {}

private:
  QuadEdgeMeshDiscreteCurvatureTensorEstimator(const Self &); // purposely not
                                                              // implemented
  void operator=(const Self &);                               // purposely not
                                                              // implemented
};
}

#endif