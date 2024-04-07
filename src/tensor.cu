// Copyright © 2024 Ahsan Iqbal
#include "tensor.h"

__global__ void VectorAdd(float *vector1, float *vector2, float *result,
                          float scaler1, float scaler2, uint numElements) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numElements) {
    result[index] = scaler1 * vector1[index] + scaler2 * vector2[index];
  }
}

__host__ __device__ uint CalculateLinearIndex(uint referenceLinearIndex, const uint *referenceShape, const uint *tensorShape,
                                 const uint *referenceStrides, const uint *tensorStrides, const uint shapeSize) {
  uint resultIndex = 0;
  for (int i = 0; i < shapeSize; i++) {
    int dimIndex = referenceLinearIndex / referenceShape[i];
    referenceLinearIndex %= referenceStrides[i];
    
    resultIndex += (tensorShape[i] == 1 ? 0 : dimIndex) * tensorStrides[i];
  }
  return  resultIndex;
}

/**
 * Adds two tensor of compatible shapes.
 * 
 * The kernel expects the tensors shapes to be of same length and compatible according to broadcasting rules.
 * Similarly the strides of the tensors must be of same length and compatible with their shapes.
 * 
 * @param tensor1: The first tensor.
 * @param tensor2: The second tensor.
 * @param result: The result tensor.
 * @param resultShape: The shape of the result tensor.
 * @param tensor1Shape: The shape of the first tensor.
 * @param tensor2Shape: The shape of the second tensor.
 * @param resultStrides: The strides of the result tensor.
 * @param tensor1Strides: The strides of the first tensor.
 * @param tensor2Strides: The strides of the second tensor.
 * @param shapeSize: The size of the shapes.
 * @param numElements: The number of elements in the result tensor.
*/

__global__ void TensorAdd(float *result, float *tensor1, float *tensor2,
                          uint *resultShape, uint *tensor1Shape, uint *tensor2Shape, 
                          uint *resultStrides, uint *tensor1Strides, uint *tensor2Strides,
                          uint shapeSize, uint numElements) {
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numElements) {
    /*int tempIndex = index;
    for (int i = 0; i < shapeSize; i++) {
      int dimIndex = tempIndex / resultStrides[i];
      tempIndex %= resultStrides[i];

      index1 += (tensor1Shape[i] == 1 ? 0 : dimIndex) * tensor1Strides[i];
      index2 += (tensor2Shape[i] == 1 ? 0 : dimIndex) * tensor2Strides[i];
    }*/
    uint index1 = CalculateLinearIndex(index, resultShape, tensor1Shape, resultStrides, tensor1Strides, shapeSize);
    uint index2 = CalculateLinearIndex(index, resultShape, tensor2Shape, resultStrides, tensor2Strides, shapeSize);
    result[index] = tensor1[index1] + tensor2[index2];
  }
}

/**
 * Multilpies two tensor of compatible shapes.
 * 
 * The kernel expects the tensors shapes to be of same length and compatible according to broadcasting rules.
 * Similarly the strides of the tensors must be of same length and compatible with their shapes.
 * 
 * @param tensor1: The first tensor.
 * @param tensor2: The second tensor.
 * @param result: The result tensor.
 * @param resultShape: The shape of the result tensor.
 * @param tensor1Shape: The shape of the first tensor.
 * @param tensor2Shape: The shape of the second tensor.
 * @param resultStrides: The strides of the result tensor.
 * @param tensor1Strides: The strides of the first tensor.
 * @param tensor2Strides: The strides of the second tensor.
 * @param shapeSize: The size of the shapes.
 * @param numElements: The number of elements in the result tensor.
*/

__global__ void TensorMul(float *result, float *tensor1, float *tensor2,
                          uint *resultShape, uint *tensor1Shape, uint *tensor2Shape, 
                          uint *resultStrides, uint *tensor1Strides, uint *tensor2Strides,
                          uint shapeSize, uint numElements) {
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numElements) {
    /*int tempIndex = index;
    for (int i = 0; i < shapeSize; i++) {
      int dimIndex = tempIndex / resultStrides[i];
      tempIndex %= resultStrides[i];

      index1 += (tensor1Shape[i] == 1 ? 0 : dimIndex) * tensor1Strides[i];
      index2 += (tensor2Shape[i] == 1 ? 0 : dimIndex) * tensor2Strides[i];
    }*/
    uint index1 = CalculateLinearIndex(index, resultShape, tensor1Shape, resultStrides, tensor1Strides, shapeSize);
    uint index2 = CalculateLinearIndex(index, resultShape, tensor2Shape, resultStrides, tensor2Strides, shapeSize);
    result[index] = tensor1[index1] * tensor2[index2];
  }
}


void LaunchAddKernel(Tensor &result, Tensor &tensor1, Tensor &tensor2) {

    uint shapeSize = result.GetShape().size();
    uint numElements = result.GetNumElements();
    
    uint *resultShape, *resultStrides, *tensor1Shape, *tensor1Strides, *tensor2Shape, *tensor2Strides;

    cudaMalloc((void**)&resultShape, shapeSize * sizeof(uint));
    cudaMalloc((void**)&resultStrides, shapeSize * sizeof(uint));
    cudaMemcpy(resultShape, result.GetShape().data(), shapeSize * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(resultStrides, result.GetStrides().data(), shapeSize * sizeof(uint), cudaMemcpyHostToDevice);


    cudaMalloc((void**)&tensor1Shape, shapeSize * sizeof(uint));
    cudaMalloc((void**)&tensor1Strides, shapeSize * sizeof(uint));
    cudaMemcpy(tensor1Shape, tensor1.GetShape().data(), shapeSize * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(tensor1Strides, tensor1.GetStrides().data(), shapeSize * sizeof(uint), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&tensor2Shape, shapeSize * sizeof(uint));
    cudaMalloc((void**)&tensor2Strides, shapeSize * sizeof(uint));
    cudaMemcpy(tensor2Shape, tensor2.GetShape().data(), shapeSize * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(tensor2Strides, tensor2.GetStrides().data(), shapeSize * sizeof(uint), cudaMemcpyHostToDevice);


    TensorAdd<<<(numElements) / 256, 256>>>(result.GetData(), tensor1.GetData(), tensor2.GetData(), resultShape, tensor1Shape, tensor2Shape, resultStrides, tensor1Strides, tensor2Strides, shapeSize, numElements);
    
    

    cudaFree(resultShape);
    cudaFree(resultStrides);
    cudaFree(tensor1Shape);
    cudaFree(tensor1Strides);
    cudaFree(tensor2Shape);
    cudaFree(tensor2Strides);
}