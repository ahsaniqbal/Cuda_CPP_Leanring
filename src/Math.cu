// Copyright © 2024 Ahsan Iqbal
#include <cuda_runtime.h>
#include <iostream>

__global__ void VectorAdd(float *vector1, float *vector2, float *result,
                          float scaler1, float scaler2, uint numElements) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numElements) {
    result[index] = scaler1 * vector1[index] + scaler2 * vector2[index];
  }
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

__global__ void TensorAdd(float *tensor1, float *tensor2, float *result,
                          int *resultShape, int *tensor1Shape, int *tensor2Shape, 
                          int *resultStrides, int *tensor1Strides, int *tensor2Strides,
                          int shapeSize, int numElements) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numElements) {
    int index1 = 0, index2 = 0;
    int tempIndex = index;
    for (int i = 0; i < shapeSize; i++) {
      int dimIndex = tempIndex / resultStrides[i];
      tempIndex %= resultStrides[i];

      index1 += (tensor1Shape[i] == 1 ? 0 : dimIndex) * tensor1Strides[i];
      index2 += (tensor2Shape[i] == 1 ? 0 : dimIndex) * tensor2Strides[i];
    }
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

__global__ void TensorMul(float *tensor1, float *tensor2, float *result,
                          int *resultShape, int *tensor1Shape, int *tensor2Shape, 
                          int *resultStrides, int *tensor1Strides, int *tensor2Strides,
                          int shapeSize, int numElements) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numElements) {
    int index1 = 0, index2 = 0;
    int tempIndex = index;
    for (int i = 0; i < shapeSize; i++) {
      int dimIndex = tempIndex / resultStrides[i];
      tempIndex %= resultStrides[i];

      index1 += (tensor1Shape[i] == 1 ? 0 : dimIndex) * tensor1Strides[i];
      index2 += (tensor2Shape[i] == 1 ? 0 : dimIndex) * tensor2Strides[i];
    }
    result[index] = tensor1[index1] * tensor2[index2];
  }
}