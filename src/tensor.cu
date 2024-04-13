// Copyright © 2024 Ahsan Iqbal
#include "tensor.h"

__global__ void VectorAdd(float *vector1, float *vector2, float *result,
                          float scaler1, float scaler2, uint numElements) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numElements) {
    result[index] = scaler1 * vector1[index] + scaler2 * vector2[index];
  }
}

__host__ __device__ uint CalculateLinearIndex(uint referenceLinearIndex, const uint *referenceStrides, const uint *tensorShape,
                                              const uint *tensorStrides, const uint shapeSize) {
  uint resultIndex = 0;
  for (int i = 0; i < shapeSize; i++) {
    int dimIndex = referenceLinearIndex / referenceStrides[i];
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
                          uint *tensor1Shape, uint *tensor2Shape, 
                          uint *resultStrides, uint *tensor1Strides, uint *tensor2Strides,
                          uint shapeSize, uint numElements) {
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numElements) {
    uint index1 = CalculateLinearIndex(index, resultStrides, tensor1Shape, tensor1Strides, shapeSize);
    uint index2 = CalculateLinearIndex(index, resultStrides, tensor2Shape, tensor2Strides, shapeSize);
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
                          uint *tensor1Shape, uint *tensor2Shape, 
                          uint *resultStrides, uint *tensor1Strides, uint *tensor2Strides,
                          uint shapeSize, uint numElements) {
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < numElements) {
    uint index1 = CalculateLinearIndex(index, resultStrides, tensor1Shape, tensor1Strides, shapeSize);
    uint index2 = CalculateLinearIndex(index, resultStrides, tensor2Shape, tensor2Strides, shapeSize);
    result[index] = tensor1[index1] * tensor2[index2];
  }
}


void LaunchAddKernel(Tensor &result, const Tensor &tensor1, const Tensor &tensor2) {
    uint shapeSize = result.GetShape().size();
    uint numElements = result.GetNumElements();

    uint *resultStrides, *tensor1Shape, *tensor1Strides, *tensor2Shape, *tensor2Strides;

    cudaMalloc((void**)&resultStrides, shapeSize * sizeof(uint));
    cudaMemcpy(resultStrides, result.GetStrides().data(), shapeSize * sizeof(uint), cudaMemcpyHostToDevice);


    cudaMalloc((void**)&tensor1Shape, shapeSize * sizeof(uint));
    cudaMalloc((void**)&tensor1Strides, shapeSize * sizeof(uint));
    cudaMemcpy(tensor1Shape, tensor1.CalculateBroadcastedShape(shapeSize).data(), shapeSize * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(tensor1Strides, tensor1.CalculateBroadcastedStrides(shapeSize).data(), shapeSize * sizeof(uint), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&tensor2Shape, shapeSize * sizeof(uint));
    cudaMalloc((void**)&tensor2Strides, shapeSize * sizeof(uint));
    cudaMemcpy(tensor2Shape, tensor2.CalculateBroadcastedShape(shapeSize).data(), shapeSize * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(tensor2Strides, tensor2.CalculateBroadcastedStrides(shapeSize).data(), shapeSize * sizeof(uint), cudaMemcpyHostToDevice);


    TensorAdd<<<static_cast<int>(std::ceil(numElements/ 256.0)), 256>>>(result.GetData(), tensor1.GetData(), tensor2.GetData(), tensor1Shape, tensor2Shape, resultStrides, tensor1Strides, tensor2Strides, shapeSize, numElements);

    cudaFree(resultStrides);
    cudaFree(tensor1Shape);
    cudaFree(tensor1Strides);
    cudaFree(tensor2Shape);
    cudaFree(tensor2Strides);
}