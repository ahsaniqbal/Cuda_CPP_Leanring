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
