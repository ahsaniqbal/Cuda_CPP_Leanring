// Copyright © 2024 Ahsan Iqbal
#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <math.h>
#include <cuda_runtime.h>
#define uint std::uint32_t


class Tensor {
private:
    float* data_h;
    float* data_d;
    bool isOnDevice;

    uint numElements;
    std::vector<uint> shape;
    std::vector<uint> strides;
public:
    Tensor(std::vector<uint> shape, bool isOnDevice);
    Tensor(Tensor& other);
    ~Tensor();

    std::vector<uint> GetShape() const { return shape; }
    std::vector<uint> GetStrides() const { return strides; }
    float* GetData() const { return isOnDevice ? data_d : data_h; }
    uint GetNumElements() const { return numElements; }

    void ToDevice();
    void ToHost();

    uint CalculateLinearIndex(uint referenceLinearIndex, const std::vector<uint>& referenceShape, const std::vector<uint>& referenceStrides) const;
    std::vector<uint> CalculateBroadcastedShape(const uint referenceShapeSize) const;
    std::vector<uint> CalculateBroadcastedStrides(const uint referenceShapeSize) const;

    void ValidateShapesBroadcastOperation(const Tensor& other);
    std::vector<uint> CalculateBroadcastResultShape(const Tensor& other) const;

    Tensor operator+(Tensor& b);
};
void LaunchAddKernel(Tensor &result, Tensor &tensor1, Tensor &tensor2);

__host__ __device__ uint CalculateLinearIndex(uint referenceLinearIndex, const uint *referenceShape, const uint *tensorShape,
                                 const uint *referenceStrides, const uint *tensorStrides, const uint shapeSize);
#endif