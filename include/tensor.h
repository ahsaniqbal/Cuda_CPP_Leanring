// Copyright © 2024 Ahsan Iqbal
#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
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

    void CalculateStrides() {
        for (int i = strides.size() - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
public:
    Tensor(std::vector<uint> shape, bool isOnDevice);
    Tensor(std::vector<float> data, std::vector<uint> shape, bool isOnDevice);
    Tensor(const Tensor& other);
    ~Tensor();

    std::vector<uint> GetShape() const { return shape; }
    std::vector<uint> GetStrides() const { return strides; }
    bool IsOnDevice() const { return isOnDevice; }
    float* GetHostData() const { return data_h; }
    float* GetDeviceData() const { return data_d; }
    float* GetData() const { return isOnDevice ? data_d : data_h; }
    uint GetNumElements() const { return numElements; }

    void ToDevice();
    void ToHost();

    uint CalculateLinearIndex(uint referenceLinearIndex, const std::vector<uint>& referenceStrides) const;
    std::vector<uint> CalculateBroadcastedShape(const uint referenceShapeSize) const;
    std::vector<uint> CalculateBroadcastedStrides(const uint referenceShapeSize) const;

    void ValidateShapesBroadcastOperation(const Tensor& other) const;
    std::vector<uint> CalculateBroadcastResultShape(const Tensor& other) const;

    Tensor operator+(const Tensor& b) const;
};
void LaunchAddKernel(Tensor &result, const Tensor &tensor1, const Tensor &tensor2);

__host__ __device__ uint CalculateLinearIndex(uint referenceLinearIndex, const uint *referenceStrides, const uint *tensorShape,
                                const uint *tensorStrides, const uint shapeSize);
#endif