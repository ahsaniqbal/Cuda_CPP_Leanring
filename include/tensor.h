// Copyright © 2024 Ahsan Iqbal
#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#include <numeric>
#define uint std::uint32_t

template <typename T>
class Tensor {
private:
    T* data_h;
    T* data_d;

    uint numElements;
    std::vector<uint> shape;
    std::vector<uint> strides;
public:
    Tensor(std::vector<uint> shape);
    Tensor(Tensor<T>& other);
    ~Tensor();

    std::vector<uint> getShape() const { return shape; }
    std::vector<uint> getStrides() const { return strides; }
    uint getNumElements() const { return numElements; }

    void ToDevice();
    void ToHost();

    std::vector<uint> calculateMultiDimIndex(uint linearIndex) const;
    uint calculateLinearIndex(const std::vector<uint>& multiDimIndex) const;
    std::vector<uint> calculateBroadcastedIndex(const std::vector<uint>& indices) const;

    void validateShape(const Tensor<T>& other);
    std::vector<uint> calculateResultShape(const Tensor<T>& other);

    //friend Tensor<T> operator+(const Tensor<T>& a, Tensor<T>& b);
};
#endif