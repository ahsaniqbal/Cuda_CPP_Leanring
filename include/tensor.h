// Copyright © 2024 Ahsan Iqbal
#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>
#define uint std::uint32_t

template <typename T>
class Tensor {
private:
    T* data_h;
    T* data_d;
    std::vector<uint> shape;
    std::vector<uint> strides;

    uint numElements;
public:
    Tensor(uint numElements);
    Tensor(std::vector<uint> shape);
    Tensor(Tensor<T>& other);
    ~Tensor();

    void ToDevice();
    void ToHost();

    void validateShape(const Tensor<T>& other);
    std::vector<uint> calculateResultShape(const Tensor<T>& other);
};

#endif