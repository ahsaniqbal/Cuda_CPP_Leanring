// Copyright © 2024 Ahsan Iqbal
#include "tensor.h"
#include <cuda.h>
#include <cuda_runtime.h>

/**
 * Constructs a 1D tensor with numElements elements
 *
 * @param numElements an unsigned integer, represents the number of elements in the tensor
 */
template <typename T>
Tensor<T>::Tensor(uint numElements) {
    this->numElements = numElements;
    data_h = new T[this->numElements];
    cudaMalloc((void**)&data_d, this->numElements * sizeof(T));
 
    shape = {this->numElements};
    strides = {1};
}

/**
 * Constructs a tensor with the given shape
 *
 * @param shape a vector of unsigned integers, represents the shape of the tensor
 */
template <typename T>
Tensor<T>::Tensor(std::vector<uint> shape) {
    this->numElements = 1;
    for (uint i = 0; i < shape.size(); i++) {
        this->numElements *= shape[i];
    }
    data_h = new T[this->numElements];
    cudaMalloc((void**)&data_d, this->numElements * sizeof(T));
 
    this->shape = shape;
    
    strides = {1};
    for(uint i = shape.size() - 2; i >= 0; i--) {
        strides.insert(strides.begin(), strides[0] * shape[i + 1]);
    }
}

/**
 * Constructs a tensor with the same data as the given tensor
 *
 * @param other a tensor, represents the tensor to be copied
 */
template <typename T>
Tensor<T>::Tensor(Tensor<T>& other) {
    data_h = new T[other.numElements];
    cudaMalloc((void**)&data_d, other.numElements * sizeof(T));
    
    cudaMemcpy(data_d, other.data_d, other.numElements * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(data_h, other.data_h, other.numElements * sizeof(T), cudaMemcpyHostToHost);

    shape = other.shape;
    strides = other.strides;
}

/**
 * Destructs the tensor
 */
template <typename T>
Tensor<T>::~Tensor() {
    delete[] data_h;
    cudaFree(data_d);
}

/**
 * Copies the data from host to device
 */
template <typename T>
void Tensor<T>::ToDevice() {
    cudaMemcpy(data_d, data_h, numElements * sizeof(T), cudaMemcpyHostToDevice);
}

/**
 * Copies the data from device to host
 */
template <typename T>
void Tensor<T>::ToHost() {
    cudaMemcpy(data_h, data_d, numElements * sizeof(T), cudaMemcpyDeviceToHost);
}

/**
 * Validates the compatibility of shape of the tensor with the given tensor
 *
 * @param other a tensor, represents the tensor to be compared with
 */
template <typename T>
void Tensor<T>::validateShape(const Tensor<T>& other) {
    uint i = this->shape.size() - 1, j = other.shape.size() - 1;

    for (; i >= 0 and j >= 0; i--, j-- ) {
        if (this->shape[i] != other.shape[j] and (this->shape[i] != 1 or other.shape[j] != 1)) {
            throw std::invalid_argument("Shapes are not compatible for addition");
        }
    }
}

/**
 * Calculates the shape of the resulting tensor after performing an operation with the given tensor
 *
 * @param other a tensor, represents the tensor to be compared with
 * @return a vector of unsigned integers, represents the shape of the resulting tensor
*/
template <typename T>
std::vector<uint> Tensor<T>::calculateResultShape(const Tensor<T>& other) {
    std::vector<uint> resultShape;
    uint i = this->shape.size() - 1, j = other.shape.size() - 1;

    for (; i >= 0 and j >= 0; i--, j-- ) {
        resultShape.insert(resultShape.begin(), std::max(this->shape[i], other.shape[j]));
    }

    for (; i >= 0; i--) {
        resultShape.insert(resultShape.begin(), this->shape[i]);
    }
    for (; j >= 0; j--) {
        resultShape.insert(resultShape.begin(), other.shape[j]);
    }

    return resultShape;
}