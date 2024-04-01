// Copyright © 2024 Ahsan Iqbal
#include "tensor.h"
#include <cuda.h>
#include <cuda_runtime.h>

template class Tensor<double>;
template class Tensor<float>;
template class Tensor<int>;
/**
 * Constructs a tensor with the given shape
 *
 * @param shape a vector of unsigned integers, represents the shape of the tensor
 */
template <typename T>
Tensor<T>::Tensor(std::vector<uint> shape): data_h(nullptr), data_d(nullptr), numElements(1), shape(shape), strides(shape.size(), 1) {
    numElements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint>());
    data_h = new T[numElements];
    cudaMalloc((void**)&data_d, numElements * sizeof(T));
    
    for (uint i = strides.size() - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
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

template <typename T>
std::vector<uint> Tensor<T>::calculateMultiDimIndex(uint linearIndex) const {
    std::vector<uint> indices;
    for (uint i = 0; i < shape.size(); i++) {
        indices.push_back(linearIndex / strides[i]);
        linearIndex %= strides[i];
    }
    return indices;
}
template <typename T>
uint Tensor<T>::calculateLinearIndex(const std::vector<uint>& multiDimIndex) const {
    uint linearIndex = 0;
    uint offset = multiDimIndex.size() - shape.size();
    for (uint i = 0; i < shape.size(); i++) {
        linearIndex += multiDimIndex[i + offset] * strides[i];
    }
    return linearIndex;
}
template <typename T>
std::vector<uint> Tensor<T>::calculateBroadcastedIndex(const std::vector<uint>& indices) const {
    std::vector<uint> broadcastedIndices(indices.size(), 0);
    uint offset = indices.size() - shape.size();
    for (uint i = 0; i < shape.size(); i++) {
        if (shape[i] != 1) {
            broadcastedIndices[i + offset] = indices[i + offset];
        }
    }
    return broadcastedIndices;
}
/**
 * Adds two tensors
 *
 * @param a a tensor, represents the first tensor
 * @param b a tensor, represents the second tensor
 * @return a tensor, represents the sum of the two tensors
 */
/*template <typename T>
Tensor<T> operator+(const Tensor<T>& a, Tensor<T>& b) {
    a.validateShape(b);

    Tensor<T> result(a.calculateResultShape(b));

    for (uint i = 0; i < result.numElements; i++) {
        vector<uint> indices = result.calculateMultiDimIndex(i);

        vector<uint> aIndices = a.calculateBroadcastedIndex(indices);
        vector<uint> bIndices = b.calculateBroadcastedIndex(indices);
        
        uint aIndex = a.calculateLinearIndex(aIndices);
        uint bIndex = b.calculateLinearIndex(bIndices);
        
        result.data_h[i] = a.data_h[aIndex] + b.data_h[bIndex];
    }
    return result;
}*/