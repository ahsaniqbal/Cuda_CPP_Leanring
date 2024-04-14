// Copyright © 2024 Ahsan Iqbal
#include "tensor.h"

/**
 * Constructs a tensor with the given shape
 *
 * @param shape a vector of unsigned integers, represents the shape of the tensor
 */
Tensor::Tensor(std::vector<uint> shape, bool isOnDevice): data_h(nullptr), data_d(nullptr), isOnDevice(isOnDevice), numElements(1), shape(shape), strides(shape.size(), 1) {
    auto countZeroDims = std::count(shape.begin(), shape.end(), 0);
    if (countZeroDims >= 1) {
        throw std::invalid_argument("Tensor cannot have dimension with size 0.");
    }
    
    numElements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint>());
    if (this->isOnDevice) {
        cudaMalloc((void**)&data_d, numElements * sizeof(float));
        cudaMemset(data_d, 0, numElements * sizeof(float));
    } else {
        data_h = new float[numElements];
        std::fill(data_h, data_h + numElements, 0);
    }
    
    CalculateStrides();
}

Tensor::Tensor(std::vector<float> data, std::vector<uint> shape, bool isOnDevice): data_h(nullptr), data_d(nullptr), isOnDevice(isOnDevice), numElements(1), shape(shape), strides(shape.size(), 1) {
    auto countZeroDims = std::count(shape.begin(), shape.end(), 0);
    if (countZeroDims >= 1) {
        throw std::invalid_argument("Tensor cannot have dimension with size 0.");
    }
    
    numElements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint>());
    if (data.size() != numElements) {
        throw std::invalid_argument("Data and shape are not consistent.");
    }

    if (this->isOnDevice) {
        cudaMalloc((void**)&data_d, numElements * sizeof(float));
        cudaMemcpy(data_d, data.data(), numElements * sizeof(float), cudaMemcpyHostToDevice);
    } else {
        data_h = new float[numElements];
        std::copy(data.begin(), data.end(), data_h);
    }
    
    CalculateStrides();
}

/**
 * Constructs a tensor with the same data as the given tensor
 * 
 * @param other a tensor, represents the tensor to be copied
 */
Tensor::Tensor(const Tensor& other) {
    this->isOnDevice = other.isOnDevice;
    this->numElements = other.numElements;
    this->shape = other.shape;
    this->strides = other.strides;

    if (this->isOnDevice) {
        cudaMalloc((void**)&data_d, this->numElements * sizeof(float));
        cudaMemcpy(data_d, other.data_d, this->numElements * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        data_h = new float[this->numElements];
        cudaMemcpy(data_h, other.data_h, this->numElements * sizeof(float), cudaMemcpyHostToHost);
    }
}

/**
 * Destructs the tensor
 */
Tensor::~Tensor() {
    if (data_h != nullptr)
        delete[] data_h;
    
    if (data_d != nullptr)
        cudaFree(data_d);
}

/**
 * Copies the data from host to device
 */
void Tensor::ToDevice() {
    if (!this->isOnDevice) {
        if (data_d == nullptr) {
            cudaMalloc((void**)&data_d, numElements * sizeof(float));
        }
        cudaMemcpy(data_d, data_h, numElements * sizeof(float), cudaMemcpyHostToDevice);

        this->isOnDevice = true;

        delete []data_h;
        data_h = nullptr;
    }
}

/**
 * Copies the data from device to host
 */
void Tensor::ToHost() {
    if (this->isOnDevice) {
        if (data_h == nullptr) {
            data_h = new float[numElements];
        }
        cudaMemcpy(data_h, data_d, numElements * sizeof(float), cudaMemcpyDeviceToHost);

        this->isOnDevice = false;
        
        cudaFree(data_d);
        data_d = nullptr;
    }
}

/**
 * Calculates the shape of the resulting tensor after performing an broadcasted operation with the given tensor
 *
 * @param other a tensor, represents the tensor to be compared with
 * @return a vector of unsigned integers, represents the shape of the resulting tensor
*/
std::vector<uint> Tensor::CalculateBroadcastResultShape(const Tensor& other) const {
    std::vector<uint> resultShape(std::max(shape.size(), other.shape.size()), 1);
    int i = this->shape.size() - 1, j = other.shape.size() - 1, k = resultShape.size() - 1;

    for (; i >= 0 and j >= 0; i--, j--, k-- ) {
        resultShape[k] = std::max(this->shape[i], other.shape[j]);
    }

    for (; i >= 0; i--, k--) {
        resultShape[k] = shape[i];
    }
    for (; j >= 0; j--, k--) {
        resultShape[k] = other.shape[j];
    }
    return resultShape;
}

/**
 * Validates the compatibility of shape of the tensor with the given tensor
 *
 * @param other a tensor, represents the tensor to be compared with
 */
void Tensor::ValidateShapesBroadcastOperation(const Tensor& other) const {
    int i = this->shape.size() - 1, j = other.shape.size() - 1;

    for (; i >= 0 and j >= 0; i--, j-- ) {
        if (this->shape[i] != other.shape[j] and (this->shape[i] != 1 and other.shape[j] != 1)) {
            throw std::invalid_argument("Shapes are not compatible for addition");
        }
    }
}

void Tensor::ValidateOperationDevice(const Tensor& other) const {
    if (isOnDevice != other.isOnDevice) {
        throw std::invalid_argument("Tensors must be on the same device");
    }
}

std::vector<uint> Tensor::CalculateBroadcastedShape(const uint referenceShapeSize) const {
    std::vector<uint> result(referenceShapeSize, 1);
    std::copy(shape.begin(), shape.end(), result.begin() + (referenceShapeSize - shape.size()));
    return result;
}

std::vector<uint> Tensor::CalculateBroadcastedStrides(const uint referenceShapeSize) const {
    std::vector<uint> result(referenceShapeSize, numElements);
    std::copy(strides.begin(), strides.end(), result.begin() + (referenceShapeSize - shape.size()));
    return result;
}

uint Tensor::CalculateLinearIndex(uint referenceLinearIndex, const std::vector<uint>& referenceStrides) const {
    auto referenceShapeSize = referenceStrides.size();
    return ::CalculateLinearIndex(referenceLinearIndex, referenceStrides.data(), CalculateBroadcastedShape(referenceShapeSize).data(),
                                  CalculateBroadcastedStrides(referenceShapeSize).data(), referenceShapeSize);
}

Tensor Tensor::operator+(const Tensor& other) const {
    ValidateOperationDevice(other);
    ValidateShapesBroadcastOperation(other);

    Tensor result(CalculateBroadcastResultShape(other), isOnDevice);

    if (isOnDevice) {
        LaunchAddKernel(result, *this, other);
    }
    else {
        for (uint i = 0; i < result.numElements; i++) {            
            uint aIndex = this->CalculateLinearIndex(i, result.strides);
            uint bIndex = other.CalculateLinearIndex(i, result.strides);
            
            result.data_h[i] = data_h[aIndex] + other.data_h[bIndex];
        }
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    ValidateOperationDevice(other);
    ValidateShapesBroadcastOperation(other);

    Tensor result(CalculateBroadcastResultShape(other), isOnDevice);

    if (isOnDevice) {
        LaunchSubKernel(result, *this, other);
    }
    else {
        for (uint i = 0; i < result.numElements; i++) {            
            uint aIndex = this->CalculateLinearIndex(i, result.strides);
            uint bIndex = other.CalculateLinearIndex(i, result.strides);
            
            result.data_h[i] = data_h[aIndex] - other.data_h[bIndex];
        }
    }
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    ValidateOperationDevice(other);
    ValidateShapesBroadcastOperation(other);

    Tensor result(CalculateBroadcastResultShape(other), isOnDevice);

    if (isOnDevice) {
        LaunchMulKernel(result, *this, other);
    }
    else {
        for (uint i = 0; i < result.numElements; i++) {            
            uint aIndex = this->CalculateLinearIndex(i, result.strides);
            uint bIndex = other.CalculateLinearIndex(i, result.strides);
            
            result.data_h[i] = data_h[aIndex] * other.data_h[bIndex];
        }
    }
    return result;
}

Tensor Tensor::operator/(const Tensor& other) const {
    ValidateOperationDevice(other);
    ValidateShapesBroadcastOperation(other);

    Tensor result(CalculateBroadcastResultShape(other), isOnDevice);

    if (isOnDevice) {
        LaunchDivKernel(result, *this, other);
    }
    else {
        for (uint i = 0; i < result.numElements; i++) {            
            uint aIndex = this->CalculateLinearIndex(i, result.strides);
            uint bIndex = other.CalculateLinearIndex(i, result.strides);
            
            result.data_h[i] = data_h[aIndex] / other.data_h[bIndex];
        }
    }
    return result;
}