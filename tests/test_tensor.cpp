#include <gtest/gtest.h>
#include "tensor.h" // Assuming this is where your Tensor class is defined

class TensorTest : public ::testing::Test {
protected:


    void SetUp() override {
        // Initialize tensor here if necessary
    }

    void TearDown() override {
        // Clean up after each test if necessary
    }
};

TEST_F(TensorTest, ConstructorWithShapeThrowsOnZeroDimension) {
    EXPECT_THROW(Tensor tensor({0, 1, 2}, false), std::invalid_argument);
    EXPECT_THROW(Tensor tensor({0, 1, 2}, true), std::invalid_argument);

    EXPECT_THROW(Tensor tensor({3, 0, 2}, false), std::invalid_argument);
    EXPECT_THROW(Tensor tensor({3, 0, 2}, true), std::invalid_argument);

    EXPECT_THROW(Tensor tensor({3, 1, 0}, false), std::invalid_argument);
    EXPECT_THROW(Tensor tensor({3, 1, 0}, true), std::invalid_argument);
}

TEST_F(TensorTest, ConstructorWithShapeOnHost) {
    std::vector<uint> shape = {2, 3, 4};
    Tensor tensor(shape, false);

    EXPECT_EQ(tensor.IsOnDevice(), false);
    EXPECT_EQ(tensor.GetDeviceData(), nullptr);
    ASSERT_NE(tensor.GetHostData(), nullptr);
    ASSERT_NE(tensor.GetData(), nullptr);
    EXPECT_EQ(std::count(tensor.GetData(), tensor.GetData() + tensor.GetNumElements(), 0), tensor.GetNumElements());
}

TEST_F(TensorTest, ConstructorWithShapeOnDevice) {
    std::vector<uint> shape = {2, 3, 4};
    Tensor tensor(shape, true);

    EXPECT_EQ(tensor.IsOnDevice(), true);
    EXPECT_EQ(tensor.GetHostData(), nullptr);
    ASSERT_NE(tensor.GetDeviceData(), nullptr);
    ASSERT_NE(tensor.GetData(), nullptr);

    auto device_data = tensor.GetData();
    float* host_data = new float[tensor.GetNumElements()];
    cudaMemcpy(host_data, device_data, tensor.GetNumElements() * sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(std::count(host_data, host_data + tensor.GetNumElements(), 0), tensor.GetNumElements());
    delete[] host_data;
}

TEST_F(TensorTest, ConstructorWithDataAndShapeThrowsOnZeroDimension) {
    EXPECT_THROW(Tensor tensor({1, 2, 3}, {0, 1, 2}, false), std::invalid_argument);
    EXPECT_THROW(Tensor tensor({1, 2, 3}, {0, 1, 2}, true), std::invalid_argument);

    EXPECT_THROW(Tensor tensor({1, 2, 3}, {3, 0, 2}, false), std::invalid_argument);
    EXPECT_THROW(Tensor tensor({1, 2, 3}, {3, 0, 2}, true), std::invalid_argument);

    EXPECT_THROW(Tensor tensor({1, 2, 3}, {3, 1, 0}, false), std::invalid_argument);
    EXPECT_THROW(Tensor tensor({1, 2, 3}, {3, 1, 0}, true), std::invalid_argument);
}

TEST_F(TensorTest, ConstructorWithDataAndShapeThrowsOnInconsistentDataAndShape) {
    EXPECT_THROW(Tensor tensor({1, 2, 3}, {1, 2, 1}, false), std::invalid_argument);
    EXPECT_THROW(Tensor tensor({1, 2, 3}, {2, 1, 1}, true), std::invalid_argument);

    EXPECT_THROW(Tensor tensor({1, 2, 3}, {3, 2, 2}, false), std::invalid_argument);
    EXPECT_THROW(Tensor tensor({1, 2, 3}, {3, 1, 2}, true), std::invalid_argument);

    EXPECT_THROW(Tensor tensor({1, 2, 3, 4, 5}, {1}, false), std::invalid_argument);
    EXPECT_THROW(Tensor tensor({1}, {3, 1, 1}, true), std::invalid_argument);
}

TEST_F(TensorTest, ConstructorWithDataAndShapeOnHost) {
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    std::vector<uint> shape = {2, 3};
    Tensor tensor(data, shape, false);

    EXPECT_EQ(tensor.IsOnDevice(), false);
    EXPECT_EQ(tensor.GetDeviceData(), nullptr);
    ASSERT_NE(tensor.GetHostData(), nullptr);
    ASSERT_NE(tensor.GetData(), nullptr);
    EXPECT_EQ(std::equal(data.begin(), data.end(), tensor.GetData()), true);
}

TEST_F(TensorTest, ConstructorWithDataAndShapeOnDevice) {
    std::vector<float> data = {1, 2, 3, 4, 5, 6};
    std::vector<uint> shape = {2, 3};
    Tensor tensor(data, shape, true);

    EXPECT_EQ(tensor.IsOnDevice(), true);
    EXPECT_EQ(tensor.GetHostData(), nullptr);
    ASSERT_NE(tensor.GetDeviceData(), nullptr);
    ASSERT_NE(tensor.GetData(), nullptr);

    auto device_data = tensor.GetData();
    float* host_data = new float[tensor.GetNumElements()];
    cudaMemcpy(host_data, device_data, tensor.GetNumElements() * sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(std::equal(data.begin(), data.end(), host_data), true);
    delete[] host_data;
}

TEST_F(TensorTest, TestShape) {
    std::vector<uint> expectedShape = {1, 2, 3};
    EXPECT_EQ(Tensor(expectedShape, false).GetShape(), expectedShape);
}

TEST_F(TensorTest, TestNumElements) {
    std::vector<uint> shape = {2, 3, 4};
    EXPECT_EQ(Tensor(shape, false).GetNumElements(), 2*3*4);
}

TEST_F(TensorTest, TestStrides) {
    std::vector<uint> shape = {2, 3, 4};
    std::vector<uint> expectedStrides = {12, 4, 1};
    EXPECT_EQ(Tensor(shape, false).GetStrides(), expectedStrides);
}

TEST_F(TensorTest, TestToDevice) {
    std::vector<uint> shape = {2, 3, 4};
    Tensor tensor(shape, false);
    
    EXPECT_TRUE(tensor.IsOnDevice() == false);
    EXPECT_TRUE(tensor.GetHostData() != nullptr);
    EXPECT_TRUE(tensor.GetDeviceData() == nullptr);
    EXPECT_TRUE(tensor.GetData() != nullptr);
    
    tensor.ToDevice();
    EXPECT_TRUE(tensor.IsOnDevice() == true);
    EXPECT_TRUE(tensor.GetHostData() == nullptr);
    EXPECT_TRUE(tensor.GetDeviceData() != nullptr);
    EXPECT_TRUE(tensor.GetData() != nullptr);
}

TEST_F(TensorTest, TestToHost) {
    std::vector<uint> shape = {2, 3, 4};
    Tensor tensor(shape, true);
    
    EXPECT_TRUE(tensor.IsOnDevice() == true);
    EXPECT_TRUE(tensor.GetHostData() == nullptr);
    EXPECT_TRUE(tensor.GetDeviceData() != nullptr);
    EXPECT_TRUE(tensor.GetData() != nullptr);
    
    tensor.ToHost();
    EXPECT_TRUE(tensor.IsOnDevice() == false);
    EXPECT_TRUE(tensor.GetHostData() != nullptr);
    EXPECT_TRUE(tensor.GetDeviceData() == nullptr);
    EXPECT_TRUE(tensor.GetData() != nullptr);
}

TEST_F(TensorTest, TestValidateShapesBroadcastOperation) {
    EXPECT_NO_THROW(Tensor({2, 3, 4}, false).ValidateShapesBroadcastOperation(Tensor({4}, false)));
    EXPECT_NO_THROW(Tensor({2, 3, 4}, false).ValidateShapesBroadcastOperation(Tensor({1, 4}, false)));
    EXPECT_NO_THROW(Tensor({2, 3, 4}, false).ValidateShapesBroadcastOperation(Tensor({1, 1, 1, 1, 4}, false)));

    EXPECT_NO_THROW(Tensor({2, 3, 4}, false).ValidateShapesBroadcastOperation(Tensor({1, 3, 1}, false)));
    EXPECT_NO_THROW(Tensor({2, 3, 4}, false).ValidateShapesBroadcastOperation(Tensor({2, 1, 1}, false)));
    EXPECT_NO_THROW(Tensor({2, 3, 4}, false).ValidateShapesBroadcastOperation(Tensor({2, 3, 1}, false)));
    EXPECT_NO_THROW(Tensor({2, 3, 4}, false).ValidateShapesBroadcastOperation(Tensor({2, 3, 4}, false)));
    EXPECT_NO_THROW(Tensor({2, 3, 4}, false).ValidateShapesBroadcastOperation(Tensor({1,1,1,1,1,1}, false)));


    EXPECT_THROW(Tensor({2, 3, 4}, false).ValidateShapesBroadcastOperation(Tensor({2, 3, 5}, false)), std::invalid_argument);
    EXPECT_THROW(Tensor({2, 3, 4}, false).ValidateShapesBroadcastOperation(Tensor({2, 6, 4}, false)), std::invalid_argument);
    EXPECT_THROW(Tensor({2, 3, 4}, false).ValidateShapesBroadcastOperation(Tensor({8, 3, 4}, false)), std::invalid_argument);
    EXPECT_THROW(Tensor({2, 3, 4}, false).ValidateShapesBroadcastOperation(Tensor({4, 1, 1}, false)), std::invalid_argument);
}

TEST_F(TensorTest, TestCalculateBroadcastResultShape) {
    EXPECT_EQ(Tensor({2, 4, 6, 4}, false).CalculateBroadcastResultShape(Tensor({4}, false)), std::vector<uint>({2,4,6,4}));
    EXPECT_EQ(Tensor({2, 4, 6, 4}, false).CalculateBroadcastResultShape(Tensor({1, 4, 1, 4}, false)), std::vector<uint>({2,4,6,4}));
    EXPECT_EQ(Tensor({1}, false).CalculateBroadcastResultShape(Tensor({4}, false)), std::vector<uint>({4}));
}

TEST_F(TensorTest, CalculateBroadcastedShape) {
    EXPECT_EQ(Tensor({4}, false).CalculateBroadcastedShape(4), std::vector<uint>({1,1,1,4}));   
    EXPECT_EQ(Tensor({4,4}, false).CalculateBroadcastedShape(2), std::vector<uint>({4,4}));
}

TEST_F(TensorTest, CalculateBroadcastedStrides) {
    EXPECT_EQ(Tensor({4}, false).CalculateBroadcastedStrides(4), std::vector<uint>({4,4,4,1}));
}

TEST_F(TensorTest, CalculateLinearIndex) {
    Tensor tensor1({4}, false);
    
    EXPECT_EQ(tensor1.CalculateLinearIndex(0, {12, 4, 1}), 0);
    EXPECT_EQ(tensor1.CalculateLinearIndex(5, {12, 4, 1}), 1);
    EXPECT_EQ(tensor1.CalculateLinearIndex(10, {12, 4, 1}), 2);
    EXPECT_EQ(tensor1.CalculateLinearIndex(15, {12, 4, 1}), 3);


    Tensor tensor2({3, 1}, false);
    EXPECT_EQ(tensor2.CalculateLinearIndex(0, {12, 4, 1}), 0);
    EXPECT_EQ(tensor2.CalculateLinearIndex(7, {12, 4, 1}), 1);
    EXPECT_EQ(tensor2.CalculateLinearIndex(10, {12, 4, 1}), 2);
    EXPECT_EQ(tensor2.CalculateLinearIndex(15, {12, 4, 1}), 0);
    EXPECT_EQ(tensor2.CalculateLinearIndex(17, {12, 4, 1}), 1);
    EXPECT_EQ(tensor2.CalculateLinearIndex(22, {12, 4, 1}), 2);

    Tensor tensor3({2, 1, 1}, false);
    EXPECT_EQ(tensor3.CalculateLinearIndex(5, {12, 4, 1}), 0);
    EXPECT_EQ(tensor3.CalculateLinearIndex(21, {12, 4, 1}), 1);

    Tensor tensor4({2, 1, 4}, false);
    EXPECT_EQ(tensor4.CalculateLinearIndex(8, {12, 4, 1}), 0);
    EXPECT_EQ(tensor4.CalculateLinearIndex(5, {12, 4, 1}), 1);
    EXPECT_EQ(tensor4.CalculateLinearIndex(10, {12, 4, 1}), 2);
    EXPECT_EQ(tensor4.CalculateLinearIndex(3, {12, 4, 1}), 3);

    EXPECT_EQ(tensor4.CalculateLinearIndex(12, {12, 4, 1}), 4);
    EXPECT_EQ(tensor4.CalculateLinearIndex(17, {12, 4, 1}), 5);
    EXPECT_EQ(tensor4.CalculateLinearIndex(15, {12, 4, 1}), 7);
    EXPECT_EQ(tensor4.CalculateLinearIndex(22, {12, 4, 1}), 6);
}

void TestBroadcastOperationHelper(const std::vector<float>& data1, const std::vector<uint>& shape1,
                        const std::vector<float>& data2, const std::vector<uint>& shape2,
                        const std::vector<uint>& expectedResultShape, const std::vector<uint> &expectedResultStrides,
                        const std::vector<float>& expectedResult, const bool isOnDevice) {
    
    Tensor tensor1(data1, shape1, isOnDevice);
    Tensor tensor2(data2, shape2, isOnDevice);
    Tensor result = tensor1 + tensor2;

    EXPECT_EQ(result.GetShape(), expectedResultShape);
    EXPECT_EQ(result.GetNumElements(), expectedResult.size());
    EXPECT_EQ(result.GetStrides(), expectedResultStrides);
    EXPECT_EQ(result.IsOnDevice(), isOnDevice);
    ASSERT_NE(result.GetData(), nullptr);
    
    if (isOnDevice) {
        ASSERT_NE(result.GetDeviceData(), nullptr);
        ASSERT_EQ(result.GetHostData(), nullptr);

        auto device_data = result.GetData();
        float* host_data = new float[result.GetNumElements()];
        cudaMemcpy(host_data, device_data, result.GetNumElements() * sizeof(float), cudaMemcpyDeviceToHost);
        
        EXPECT_EQ(std::equal(expectedResult.begin(), expectedResult.end(), host_data), true);
        delete[] host_data;
    } else {
        ASSERT_NE(result.GetHostData(), nullptr);
        ASSERT_EQ(result.GetDeviceData(), nullptr);

        EXPECT_EQ(std::equal(expectedResult.begin(), expectedResult.end(), result.GetData()), true);
    }
    
    
}

TEST_F(TensorTest, TestAdditionHost) {
    TestBroadcastOperationHelper({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {3, 2, 2},
                       {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {3, 2, 2},
                       {3, 2, 2}, {4, 2, 1}, {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24}, false);

    TestBroadcastOperationHelper({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}, {3, 4, 2},
                       {11, 12}, {2}, {3, 4, 2}, {8, 2, 1},
                       {12, 14, 14, 16, 16, 18, 18, 20, 20, 22, 22, 24, 24, 26, 26, 28, 28, 30, 30, 32, 32, 34, 34, 36}, false);

    TestBroadcastOperationHelper({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}, {3, 4, 2},
                       {11, 12, 0, -1}, {4, 1}, {3, 4, 2}, {8, 2, 1},
                       {12, 13, 15, 16, 5, 6, 6, 7, 20, 21, 23, 24, 13, 14, 14, 15, 28, 29, 31, 32, 21, 22, 22, 23}, false);

    TestBroadcastOperationHelper({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}, {3, 4, 2},
                       {-3, 9, -8}, {3, 1, 1}, {3, 4, 2}, {8, 2, 1},
                       {-2, -1, 0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23, 24, 25, 9, 10, 11, 12, 13, 14, 15, 16}, false);

    TestBroadcastOperationHelper({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}, {3, 4, 2},
                       {10, -8, 12, 0, -12, 1, 2, 17}, {4, 2}, {3, 4, 2}, {8, 2, 1},
                       {11, -6, 15, 4, -7, 7, 9, 25, 19, 2, 23, 12, 1, 15, 17, 33, 27, 10, 31, 20, 9, 23, 25, 41}, false);

    TestBroadcastOperationHelper({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}, {3, 4, 2},
                       {-6, 18, 23, -29, 7, 19}, {3, 1, 2}, {3, 4, 2}, {8, 2, 1},
                       {-5, 20, -3, 22, -1, 24, 1, 26, 32, -19, 34, -17, 36, -15, 38, -13, 24, 37, 26, 39, 28, 41, 30, 43}, false);
}

TEST_F(TensorTest, TestAdditionDevice) {
    TestBroadcastOperationHelper({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {3, 2, 2},
                       {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {3, 2, 2},
                       {3, 2, 2}, {4, 2, 1}, {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24}, true);

    TestBroadcastOperationHelper({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}, {3, 4, 2},
                       {11, 12}, {2}, {3, 4, 2}, {8, 2, 1},
                       {12, 14, 14, 16, 16, 18, 18, 20, 20, 22, 22, 24, 24, 26, 26, 28, 28, 30, 30, 32, 32, 34, 34, 36}, true);

    TestBroadcastOperationHelper({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}, {3, 4, 2},
                       {11, 12, 0, -1}, {4, 1}, {3, 4, 2}, {8, 2, 1},
                       {12, 13, 15, 16, 5, 6, 6, 7, 20, 21, 23, 24, 13, 14, 14, 15, 28, 29, 31, 32, 21, 22, 22, 23}, true);

    TestBroadcastOperationHelper({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}, {3, 4, 2},
                       {-3, 9, -8}, {3, 1, 1}, {3, 4, 2}, {8, 2, 1},
                       {-2, -1, 0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23, 24, 25, 9, 10, 11, 12, 13, 14, 15, 16}, true);

    TestBroadcastOperationHelper({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}, {3, 4, 2},
                       {10, -8, 12, 0, -12, 1, 2, 17}, {4, 2}, {3, 4, 2}, {8, 2, 1},
                       {11, -6, 15, 4, -7, 7, 9, 25, 19, 2, 23, 12, 1, 15, 17, 33, 27, 10, 31, 20, 9, 23, 25, 41}, true);

    TestBroadcastOperationHelper({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}, {3, 4, 2},
                       {-6, 18, 23, -29, 7, 19}, {3, 1, 2}, {3, 4, 2}, {8, 2, 1},
                       {-5, 20, -3, 22, -1, 24, 1, 26, 32, -19, 34, -17, 36, -15, 38, -13, 24, 37, 26, 39, 28, 41, 30, 43}, true);
}