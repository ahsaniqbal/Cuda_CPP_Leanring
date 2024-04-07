#include <gtest/gtest.h>
#include "tensor.h" // Assuming this is where your Tensor class is defined

class TensorTest : public ::testing::Test {
protected:
    Tensor* tensorPtr;

    void SetUp() override {
        // Initialize tensor here if necessary
    }

    void TearDown() override {
        // Clean up after each test if necessary
    }
};

TEST_F(TensorTest, ConstructorThrowsOnZeroDimension) {
    std::vector<uint> shape = {3, 0, 2}; // Shape with a dimension of size 0
    EXPECT_THROW(Tensor tensor(shape, false), std::invalid_argument);
}

TEST_F(TensorTest, TestShape) {
    std::vector<uint> expectedShape = {1, 2, 3};
    tensorPtr = new Tensor(expectedShape, false);
    EXPECT_EQ(tensorPtr->GetShape(), expectedShape);
    delete tensorPtr;
}

TEST_F(TensorTest, TestNumElements) {
    std::vector<uint> shape = {2, 3, 4};
    tensorPtr = new Tensor(shape, false);
    EXPECT_EQ(tensorPtr->GetNumElements(), 2*3*4);
    delete tensorPtr;
}

TEST_F(TensorTest, TestStrides) {
    std::vector<uint> shape = {2, 3, 4};
    tensorPtr = new Tensor(shape, false);
    std::vector<uint> expectedStrides = {12, 4, 1};
    EXPECT_EQ(tensorPtr->GetStrides(), expectedStrides);
    delete tensorPtr;
}