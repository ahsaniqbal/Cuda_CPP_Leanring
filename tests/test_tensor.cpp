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

TEST_F(TensorTest, ConstructorThrowsOnZeroDimension) {
    std::vector<uint> shape = {3, 0, 2}; // Shape with a dimension of size 0
    EXPECT_THROW(Tensor tensor(shape, false), std::invalid_argument);
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

TEST_F(TensorTest, TestData) {
    std::vector<uint> shape = {2, 3, 4};
    Tensor tensor(shape, false);
    
    EXPECT_TRUE(tensor.GetData() != nullptr);
    tensor.ToDevice();
    EXPECT_TRUE(tensor.GetData() != nullptr);
    
    Tensor tensor1(shape, true);
    EXPECT_TRUE(tensor1.GetData() != nullptr);
    tensor1.ToHost();
    EXPECT_TRUE(tensor1.GetData() != nullptr);
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

}