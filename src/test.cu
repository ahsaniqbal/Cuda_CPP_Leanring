#include<iostream>
#include<cuda_runtime.h>

__global__
void VectorAdd(float* vector1, float* vector2, float* result, float scaler1, float scaler2, uint numElements){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numElements) {
        result[index] = scaler1 * vector1[index] + scaler2 * vector2[index];
    }
}

/*/__global__
//void ImageColorToIntensity() {}

float MaxError(float *x, float maxError, uint numElements) {
    for (uint32_t i = 0; i < numElements; i++) {
        maxError = fmax(maxError, fabs(x[i] - 5.5f));
    }
    return maxError;
}

int main() {
    //size_t n = 1 << 20;
    uint numElements = 100;

    float *x_d, *y_d, *z_d, *x, *y, *z;

    //cudaMallocManaged(&x_d, n * sizeof(float));
    //cudaMallocManaged(&y_d, n * sizeof(float));
    //cudaMallocManaged(&z_d, n * sizeof(float));
    cudaMalloc((void**)&x_d, numElements * sizeof(float));
    cudaMalloc((void**)&y_d, numElements * sizeof(float));
    cudaMalloc((void**)&z_d, numElements * sizeof(float));

    x = (float*)malloc(numElements * sizeof(float));
    y = (float*)malloc(numElements * sizeof(float));
    z = (float*)malloc(numElements * sizeof(float));

    Init(x, 1.0f, numElements);
    Init(y, 2.0f, numElements);


    cudaMemcpy(x_d, x, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, numElements * sizeof(float), cudaMemcpyHostToDevice);

    VectorAdd<<<ceil(numElements / 256.0), 256>>>(x_d, y_d, z_d, 1.5, 2, numElements);
    cudaDeviceSynchronize();

    cudaMemcpy(z, z_d, numElements * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout<<MaxError(z, 0.0f, numElements)<<std::endl;

    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);

    free(x);
    free(y);
    free(z);

    Mat img = imread("/home/ahsan/Downloads/lena.png");
    imshow("Image", img);
    waitKey(0);
    return 0;
}*/