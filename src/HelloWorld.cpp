// Copyright © 2024 Ahsan Iqbal
#include<iostream>
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat img = cv::imread("/home/ahsan/Downloads/lena.png");
    cv::imshow("Image", img);
    cv::waitKey(0);
    return 0;
}
