#include<iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main() {

    Mat img = imread("/home/ahsan/Downloads/lena.png");
    imshow("Image", img);
    waitKey(0);
    return 0;
}