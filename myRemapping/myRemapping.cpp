#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    Mat img = imread(argv[1]);
    Mat map_x(img.size(), CV_32FC1);
    Mat map_y(img.size(), CV_32FC1);
    //map_x和map_y分别存储的是水平和竖直方向的变换矩阵
    Mat result;
    for(int i=0; i<img.rows; i++)
    {
        for(int j=0; j<img.cols; j++)
        {
                //做上下颠倒 故水平坐标不变 竖直坐标翻转
            map_y.at<float>(i, j) = (float)(img.rows - i);
            map_x.at<float>(i, j) = (float)(j);
        }
    }

    remap(img, result, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
    imshow("upside down", result);
    waitKey(0);
    return 0;
}
