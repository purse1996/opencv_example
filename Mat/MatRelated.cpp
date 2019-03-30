//
// Created by bywind on 19-3-30.
//
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
using  namespace cv;
using namespace std;

int main()
{
    /*　Mat矩阵初始化　*/
    // 1. 直接初始化
    Mat mat1(2, 2, CV_32FC3, Scalar(0, 0, 255));
    // 2. 利用特殊函数初始化
    Mat mat2 = Mat::eye(3, 3, CV_64F);
    //　3. 较小的ｍａｔ矩阵初始化
    Mat mat3 = (Mat_<double>(3, 3) << 0, 2, 4, 9, 1, 2, 4, 9, 0);
    // 4. 浅拷贝，共享指针
    Mat mat4 = mat3;
    //5. 新建一个头指针，拷贝初始化
    Mat mat5_1 = mat1.row(0).clone();
    Mat mat5_2;
    mat1.row(0).copyTo(mat5_2);

    /*　访问Mat矩阵中的元素　*/
    // 1. 利用Mat行列指针访问
    int nRows = mat1.rows;
    int nCols = mat1.cols * mat1.channels();
    float *p; // 指针类型要与Mat矩阵类型相同
    for(int i = 0; i < nRows; i++)
    {
        p = mat1.ptr<float>(i);
        for(int j = 0; j < nCols; j++)
        {
            // 这里只能赋值　不能修改吗？？？
            cout << p[j] << endl;
        }
    }

    // 2. 使用at<>访问矩阵元素
    for(int y = 0; y < mat1.rows; y++)
    {
        for(int x = 0; x < mat1.cols; x++)
        {
            cout << mat1.at<Vec3f>(y, x)[0] << mat1.at<Vec3f>(y, x)[1] << mat1.at<Vec3f>(y, x)[2] << endl;
        }
    }

    // 3. 使用迭代器访问
    MatIterator_<Vec3f> itm, itmEnd;
    for( itm = mat1.begin<Vec3f>(), itmEnd = mat1.end<Vec3f>(); itm != itmEnd; itm++)
    {
        cout << (*itm)[0] << "..." << (*itm)[1] << "..." << (*itm)[2] << endl;
    }

    return 0;
}




