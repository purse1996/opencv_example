//
// Created by byiwind on 19-2-21.
//

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <fstream>
#include <string>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::detail;

bool readCamera(const string& filename, Mat& cameraMatrix, Mat& distCoeffs,  float& ratio)
{
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs["ratio"] >> ratio;
    return true;
}

int main() {
    float ratio = 0.5;
    Mat distort_img1, distort_img2, undistort_img, cameraMatrix, distCoeffs, new_intrinsic_mat;
    distort_img1 = imread("../1.jpg");

    distort_img2 = imread("../2.jpg");
    bool need_calib = readCamera("../calibra", cameraMatrix, distCoeffs, ratio);
//bool need_calib = readCamera(parser.get<string>("calib"), cameraMatrix[i], distCoeffs[i], ratio);

    Mat intrinsic_mat(cameraMatrix);

    intrinsic_mat.copyTo(new_intrinsic_mat);
//调节视场大小,乘的系数越小视场越大
    new_intrinsic_mat.at<double>(0, 0) *= ratio;
    new_intrinsic_mat.at<double>(1, 1) *= ratio;
//调节校正图中心，建议置于校正图中心
    new_intrinsic_mat.at<double>(0, 2) = ratio * distort_img1.cols;
    new_intrinsic_mat.at<double>(1, 2) = ratio * distort_img1.rows;

//fisheye::undistortImage(distort_img, undistort_img, intrinsic_matrix, distortion_coeffs, new_intrinsic_mat);


    fisheye::undistortImage(distort_img1, undistort_img, cameraMatrix, distCoeffs, new_intrinsic_mat);
    imwrite( "1_new.jpg", undistort_img);
    fisheye::undistortImage(distort_img2, undistort_img, cameraMatrix, distCoeffs, new_intrinsic_mat);
    imwrite( "2_new.jpg", undistort_img);



}