//
// Created by byiwind on 19-2-23.
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

bool readCamera(const string& filename, Mat& cameraMatrix, Mat& distCoeffs,  float& ratio);
int main()
{
    Mat intrinsic_matrix, distortion_coeffs;
    float ratio;
    float cutRatio = 0.2;
    ifstream fin_image("image");
    bool need_calib = readCamera("calibra", intrinsic_matrix, distortion_coeffs, ratio);
    Mat undistort_xmap, undistort_ymap;
    int count_image = 0;
    string imageName;
    Mat R = Mat::eye(3,3, CV_32F);
    while(getline(fin_image, imageName))
    {
        cout << "TestImage ..." << imageName << endl;
        Mat distort_img = imread(imageName);
        Size imageSize = distort_img.size();
        Mat undistort_img;
        Mat intrinsic_mat(intrinsic_matrix), new_intrinsic_mat;

        intrinsic_mat.copyTo(new_intrinsic_mat);
        //调节视场大小,乘的系数越小视场越大
        new_intrinsic_mat.at<double>(0, 0) *= ratio;
        new_intrinsic_mat.at<double>(1, 1) *= ratio;
        //调节校正图中心，建议置于校正图中心
        new_intrinsic_mat.at<double>(0, 2) = ratio * distort_img.cols;
        new_intrinsic_mat.at<double>(1, 2) = ratio * distort_img.rows;

//        fisheye::undistortImage(distort_img, undistort_img, intrinsic_matrix, distortion_coeffs, new_intrinsic_mat);
//        imwrite("output/" + to_string(count_image++) + ".jpg", undistort_img);
        fisheye::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, new_intrinsic_mat, imageSize,CV_32FC1,undistort_xmap,undistort_ymap);

        undistort_xmap(Rect(0, undistort_xmap.rows/2, (int)(cutRatio*undistort_xmap.cols), undistort_xmap.rows - undistort_xmap.rows/2)) = undistort_xmap.at<int>(0, undistort_xmap.rows/2);
        int x_right = undistort_xmap.cols - (int)(cutRatio*undistort_xmap.cols);
        undistort_xmap(Rect(x_right, undistort_xmap.rows/2, (int)(cutRatio*undistort_xmap.cols), undistort_xmap.rows - undistort_xmap.rows/2)) = undistort_xmap.at<int>(0, undistort_xmap.rows/2);

        undistort_ymap(Rect(0, undistort_ymap.rows/2, (int)(cutRatio*undistort_ymap.cols), undistort_ymap.rows - undistort_ymap.rows/2)) = undistort_ymap.at<int>(0, undistort_ymap.rows/2);
        undistort_ymap(Rect(x_right, undistort_ymap.rows/2, (int)(cutRatio*undistort_ymap.cols), undistort_ymap.rows - undistort_ymap.rows/2)) = undistort_ymap.at<int>(0, undistort_ymap.rows/2);

        remap(distort_img, distort_img, undistort_xmap, undistort_ymap, INTER_LINEAR, BORDER_CONSTANT);
        imwrite("output/"+to_string(count_image++)+".jpg", distort_img);
        // 设置undistort_xmap undistort_ymap；
//        capture.resetWarp(camId[i]);
//        // 分别对undistort_xmap undistort_ymap进行映射
//        capture.setWarp(camId[i], undistort_xmap[i], undistort_ymap[i], BORDER_CONSTANT);
    }
}

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
