//
// Created by byiwind on 19-2-21.
//

#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
// #include <opencv2/nofree/nofree.hpp>
#include<opencv2/xfeatures2d.hpp>


using namespace std;
using namespace cv;
using namespace cv::detail;

int main(int argc, char** argv)
{
    Mat img = imread("2_new.jpg");    //读入图像

    Ptr<FeaturesFinder> finder;    //定义FeaturesFinder类

    finder = new SurfFeaturesFinder();    //应用SURF方法
    //finder = new OrbFeaturesFinder();    //应用ORB方法

    ImageFeatures features;    //表示特征

    (*finder)(img, features);    //特征检测

    Mat output_img;
    //调用drawKeypoints函数绘制特征
    drawKeypoints(img, features.keypoints, output_img, Scalar::all(-1));

    namedWindow("features");
    imshow("features", output_img);

    waitKey(0);
    imwrite("2_feature.jpg",output_img);

    return 0;
}