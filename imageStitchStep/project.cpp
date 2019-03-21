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

    vector<Mat> imgs;    //输入图像
    Mat img = imread("1_new.jpg");
    imgs.push_back(img);
    img = imread("2_new.jpg");
    imgs.push_back(img);

    Ptr<FeaturesFinder> finder;    //特征检测
    finder = new SurfFeaturesFinder();
    vector<ImageFeatures> features(2);
    (*finder)(imgs[0], features[0]);
    (*finder)(imgs[1], features[1]);

    vector<MatchesInfo> pairwise_matches;    //特征匹配
    BestOf2NearestMatcher matcher(false, 0.3f, 6, 6);
    matcher(features, pairwise_matches);

    HomographyBasedEstimator estimator;    //相机参数评估
    vector<CameraParams> cameras;
    estimator(features, pairwise_matches, cameras);
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
    }

    Ptr<detail::BundleAdjusterBase> adjuster;    //光束平差法，精确相机参数
    adjuster = new detail::BundleAdjusterReproj();
    adjuster->setConfThresh(1);
    (*adjuster)(features, pairwise_matches, cameras);

    vector<Mat> rmats;
    for (size_t i = 0; i < cameras.size(); ++i)
        rmats.push_back(cameras[i].R.clone());
    waveCorrect(rmats, WAVE_CORRECT_HORIZ);    //波形校正
    for (size_t i = 0; i < cameras.size(); ++i)
        cameras[i].R = rmats[i];

    vector<Point> corners(2);    //表示映射变换后图像的左上角坐标
    vector<Mat> masks_warped(2);    //表示映射变换后的图像掩码
    vector<Mat> images_warped(2);    //表示映射变换后的图像
    vector<Size> sizes(2);    //表示映射变换后的图像尺寸
    vector<Mat> masks(2);    //表示源图的掩码

    for (int i = 0; i < 2; ++i)    //初始化源图的掩码
    {
        masks[i].create(imgs[i].size(), CV_8U);    //定义尺寸大小
        masks[i].setTo(Scalar::all(255));    //全部赋值为255，表示源图的所有区域都使用
    }

    Ptr<WarperCreator> warper_creator;    //定义图像映射变换创造器
    //warper_creator = new cv::PlaneWarper();    //平面投影
    //warper_creator = new cv::CylindricalWarper();    //柱面投影
    warper_creator = new cv::SphericalWarper();    //球面投影
    //warper_creator = new cv::FisheyeWarper();    //鱼眼投影
    //warper_creator = new cv::StereographicWarper();    //立方体投影

    //定义图像映射变换器，设置映射的尺度为相机的焦距，所有相机的焦距都相同
    Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>((cameras[0].focal+cameras[1].focal)/2));
    for (int i = 0; i < 2; ++i)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);    //转换相机内参数的数据类型
        //对当前图像镜像投影变换，得到变换后的图像以及该图像的左上角坐标
        corners[i] = warper->warp(imgs[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();    //得到尺寸
        //得到变换后的图像掩码
        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }

    //通过掩码，只得到映射变换后的图像
    for(int k =0;k<2;k++)
    {
        for(int i=0;i<sizes[k].height;i++)
        {
            for(int j=0;j<sizes[k].width;j++)
            {
                if(masks_warped[k].at<uchar>(i, j)==0)    //掩码
                {
                    images_warped[k].at<Vec3b>(i, j)[0]=0;
                    images_warped[k].at<Vec3b>(i, j)[1]=0;
                    images_warped[k].at<Vec3b>(i, j)[2]=0;
                }
            }
        }
    }

    imwrite("warp1.jpg", images_warped[0]);
    imwrite("warp2.jpg", images_warped[1]);

    return 0;
}
