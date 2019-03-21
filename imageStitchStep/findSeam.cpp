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

int main(int argc, char** argv) {
    vector <Mat> imgs;    //输入图像
    Mat img = imread("1_new.jpg");
    imgs.push_back(img);
    img = imread("2_new.jpg");
    imgs.push_back(img);

    Ptr <FeaturesFinder> finder;    //特征检测
    finder = new SurfFeaturesFinder();
    vector <ImageFeatures> features(2);
    (*finder)(imgs[0], features[0]);
    (*finder)(imgs[1], features[1]);

    vector <MatchesInfo> pairwise_matches;    //特征匹配
    BestOf2NearestMatcher matcher(false, 0.3f, 6, 6);
    matcher(features, pairwise_matches);

    HomographyBasedEstimator estimator;    //相机参数评估
    vector <CameraParams> cameras;
    estimator(features, pairwise_matches, cameras);
    for (size_t i = 0; i < cameras.size(); ++i) {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
    }

    Ptr <detail::BundleAdjusterBase> adjuster;    //光束平差法，精确化相机参数
    adjuster = new detail::BundleAdjusterReproj();
    adjuster->setConfThresh(1);
    (*adjuster)(features, pairwise_matches, cameras);

    vector <Mat> rmats;
    for (size_t i = 0; i < cameras.size(); ++i)
        rmats.push_back(cameras[i].R.clone());
    waveCorrect(rmats, WAVE_CORRECT_HORIZ);    //波形校正
    for (size_t i = 0; i < cameras.size(); ++i)
        cameras[i].R = rmats[i];

    //图像映射变换
    vector <Point> corners(2);
    vector <UMat> masks_warped(2);
    vector <UMat> images_warped(2);
    vector <UMat> masks(2);
    for (int i = 0; i < 2; ++i) {
        masks[i].create(imgs[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }
    Ptr <WarperCreator> warper_creator;

    //warper_creator = new cv::StereographicWarper();
    warper_creator = new cv::SphericalWarper();
    //warper_creator = new cv::PlaneWarper();
    //warper_creator = new cv::CylindricalWarper();
    //warper_creator = new cv::FisheyeWarper();

    vector<Size> sizes(2);
    Ptr <RotationWarper> warper = warper_creator->create(static_cast<float>(cameras[0].focal));
    for (int i = 0; i < 2; ++i) {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        corners[i] = warper->warp(imgs[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();
        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }


    vector<Mat> mask_origin(2);
    masks_warped[0].copyTo(mask_origin[0]);
    masks_warped[1].copyTo(mask_origin[1]);

    //曝光补偿
//    Ptr <ExposureCompensator> compensator =
//            ExposureCompensator::createDefault(ExposureCompensator::GAIN);
//    compensator->feed(corners, images_warped, masks_warped);
//    for (int i = 0; i < 2; ++i) {
//        compensator->apply(i, corners[i], images_warped[i], masks_warped[i]);
//    }

    Ptr <SeamFinder> seam_finder;    //定义接缝线寻找器
    //seam_finder = new NoSeamFinder();    //无需寻找接缝线
    //seam_finder = new VoronoiSeamFinder();    //逐点法
    //seam_finder = new DpSeamFinder(DpSeamFinder::COLOR);    //动态规范法
    //seam_finder = new DpSeamFinder(DpSeamFinder::COLOR_GRAD);
    //图割法
    //seam_finder = new GraphCutSeamFinder(GraphCutSeamFinder::COST_COLOR);
    seam_finder = new GraphCutSeamFinder(GraphCutSeamFinder::COST_COLOR_GRAD);


    vector <UMat> images_warped_f(2);
    for (int i = 0; i < 2; ++i)    //图像数据类型转换
        images_warped[i].convertTo(images_warped_f[i], CV_32F);
    //得到接缝线的掩码图像masks_warped
    seam_finder->find(images_warped_f, corners, masks_warped);


    // 类型转换
    vector <Mat> images_warped_new(2);
    images_warped[0].copyTo(images_warped_new[0]);
    images_warped[1].copyTo(images_warped_new[1]);

    vector <Mat> mask_warped_new(2);
    masks_warped[0].copyTo(mask_warped_new[0]);
    masks_warped[1].copyTo(mask_warped_new[1]);


    //通过canny边缘检测，得到掩码边界，其中有一条边界就是接缝线
    for (int k = 0; k < 2; k++)
    {
        namedWindow("mask", WINDOW_NORMAL);
        imshow("mask", mask_warped_new[k]);
        waitKey(0);
        Canny(mask_warped_new[k], mask_warped_new[k], 3, 9, 3);
    }



    //为了使接缝线看得更清楚，这里使用了膨胀运算来加粗边界线
    vector <Mat> dilate_img(2);
    Mat element = getStructuringElement(MORPH_RECT, Size(10, 10));    //定义结构元素

    for (int k = 0; k < 2; k++)    //遍历两幅图像
    {
        dilate(mask_warped_new[k], dilate_img[k], element);    //膨胀运算
        //在映射变换图上画出接缝线，在这里只是为了呈现出的一种效果，所以并没有区分接缝线和其他掩码边界
        for (int y = 0; y < images_warped[k].rows; y++) {
            for (int x = 0; x < images_warped[k].cols; x++) {
                if (dilate_img[k].at<uchar>(y, x) == 255 && mask_origin[k].at<uchar>(y, x)!=0)    //掩码边界
                {
                    images_warped_new[k].at<Vec3b>(y, x)[0] = 255;
                    images_warped_new[k].at<Vec3b>(y, x)[1] = 100;
                    images_warped_new[k].at<Vec3b>(y, x)[2] = 200;
                }
            }
        }
    }

    for(int k =0;k<2;k++)
    {
        for(int i=0;i<sizes[k].height;i++)
        {
            for(int j=0;j<sizes[k].width;j++)
            {
                if(mask_origin[k].at<uchar>(i, j)==0)    //掩码
                {
                    images_warped_new[k].at<Vec3b>(i, j)[0]=0;
                    images_warped_new[k].at<Vec3b>(i, j)[1]=0;
                    images_warped_new[k].at<Vec3b>(i, j)[2]=0;
                }
            }
        }
    }


    imwrite("seam1.jpg", images_warped_new[0]);    //存储图像
    imwrite("seam2.jpg", images_warped_new[1]);

    return 0;

}