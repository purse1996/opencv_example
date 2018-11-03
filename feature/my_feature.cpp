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

#include<iomanip>
#include<iostream>
#include<string>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::detail;

int main(int argc, char *argv[])
{
    if(argc!=2)
    {
        cout<<"not enough parameters"<<endl;
        return -1;
    }
    Mat img = imread(argv[1]);
    //单纯的特征点检测也可以使用xfeature2d库中，这里使用的是detail，
    //但是detail只提供了ORB和SURF 这里使用xfeature来提取sift特征
    vector<KeyPoint> sift_feature;
    vector<ImageFeatures> features(2);
    vector<Mat> img_feature(3);
    Ptr<Feature2D> sift = SIFT::create();
    Ptr<FeaturesFinder> finder[2];
    finder[0] = makePtr<OrbFeaturesFinder>();
    finder[1] = makePtr<SurfFeaturesFinder>();

    Mat descriptors;

    double t1 = (double)getTickCount();
    (*(finder[0]))(img, features[0]);
    t1 = ((double)getTickCount()-t1)*1000/getTickFrequency();
    double t2 = (double)getTickCount();
    (*(finder[1]))(img, features[1]);
    t2 = ((double)getTickCount()-t2)*1000/getTickFrequency();
    double t3 = (double)getTickCount();
    sift->detect(img, sift_feature);
    sift->compute(img, sift_feature, descriptors);
    t3 = ((double)getTickCount()-t3)*1000/getTickFrequency();
    drawKeypoints(img, features[0].keypoints, img_feature[0], Scalar::all(-1));
    drawKeypoints(img, features[1].keypoints, img_feature[1], Scalar::all(-1));
    drawKeypoints(img, sift_feature, img_feature[2], Scalar::all(-1));

    //将特征点合并在一幅图上来观察
    Mat img_feature_all;
    img_feature_all.create(Size(img_feature[0].cols + img_feature[1].cols + img_feature[2].cols, max(img_feature[0].rows, img_feature[1].rows)), CV_8UC3);
    Mat imgROI = img_feature_all(Rect(0,0, img_feature[0].cols, img_feature[0].rows));
    resize(img_feature[0], imgROI, Size(img_feature[0].cols, img_feature[0].rows));
    imgROI = img_feature_all(Rect(img_feature[0].cols,0,  img_feature[1].cols, img_feature[1].rows));
    resize(img_feature[1], imgROI, Size(img_feature[1].cols, img_feature[1].rows));
    imgROI = img_feature_all(Rect(img_feature[0].cols+img_feature[1].cols ,0,  img_feature[2].cols, img_feature[2].rows));
    resize(img_feature[2], imgROI, Size(img_feature[2].cols, img_feature[2].rows));


    cout<<left<<setw(8)<<"name"<<left<<setw(8)<<"time/ms"<<left<<setw(8)<<"number"<<endl;
    cout<<left<<setw(8)<<"ORB"<<left<<setw(8)<<t1<<left<<setw(8)<<features[0].keypoints.size()<<endl;
    cout<<left<<setw(8)<<"SURF"<<left<<setw(8)<<t2<<left<<setw(8)<<features[1].keypoints.size()<<endl;
    cout<<left<<setw(8)<<"SIFT"<<left<<setw(8)<<t3<<left<<setw(8)<<sift_feature.size()<<endl;
    namedWindow("features, the left is ORB, the middle Surf, the right is SIFT", WINDOW_NORMAL);
    imshow("features, the left is ORB, the middle Surf, the right is SIFT", img_feature_all);
    waitKey(0);

    return 0;
}
