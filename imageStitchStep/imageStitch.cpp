//
// Created by byiwind on 19-2-21.
//

#include <fstream>
#include <string>
#include<iostream>
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

//void findMaxSpanningTree(int num_images, const std::vector<MatchesInfo> &pairwise_matches,
//                             Graph &span_tree, std::vector<int> &centers);

int main(int argc, char** argv)
{
    vector<Mat> imgs;    //输入9幅图像
    Mat img;
    // 第一组测试数据
//    img = imread("8mm_16mm/1.jpg");
//    imgs.push_back(img);
//    img = imread("8mm_16mm/2.jpg");
//    imgs.push_back(img);
//    img = imread("8mm_16mm/3.jpg");
//    imgs.push_back(img);
//
//    img = imread("8mm_16mm/4.jpg");
//    imgs.push_back(img);
    //   imgs.push_back(img);
//    // 第二组测试数据
//    img = imread("4.2mm/0.jpg");
//    imgs.push_back(img);
//    img = imread("4.2mm/1.jpg");
//    imgs.push_back(img);
//    img = imread("4.2mm/2.jpg");
//    imgs.push_back(img);
//    img = imread("4.2mm/3.jpg");
//    imgs.push_back(img);
//    img = imread("4.2mm/4.jpg");
//    imgs.push_back(img);
//    img = imread("4.2mm/5.jpg");
//    imgs.push_back(img);
//    img = imread("4.2mm/6.jpg");
//    imgs.push_back(img);
//    img = imread("4.2mm/7.jpg");
//    imgs.push_back(img);
    // 第三组测试数据
//    img = imread("5mm/1.jpg");
//    imgs.push_back(img);
//    img = imread("5mm/2.jpg");
//    imgs.push_back(img);
//    img = imread("5mm/3.jpg");
//    imgs.push_back(img);
//    img = imread("5mm/4.jpg");
//    imgs.push_back(img);
//    img = imread("5mm/5.jpg");
//    imgs.push_back(img);
//    img = imread("5mm/6.jpg");
//    imgs.push_back(img);
//    img = imread("5mm/7.jpg");
//    imgs.push_back(img);
//    img = imread("5mm/8.jpg");
//    imgs.push_back(img);


    // 第四组测试数据
//    img = imread("test01/1.jpg");
//    imgs.push_back(img);
//    img = imread("test01/2.png");
//    imgs.push_back(img);



//    img = imread("15.jpg");
//    imgs.push_back(img);
//    img = imread("16.jpg");
//    imgs.push_back(img);
//    img = imread("17.jpg");
//    imgs.push_back(img);
//    img = imread("18.jpg");
//    imgs.push_back(img);
    //img = imread("9.jpg");
    //imgs.push_back(img);



    img = imread("1_new.jpg");
    imgs.push_back(img);
    img = imread("2_new.jpg");
    imgs.push_back(img);


    int num_images = imgs.size();    //图像数量
    cout<<"图像数量为"<<num_images<<endl;
    cout<<"图像读取完毕"<<endl;
    Ptr<FeaturesFinder> finder;    //定义特征寻找器
    finder = new SurfFeaturesFinder();    //应用SURF方法寻找特征
    //finder = new OrbFeaturesFinder();    //应用ORB方法寻找特征
    vector<ImageFeatures> features(num_images);    //表示图像特征
    for (int i =0 ;i<num_images;i++)
        (*finder)(imgs[i], features[i]);    //特征检测
    cout<<"特征提取完毕"<<endl;
    vector<MatchesInfo> pairwise_matches;    //表示特征匹配信息变量
    BestOf2NearestMatcher matcher(false, 0.3f, 6, 6);    //定义特征匹配器，2NN方法
    matcher(features, pairwise_matches);    //进行特征匹配
    /*打印图像之间的匹配关系匹配*/

    for(size_t i=0; i<num_images; i++)
        for(size_t j=0; j<num_images; j++)
        {
            if(pairwise_matches[i*num_images+j].H.empty())
                continue;
            cout<<"第"<<i<<"匹配"<<j<<"幅图片置信度为"
                <<pairwise_matches[i*num_images+j].confidence<<endl;
        }


    /*根据阈值来判断图片之间的匹配关系*/
    // vector<Mat> img_subset;
    // vector<Mat>
//    float conf_thresh = 0;
//    vector<Mat> imgs_subset;
//    vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_threshold);
//    for(size_t i=0; i<indices.size(); i++)
//    {
//        imgs_subset.push_back(imgs[i]);
//    }
//    if(imgs_subset.size()<2)
//    {
//        cout<<"该全景图需要更多图片，"<<endl;
//    }


    /*判断图像之间的相互关系，并寻找基准图像*/
    //const int node_number = static_cast<int>(features.size());
//    int node_number = static_cast<int>(features.size());
//    cout<<"树节点数量"<<node_number<<endl;
//    Graph span_tree;
//    std::vector<int> span_tree_centers;
//    findMaxSpanningTree(node_number, pairwise_matches, span_tree, span_tree_centers);
//    for(size_t i=0; i<span_tree_centers.size(); i++)
//        cout<<"核心节点包括"<<span_tree_centers[i]<<endl;

    cout<<"特征匹配完毕"<<endl;
    HomographyBasedEstimator estimator;    //定义参数评估器
    vector<CameraParams> cameras;    //表示相机参数，内参加外参
    estimator(features, pairwise_matches, cameras);    //进行相机参数评估

    for (size_t i = 0; i < cameras.size(); ++i)    //转换相机旋转参数的数据类型
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
    }
    cout<<"相机参数预测完毕"<<endl;


    for (size_t i = 0; i < cameras.size(); ++i)
    {
        cout<<"第"<<i<<"焦距为"<<cameras[i].focal<<endl;
    }


    // 在一部可以计算重映射误差，想办法让他可以输出出来
    Ptr<detail::BundleAdjusterBase> adjuster;    //光束平差法，精确相机参数
    //adjuster->setRefinementMask();
    adjuster = new detail::BundleAdjusterReproj();    //重映射误差方法
    //adjuster = new detail::BundleAdjusterRay();    //射线发散误差方法

    adjuster->setConfThresh(0.6f);    //设置匹配置信度，该值设为1
    (*adjuster)(features, pairwise_matches, cameras);    //精确评估相机参数


    /*查看进行光束平差法之后的树节点数量和核心节点位置*/
//    const int node_number = static_cast<int>(features.size());
//    cout<<"树节点数量"<<node_number<<endl;
//    //Graph span_tree;
//    //std::vector<int> span_tree_centers;
//    findMaxSpanningTree(node_number, pairwise_matches, span_tree, span_tree_centers);
//    for(size_t i=0; i<span_tree_centers.size(); i++)
//        cout<<"核心节点包括"<<span_tree_centers[i]<<endl;


    vector<Mat> rmats;
    for (size_t i = 0; i < cameras.size(); ++i)    //复制相机的旋转参数
        rmats.push_back(cameras[i].R.clone());
    waveCorrect(rmats, WAVE_CORRECT_HORIZ);    //进行波形校正
    for (size_t i = 0; i < cameras.size(); ++i)    //相机参数赋值
        cameras[i].R = rmats[i];
    rmats.clear();    //清变量

    cout<<"利用光束平差法进行相机矩阵更新"<<endl;

    vector<Point> corners(num_images);    //表示映射变换后图像的左上角坐标
    vector<UMat> masks_warped(num_images);    //表示映射变换后的图像掩码
    vector<UMat> images_warped(num_images);    //表示映射变换后的图像
    vector<Size> sizes(num_images);    //表示映射变换后的图像尺寸
    vector<UMat> masks(num_images);    //表示源图的掩码

    for (int i = 0; i < num_images; ++i)    //初始化源图的掩码
    {
        masks[i].create(imgs[i].size(), CV_8U);    //定义尺寸大小
        masks[i].setTo(Scalar::all(255));    //全部赋值为255，表示源图的所有区域都使用
    }

    Ptr<WarperCreator> warper_creator;    //定义图像映射变换创造器
    warper_creator = new cv::SphericalWarper();
    //warper_creator = makePtr<cv::PlaneWarper>();     //平面投影
    //warper_creator = new cv::CylindricalWarper();    //柱面投影
    //warper_creator = new cv::SphericalWarper();    //球面投影
    //warper_creator = new cv::FisheyeWarper();    //鱼眼投影
    //warper_creator = new cv::StereographicWarper();    //立方体投影

    //定义图像映射变换器，设置映射的尺度为相机的焦距，所有相机的焦距都相同
    vector<double> focals;

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        cout<<"第"<<i<<"焦距为"<<cameras[i].focal<<endl;
        focals.push_back(cameras[i].focal);
    }
    sort(focals.begin(), focals.end());
    float warped_image_scale;
    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

    cout<<"最终选择的图像的焦距为"<<warped_image_scale<<endl;
    Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale));

    // 分开进行投影试试:
//    Mat_<float> temp;
//    cameras[0].K().convertTo(temp, CV_32F);
//    corners[0] = warper->warp(imgs[0], temp, cameras[0].R, INTER_LINEAR, BORDER_REFLECT, images_warped[0]);
//    sizes[0] = images_warped[0].size();
//    cout<<"width:  "<<sizes[0].width<<"height:  "<<sizes[0].height<<endl;
//    warper->warp(masks[0], temp, cameras[0].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[0]);



    // ostringstream stream;
    //Mat temp;
    for (int i = 0; i < num_images; ++i)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);    //转换相机内参数的数据类型
        //对当前图像镜像投影变换，得到变换后的图像以及该图像的左上角坐标
        corners[i] = warper->warp(imgs[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();    //得到尺寸

        cout<<"width:    "<<sizes[i].width<<"height:   "<<sizes[i].height<<endl;
        //得到变换后的图像掩码
        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);

//      stream<<i;
//      images_warped[i].copyTo(temp);// 将Umat矩阵转换为Mat,Mat转换为UMat也可以使用该方法
//      imwrite(stream.str()+"bundle.jpg", temp);
//      namedWindow("display image", WINDOW_NORMAL);
//      imshow("display image", images_warped[i]);
//      waitKey(0);
    }


    imgs.clear();    //清变量
    masks.clear();
    cout<<"图像映射完毕"<<endl;
    //创建曝光补偿器，应用增益补偿方法
    Ptr<ExposureCompensator> compensator =
            ExposureCompensator::createDefault(ExposureCompensator::GAIN);
    compensator->feed(corners, images_warped, masks_warped);    //得到曝光补偿器
    for(int i=0;i<num_images;++i)    //应用曝光补偿器，对图像进行曝光补偿
    {
        compensator->apply(i, corners[i], images_warped[i], masks_warped[i]);
    }
    cout<<"图像曝光完毕"<<endl;
    //在后面，我们还需要用到映射变换图的掩码masks_warped，因此这里为该变量添加一个副本masks_seam
    vector<UMat> masks_seam(num_images);
    for(int i = 0; i<num_images;i++)
        masks_warped[i].copyTo(masks_seam[i]);

    Ptr<SeamFinder> seam_finder;    //定义接缝线寻找器
    //seam_finder = new NoSeamFinder();    //无需寻找接缝线
    //seam_finder = new VoronoiSeamFinder();    //逐点法
    //seam_finder = new DpSeamFinder(DpSeamFinder::COLOR);    //动态规范法
    //seam_finder = new DpSeamFinder(DpSeamFinder::COLOR_GRAD);
    //图割法
    //seam_finder = new GraphCutSeamFinder(GraphCutSeamFinder::COST_COLOR);
    seam_finder = new GraphCutSeamFinder(GraphCutSeamFinder::COST_COLOR_GRAD);

    vector<UMat> images_warped_f(num_images);
    for (int i = 0; i < num_images; ++i)    //图像数据类型转换
        images_warped[i].convertTo(images_warped_f[i], CV_32F);

    images_warped.clear();    //清内存
    cout<<"拼缝优化完毕"<<endl;
    //得到接缝线的掩码图像masks_seam
    seam_finder->find(images_warped_f, corners, masks_seam);

    vector<Mat> images_warped_s(num_images);
    Ptr<Blender> blender;    //定义图像融合器

    blender = Blender::createDefault(Blender::NO, false);    //简单融合方法
    //羽化融合方法
//    blender = Blender::createDefault(Blender::FEATHER, false);
//    FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
//    fb->setSharpness(0.005);    //设置羽化锐度

//    blender = Blender::createDefault(Blender::MULTI_BAND, false);    //多频段融合
//    MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
//    mb->setNumBands(8);   //设置频段数，即金字塔层数

    blender->prepare(corners, sizes);    //生成全景图像区域
    cout<<"生成全景图像区域"<<endl;
    //在融合的时候，最重要的是在接缝线两侧进行处理，而上一步在寻找接缝线后得到的掩码的边界就是接缝线处，因此我们还需要在接缝线两侧开辟一块区域用于融合处理，这一处理过程对羽化方法尤为关键
    //应用膨胀算法缩小掩码面积
    vector<Mat> dilate_img(num_images);
    vector<Mat> masks_seam_new(num_images);
    Mat tem;
    Mat element = getStructuringElement(MORPH_RECT, Size(20, 20));    //定义结构元素
    for(int k=0;k<num_images;k++)
    {
        images_warped_f[k].convertTo(images_warped_s[k], CV_16S);    //改变数据类型
        dilate(masks_seam[k], masks_seam_new[k], element);    //膨胀运算
        //映射变换图的掩码和膨胀后的掩码相“与”，从而使扩展的区域仅仅限于接缝线两侧，其他边界处不受影响
        //resize(dilated_mask, tem, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);
        masks_warped[k].copyTo(tem);
        masks_seam_new[k] = masks_seam_new[k] & tem;
        blender->feed(images_warped_s[k], masks_seam_new[k], corners[k]);    //初始化数据
        cout<<"处理完成"<<k<<"图片"<<endl;
    }

    masks_seam.clear();    //清内存

    images_warped_s.clear();

    masks_warped.clear();

    images_warped_f.clear();


    Mat result, result_mask;
    //完成融合操作，得到全景图像result和它的掩码result_mask

    blender->blend(result, result_mask);

    imwrite("result.jpg", result);    //存储全景图像

    return 0;
}