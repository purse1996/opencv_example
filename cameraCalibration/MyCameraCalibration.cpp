#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include<iostream>
#include<fstream>

using namespace std;
using namespace cv;

int main()
{
    ifstream fin("img.txt");
    ofstream fout("result.txt");

    int img_count = 0;
    Size img_size;//图像尺寸
    Size board_size = Size(4, 6);//行列上的角点数量
    vector<Point2f> per_img_point;//每张图检测的角点数量
    vector<vector<Point2f> > img_point;

    string filename;
 //   int corner_count = -1;//角点数量
    cout<<"寻找角点"<<endl;

    while(getline(fin, filename))
    {
        cout<<filename<<endl;
        img_count++;
        Mat img = cv::imread(filename);
        if(img.empty())
        {
            cout<<"can't read the image"<<endl;
            return -2;
        }
        //获取图像尺寸
        if (img_count==1)
        {
            img_size.width = img.cols;
            img_size.height = img.rows;
        }

        if(findChessboardCorners(img, board_size, per_img_point)==0)
        {
            cout<<"can not find corners"<<endl;
            return -1;
        }
        else
        {
            Mat img_gray;
            cvtColor(img, img_gray, CV_RGB2GRAY);//转换为灰度图来通过亚像素精确化寻找角点坐标
            find4QuadCornerSubpix(img_gray, per_img_point, Size(5, 5));
            img_point.push_back(per_img_point);

            //在图像显示角点位置
            drawChessboardCorners(img_gray, board_size, per_img_point, true);
            imshow("corners", img_gray);
            waitKey(500);
        }

    }

    cout<<"图像数量"<<img_count<<endl;

    int corner_number = img_point.size();//所有的角点数
    int per_corner_number = board_size.width * board_size.height;//每张图的角点数
    //输出所有角点
    for(int i=0; i<corner_number; i++)
    {
        if(i%per_corner_number==0)
        {
            int j=i/per_corner_number;
            cout<<"-->第"<<j+1<<"图片角点"<<"-->"<<endl;
        }
        if(i%3==0)
            cout<<endl;
        else
            cout.width(10);
        cout<<"-->"<<img_point[i][0].x<<"-->"<<img_point[i][0].y;
    }
    cout<<"角点提取完成"<<endl;
    cout<<"角点数量"<<img_point.size()<<endl;



    cout<<"开始标定"<<endl;
    Size square_size = Size(10, 10);//每个棋盘格大小
    vector<vector<Point3f> > object_points; //世界坐标系中的三维坐标

    Mat camereaMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));//摄像机的内参矩阵
    vector<int> point_count; //每幅图中角点数量
    Mat distCoffeffs = Mat(1, 5, CV_32FC1, Scalar::all(0)); //摄像机的畸变系数

    vector<Mat> rotaMatrix; //旋转向量，最后需要转换为旋转矩阵
    vector<Mat> transMatrix; //平移向量


    // 初始化标定板上角点的世界坐标
    for(int num=0; num<img_count; num++)
    {
        vector<Point3f> temp;
        for(int i=0; i<board_size.height; i++)
        {
            for(int j=0; j<board_size.width; j++)
            {
                Point3f realPoints;
                // 假设世界坐标系中z=0????坐标很奇怪
                realPoints.x = i*square_size.width;
                realPoints.y = j*square_size.height;
                realPoints.z = 0;
                temp.push_back(realPoints);
            }
        }
        object_points.push_back(temp);
    }

    //标定图像中角点数量理论为这么多
    for(int i=0; i<img_count; i++)
    {
        point_count.push_back(board_size.width*board_size.height);
    }

    //开始标定
    calibrateCamera(object_points, img_point, img_size, camereaMatrix, distCoffeffs, rotaMatrix, transMatrix, 0);
    cout<<"标定完成"<<endl;
    cout<<"评价标定结果"<<endl;

    double total_err;
    double per_err;
    vector<Point2f> new_point;//重投影之后的角点坐标
    fout<<"每幅图的标定误差 \n";
    cout<<"每幅图的标定误差"<<endl;

    for(int i=0; i<img_count; i++)
    {
        vector<Point3f> temp_points = object_points[i];
        //得到新投影之后点的坐标
        projectPoints(temp_points, rotaMatrix[i], transMatrix[i], camereaMatrix, distCoffeffs, new_point);
        //计算新旧投影点的误差
        vector<Point2f> origin_points = img_point[i];
        Mat new_point_matrix = Mat(1, new_point.size(), CV_32FC2);
        Mat new_origin_matrix = Mat(1, origin_points.size(), CV_32FC2);

        for(int j=0; j<origin_points.size(); j++)
        {
            new_point_matrix.at<Vec2f>(0, j) = Vec2f(new_point[j].x, new_point[j].y);
            new_origin_matrix.at<Vec2f>(0, j) = Vec2f(origin_points[j].x, origin_points[j].y);

            per_err = norm(new_point_matrix, new_origin_matrix, NORM_L2)/origin_points.size();
            total_err += per_err;
        }
        cout<<"第"<<i+1<<"幅图的平均误差"<<per_err<<"像素"<<endl;
        fout<<"第"<<i+1<<"幅图的平均误差"<<per_err<<"像素"<<endl;
    }
    cout<<"总体平均误差为"<<total_err/img_count<<endl;
    fout<<"总体平均误差为"<<total_err/img_count<<endl;
    cout<<"评价完成"<<endl;


    cout<<"保存标定结果"<<endl;
    fout<<"相机内参矩阵"<<endl;
    fout<<camereaMatrix<<endl;
    fout<<"畸变系数"<<endl;
    fout<<distCoffeffs<<endl;
    Mat rota_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0));//旋转矩阵

    //保存旋转矩阵和平移向量
    for(int i=0; i<img_count; i++){
        fout<<"第"<<i+1<<"幅图的旋转向量"<<endl;
        fout<<rotaMatrix[i]<<endl;
        fout<<"第"<<i+1<<"幅图的旋转矩阵"<<endl;
        Rodrigues(rotaMatrix[i], rota_matrix);
        fout<<rota_matrix<<endl;
        fout<<"第"<<i+1<<"幅图的平移向量"<<endl;
        fout<<transMatrix[i]<<endl<<endl;
    }

    cout<<"完成保存"<<endl;
    fout<<endl;

    //图像矫正
    Mat src = imread("1_d.jpg");
    Mat distoration = src.clone();
    undistort(src, distoration, camereaMatrix, distCoffeffs);
    imwrite("undisotation.jpg", distoration);

    return 0;





}
