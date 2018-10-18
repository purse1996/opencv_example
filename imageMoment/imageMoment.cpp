#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>

#include<iomanip>
#include<iostream>

using namespace std;
using namespace cv;
Mat img_gray;
int thresh = 100;
const int max_thresh = 255;
RNG rng(12345);

void thresh_callback(int, void*);
int main()
{
    Mat img = imread("lena.jpeg");
    if(img.empty())
    {
        cout<<"can't read the image"<<endl;
        return -1;
    }
    cvtColor(img, img_gray, COLOR_BGR2GRAY);
    blur(img_gray, img_gray, Size(3, 3));
    imshow("source", img);
    createTrackbar("canny thresh", "source", &thresh, max_thresh, thresh_callback);
    thresh_callback(0, 0);
    waitKey(0);
    return 0;

}

void thresh_callback(int, void*)
{
    Mat edge_img;
    Canny(img_gray, edge_img, thresh, 2*thresh, 3);

    //提取轮廓
    vector<vector<Point> > contours;
    findContours(edge_img, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);

    //存储图像的3阶特征钜
    vector<Moments> mu(contours.size());
    for(size_t i=0; i<contours.size(); i++)
    {
        mu[i] = moments(contours[i]);
    }

    //计算质心的位置 (m10/m00, m01/m00)
    vector<Point2f> mc(contours.size());
    for(size_t i=0; i<contours.size() ;i++)
    {
        mc[i] = Point2f(static_cast<float>(mu[i].m10/(mu[i].m00+1e-5)),
                            static_cast<float>(mu[i].m01/(mu[i].m00+1e-5)));
        cout<<"mc["<<i<<"]"<<mc[i]<<endl;
    }
        Mat drawing = Mat::zeros( edge_img.size(), CV_8UC3 );


    for( size_t i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        drawContours( drawing, contours, (int)i, color, 2 );
        circle( drawing, mc[i], 4, color, -1 );
    }

    imshow( "Contours", drawing );
    cout << "\t Info: Area and Contour Length \n";

    // 对于二值化得图像 m00即为轮廓的面积
    for( size_t i = 0; i < contours.size(); i++ )
    {
        cout << " * Contour[" << i << "] - Area (M_00) = " << std::fixed << std::setprecision(2) << mu[i].m00
             << " - Area OpenCV: " << contourArea(contours[i]) << " - Length: " << arcLength( contours[i], true ) << endl;
    }


}
