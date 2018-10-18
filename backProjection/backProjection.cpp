#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace std;
using namespace cv;
void Hist_and_Backproj(int, void*);

Mat hue;
// 多个灰度级别
int bins = 25;

int main()
{
    Mat src = imread("2.png");
    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);

    hue.create(hsv.size(), hsv.depth());
    int ch[] = {0, 0};
    mixChannels(&hsv, 1, &hue, 1, ch, 1);

    namedWindow("source image");
    createTrackbar("hue bins", "source image", &bins, 180, Hist_and_Backproj);
    Hist_and_Backproj(0, 0);

    imshow("source image", src);
    waitKey();
    return 0;

}

void Hist_and_Backproj(int, void*)
{
    int histSize = MAX( bins, 2 );
    float hue_range[] = { 0, 180 };
    // calcHist的类型是因为clacHist函数定义决定 类似指针的指针
    const float* ranges = { hue_range };
    Mat hist;
    calcHist( &hue, 1, 0, Mat(), hist, 1, &histSize, &ranges, true, false );
    normalize( hist, hist, 0, 255, NORM_MINMAX, -1, Mat() );

    // 反向投影
    Mat backproj;
    calcBackProject( &hue, 1, 0, hist, backproj, &ranges, 1, true );
    imshow( "BackProj", backproj );

    //绘制直方图 w h分辨代表宽度和长度
    int w = 400, h = 400;
    int bin_w = cvRound( (double) w / histSize );
    Mat histImg = Mat::zeros( h, w, CV_8UC3 );
    for (int i = 0; i < bins; i++)
    {
        // 坐标原点位于左上方
        rectangle( histImg, Point( i*bin_w, h ), Point( (i+1)*bin_w, h - cvRound( hist.at<float>(i)*h/255.0 ) ),
                   Scalar( 0, 0, 255 ), FILLED );
    }
    imshow( "Histogram", histImg );

}
