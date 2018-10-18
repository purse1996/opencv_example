#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

#define w 500
//定义图片大小

using namespace std;
using namespace cv;
void MyLine( Mat img, Point pstart, Point pend);
void MyEllipse( Mat img, double angle );
void MyFilledCircle( Mat img, Point center );
int main()
{
    Mat img1 = Mat::zeros(w, w, CV_8UC3);
    MyLine(img1, Point( 0, 15*w/16 ), Point( w, 15*w/16 ) );
    MyEllipse(img1, 90 );
    MyEllipse(img1, 0 );
    MyFilledCircle(img1, Point( w/2, w/2) );
    imshow("pic1", img1);
    waitKey(0);
    return 0;
}
void MyLine( Mat img, Point pstart, Point pend)
{
  int thickness = 2;
  int lineType = LINE_8;
  //Scalar代表图片三通道的颜色
  line( img,
    pstart,
    pend,
    Scalar( 0, 255, 0 ),
    thickness,
    lineType );
}

void MyEllipse( Mat img, double angle )
{
  int thickness = 2;
  int lineType = 8;
  //angle初始旋转角度，w/2，w/2椭圆中心位置,Size代表椭圆大小
  //0, 360代表全部椭圆

  ellipse( img,
       Point( w/2, w/2 ),
       Size( w/4, w/16 ),
       angle,
       0,
       360,
       Scalar( 255, 0, 0 ),
       thickness,
       lineType );
}


void MyFilledCircle( Mat img, Point center )
{
//center圆心 w/32半径，FILLED实心圆
  circle( img,
      center,
      w/32,
      Scalar( 0, 0, 255 ),
      FILLED,
      LINE_8 );
}

