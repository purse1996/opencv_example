#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<iostream>

using namespace std;
using namespace cv;

int ero_elem = 0;
int ero_size = 0;
int dila_elem = 0;
int dila_size = 0;
int const max_elem = 2;
int const max_size = 21;
Mat img, ero_img, dila_img;
void Erosion(int , void*);
void Dilation(int, void*);
//函数原型是由createTrackbar决定的

int main(int argc, char **argv)
{
    img = imread(argv[1]);
    namedWindow("Erosion Demo", WINDOW_AUTOSIZE);
    namedWindow("Dilation Demo", WINDOW_AUTOSIZE);
    moveWindow("Dilation Demo", img.cols, 0);
    //createTrackbar参数说明
    //1名称 2显示的窗口名称 3滑块移动确定整数值 4滑块能移动的最大值
    //5回调函数，每次移动滑块重新调用 要求函数原型为void xxx(int , void*);
    createTrackbar("Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Erosion Demo",
                   &ero_elem, max_elem, Erosion);//元素的形状
    createTrackbar( "Kernel size:\n 2n +1", "Erosion Demo",
                    &ero_size, max_size,
                    Erosion );//元素的大小
    createTrackbar( "Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Dilation Demo",
          &dila_elem, max_elem,
          Dilation );
    createTrackbar( "Kernel size:\n 2n +1", "Dilation Demo",
          &dila_size, max_size,
          Dilation );
    Erosion( 0, 0 );
    Dilation( 0, 0 );
    waitKey(0);
    return 0;
}

void Erosion( int, void* )
{
  int erosion_type = 0;
  if( ero_elem == 0 ){ erosion_type = MORPH_RECT; }
  else if( ero_elem == 1 ){ erosion_type = MORPH_CROSS; }
  else if( ero_elem == 2) { erosion_type = MORPH_ELLIPSE; }
  // 结构元素Rect-->MORPH_RECT  Cross-->MORPH_CROSS Ellipse-->MORPH_ELLIPSE
  Mat element = getStructuringElement( erosion_type,
                       Size( 2*ero_size + 1, 2*ero_size+1 ),
                       Point(-1, -1 ) );
  erode( img, ero_img, element );
  imshow( "Erosion Demo", ero_img );
}

void Dilation( int, void* )
{
  int dilation_type = 0;
  if( dila_elem == 0 ){ dilation_type = MORPH_RECT; }
  else if( dila_elem == 1 ){ dilation_type = MORPH_CROSS; }
  else if( dila_elem == 2) { dilation_type = MORPH_ELLIPSE; }
  Mat element = getStructuringElement( dilation_type,
                       Size( 2*dila_size + 1, 2*dila_size+1 ),
                       Point( dila_size, dila_size ) );
  dilate( img, dila_img, element );
  imshow( "Dilation Demo", dila_img );
}
