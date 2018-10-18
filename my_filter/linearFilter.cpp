#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>

#include<iostream>
using namespace std;
using namespace cv;
int main(int argc, char *argv[])
{
    Mat img = imread(argv[1]);
    Mat out, border1, border2;
    int kernel_size = 3;
    //CV_32F 32位浮点数 值域在0到1范围内
    Mat kernel = Mat::ones(kernel_size, kernel_size, CV_32F)/float(kernel_size*kernel_size);

    //anchor = Point(-1, -1);

     //filter2D(src, dst, ddepth , kernel, anchor, delta,
                // BORDER_DEFAULT );
    filter2D(img, out, -1, kernel, Point(-1, -1), 0, BORDER_DEFAULT);
    int top = (int) (0.05*img.rows);
    int bottom = top;
    int left = (int) (0.05*img.cols);
    int right = left;
    copyMakeBorder(img, border1, top, bottom, left, right, BORDER_CONSTANT, Scalar::all(0));
    copyMakeBorder(img, border1, top, bottom, left, right, BORDER_REPLICATE, Scalar::all(0));
    //namedWindow("filter");
    //namedWindow("border1");
    //namedWindow("border2");
    imshow("filter", out);

    char c=(char)(waitKey(500));
    if (c=='o')
        imshow("filter", out);
    //waitKey(0);
    else if(c=='c')
        imshow("border1", border1);
    //waitKey(0);
    else if(c=='r')
        imshow("border2", border2);
    return 0;

}
