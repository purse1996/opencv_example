#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/imgcodecs.hpp>

#include<iostream>

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    //命令还不是完全懂
    //./mySobel "lena.jpeg" -k 3 - s 1 -d 0
    cv::CommandLineParser parser(argc, argv,
                                 "{@input | ../lena.jpeg |input image}"
                                 "{ksize k|1|ksize(hit k to increase its value)}"
                                 "{scale s|1|scale(hit s to increase its value)}"
                                 "{delta d|0|delta (hit 'D' to increase its value)}"
                                 "{help  h|false|show help message}");

    //打印输出文档
    parser.printMessage();
    Mat img, src, src_gray;
    Mat grad;
    const string window_name = "sobel detection";
    int ksize = parser.get<int>("ksize");
    int scale = parser.get<int>("scale");
    int delta = parser.get<int>("delta");
    int ddepth = CV_16S;
    String name = parser.get<String>("@input");
    img = imread(name, IMREAD_COLOR);
    if(img.empty())
    {
        cout<<"can't read the image"<<endl;
        return -1;
    }
    for(;;)
    {
        GaussianBlur(img, src, Size(3, 3), 0, 0, BORDER_DEFAULT);
        cvtColor(src, src_gray, COLOR_BGR2GRAY);

        Mat grad_x, grad_y;
        Mat abs_grad_x, abs_grad_y;
        // ddpeth设为CV_16S 防止溢出， x_order=1, y_orderj=0 计算x方向导数 , scale是否考虑图像尺度变化
         //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
         //在采用kernel_size为3时候，使用Scharr可以更准确计算数值导数
        Sobel(src_gray, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
        Sobel(src_gray, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);


        convertScaleAbs(grad_x, abs_grad_x);
        convertScaleAbs(grad_y, abs_grad_y);
        addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
        imshow(window_name, grad);
        char key = (char)waitKey(0);
        if(key == 27)
        {
        return 0;
        }
        if (key == 'k' || key == 'K')
        {
        ksize = ksize < 30 ? ksize+2 : -1;
        }
        if (key == 's' || key == 'S')
        {
        scale++;
        }
        if (key == 'd' || key == 'D')
        {
        delta++;
        }
        if (key == 'r' || key == 'R')
        {
        scale =  1;
        ksize = -1;
        delta =  0;
        }
    }
    return 0;

}
