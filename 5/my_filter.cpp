#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>

using namespace std;
using namespace cv;

static void help(char* progName)
{
    cout << endl
        <<  "This program shows how to filter images with mask: the write it yourself and the"
        << "filter2d way. " << endl
        <<  "Usage:"                                                                        << endl
        << progName << " [image_path] [G -- grayscale] "        << endl << endl;
}

void my_filter(Mat img, Mat& result);
int main(int argc, char *argv[])
{
    help(argv[0]);
    Mat img, result_img, result_img2;
    if(argc>=3&&!strcmp("G", argv[2]))
    {
        img = imread(argv[1], IMREAD_GRAYSCALE);
    }
    else{
        img = imread(argv[1], IMREAD_COLOR);
    }
    if(img.empty())
    {
        cout<<"can't read the image"<<endl;
        return -1;
    }
    my_filter(img, result_img);
    namedWindow("Input", WINDOW_AUTOSIZE);
    namedWindow("Output", WINDOW_AUTOSIZE);
    imshow("Input", img );
    waitKey(0);
    imshow("Output", result_img);
    waitKey(0);

    //使用opencv2中自带的filter函数

    Mat kernel = (Mat_<char>(3, 3) << 0, -1, 0,
                                -1, 5, -1,
                                0, -1, 0);
    //img.depth代表图像每个像素使用的位数
    filter2D(img,result_img2, img.depth(), kernel);

    return 0;
}

void my_filter(Mat img, Mat& result)
{
    //检验图像是否是uchar类型的
    CV_Assert(img.depth() == CV_8U);
    int channels = img.channels();
    result.create(img.size(), img.type());
    result.setTo(Scalar(0));
    for(int i=1; i<img.rows - 1; i++)
    {
        uchar* previous = img.ptr<uchar>(i-1);
        uchar* current = img.ptr<uchar>(i);
        uchar* next = img.ptr<uchar>(i+1);

        uchar* out = result.ptr<uchar>(i);

        for(int j=channels; j<channels*(img.cols-1); j++)
        {
            //saturate_cast防止灰度值溢出
            //先取出*out的值然后讲out代表地址+1
            *out++ = saturate_cast<uchar>(5*current[j] - current[j-channels]
                                          - current[j+channels] - previous[j]
                                          - next[j]);
        }
    }
    //result.row(0).setTo(Scalar(0));
    //result.row(result.rows-1).setTo(Scalar(0));
    //result.col(0).setTo(Scalar(0));
    //result.col(Result.cols-1).setTo(Scalar(0));

}
