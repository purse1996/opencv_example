#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/imgproc.hpp>

#include<iostream>

using namespace std;
using namespace cv;

void fourierTransform(Mat img, Mat& result);
int main(int argc, char *argv[])
{
    //注意这里也要以灰度图的形式读取下来！！！
    Mat img = imread(argv[1], IMREAD_GRAYSCALE);
    Mat result;
    if(img.empty())
    {
        cout<<"can't open the image"<<endl;
        return -1;
    }
    fourierTransform(img, result);
    imshow("input image", img);
    imshow("spectrum magnitude", result);
    waitKey(0);
    return 0;
}
void fourierTransform(Mat img, Mat& result)
{
//DFT最优的图像大小是 2 3 5的倍数，因而首先扩展图像的边缘
    int m = getOptimalDFTSize(img.rows);
    int n = getOptimalDFTSize(img.cols);
    Mat padded;
    copyMakeBorder(img, padded, 0, m-img.rows, 0, n-img.cols, BORDER_CONSTANT, Scalar::all(0) );
    //频域中变化更加精确，因而采用32位浮点数来存储数值; CV32F代表32位浮点数
    //傅里叶变化包含幅度谱和相位谱
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    //将planes数组合并成一个二通道的图像,分辨代表实数和虚数部分
    merge(planes, 2, complexI);
    //DFT变换
    dft(complexI, complexI);
    //计算幅度谱
    split(complexI, planes);
    magnitude(planes[0], planes[1], result);
    //由于傅里叶变换后值太大，所以一般采用对数形式表示
    result = result + Scalar::all(1);
    log(result, result);

    //由于一开始进行了扩充，因而在这要进行裁剪
    //-2是1111 1110 因而&-2相当于取最接近该数的偶数
    //Rect(int _x,int _y,int _width,int _height);
    result = result(Rect(0,0, result.cols&-2, result.rows&-2));
    int cx = result.cols/2;
    int cy = result.rows/2;
    //将低频成分移动至图像中间
    Mat q0(result, Rect(0, 0, cx, cy));
    Mat q1(result, Rect(cx, 0, cx, cy));
    Mat q2(result, Rect(0, cy, cx, cy));
    Mat q3(result, Rect(cx, cy, cx, cy));

    //变换左上角和右下角象限
    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    //变换右上角和左下角象限
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    //此时幅度谱的范围仍然超出了0到1 因而要进行归一化
    normalize(result, result, 0, 1, CV_MINMAX);

}

