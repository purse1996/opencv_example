#include<opencv2/core.hpp>//opencv中基础模块
#include<opencv2/highgui.hpp>//opencv I/O
#include<opencv2/imgproc.hpp> //gaussian blur
#include<opencv2/video.hpp>

#include<iostream>
#include<string>
#include<iomanip> // 控制输出精度
#include<sstream>// string to number 转换

using namespace std;
using namespace cv;

double getPSNR(const Mat& I1, const Mat& I2);//计算PSNR值
Scalar getSSIM( const Mat& i1, const Mat& i2);// 计算SSIM值

int main(int agrc, char **argv)
{
    const string sourceVideo = argv[1], testVideo = argv[2];
    int frameNumber = -1; // 视频帧数

    //声明一个视频流的对象
    VideoCapture source(sourceVideo);
    VideoCapture test;
    test.open(testVideo);// 这两种表达方式含义一致
    if(!source.isOpened())
    {
        cout<<"can't open source video"<<endl;
        return -1;
    }
    if(!test.isOpened())
    {
        cout<<"can't open test video"<<endl;
        return -1;
    }
    // 获取视频属性
    // Size为opencv默认数据类型， 构造函数为Size(width, height);
    Size ssource = Size((int) source.get(CAP_PROP_FRAME_WIDTH),
                       (int) source.get(CAP_PROP_FRAME_HEIGHT));
    Size stest   = Size((int) test.get(CAP_PROP_FRAME_WIDTH),
                       (int) test.get(CAP_PROP_FRAME_HEIGHT));

    if(ssource!=stest)
    {
        cout<<"source video and test video have different size"<<endl;
        return -1;
    }

    const char* WIN_SOURCE  = "source video";
    const char* WIN_TEST = "test video";

    namedWindow(WIN_SOURCE, WINDOW_AUTOSIZE);
    namedWindow(WIN_TEST, WINDOW_AUTOSIZE);
    moveWindow(WIN_SOURCE, 500, 0);
    moveWindow(WIN_TEST, 600+ssource.width, 0);
    cout<<"width="<<ssource.width<<"Height="<<ssource.height<<"number="<<source.get(CAP_PROP_FRAME_COUNT);

    Mat sourceImage, testImage;
    double psnrV;
    Scalar ssimV;//存储三通道的ssim
    for(;;)
    {
        // 从视频流中获取每一帧图像
        source >> sourceImage;
        test   >> testImage;

        if(sourceImage.empty()||testImage.empty())
        {
            cout<<"over"<<endl;
            break;
        }

        frameNumber++;
        cout<<"frame="<<frameNumber+1<<endl;
        psnrV = getPSNR(sourceImage, sourceImage);
        cout<<setiosflags(ios::fixed)<<setprecision(3)<<psnrV<<endl;

        ssimV = getSSIM(sourceImage, sourceImage);
        cout<<"SSIM"
            <<"R"<<setiosflags(ios::fixed)<<setprecision(3)<<ssimV[2]*100<<"%"
            <<"G"<<setiosflags(ios::fixed)<<setprecision(3)<<ssimV[1]*100<<"%"
            <<"B"<<setiosflags(ios::fixed)<<setprecision(3)<<ssimV[0]*100<<"%"<<endl;
        imshow(WIN_SOURCE, sourceImage);
        imshow(WIN_TEST, testImage);

        char c=(char)waitKey(500);
        if(c==27)
        {
            break;
        }

    }
    return 0;
}

// 对照着具体公式即可很容易理解该部分代码
double getPSNR(const Mat& I1, const Mat& I2)
{
    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2
    Scalar s = sum(s1);        // sum elements per channel
    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels
    if( sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double mse  = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}
Scalar getSSIM( const Mat& i1, const Mat& i2)
{
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d = CV_32F;
    Mat I1, I2;
    i1.convertTo(I1, d);            // cannot calculate on one byte large values
    i2.convertTo(I2, d);
    Mat I2_2   = I2.mul(I2);        // I2^2
    Mat I1_2   = I1.mul(I1);        // I1^2
    Mat I1_I2  = I1.mul(I2);        // I1 * I2
    /*************************** END INITS **********************************/
    Mat mu1, mu2;                   // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);
    Mat mu1_2   =   mu1.mul(mu1);
    Mat mu2_2   =   mu2.mul(mu2);
    Mat mu1_mu2 =   mu1.mul(mu2);
    Mat sigma1_2, sigma2_2, sigma12;
    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;
    Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);                 // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);                 // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    Mat ssim_map;
    divide(t3, t1, ssim_map);        // ssim_map =  t3./t1;
    Scalar mssim = mean(ssim_map);   // mssim = average of ssim map
    return mssim;
}
