#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>

#include<iostream>
#include<string>
using namespace std;
using namespace cv;
int main()
{
    Mat M(2, 2, CV_8UC3, Scalar(0,0,25));
    //2 2 代表矩阵行列
    //CV_8UC3 CV_[The number of bits per item][Signed or Unsigned][Type Prefix]C[The channel number]
    //Scalar代表以常数形式
    cout<<M<<endl;

    Mat R = Mat(3, 2, CV_8UC3);
    randu(R, Scalar::all(0), Scalar::all(255));
    //输出格式
    cout << "R (default) = " << endl <<        R           << endl << endl;
    cout << "R (python)  = " << endl << format(R, Formatter::FMT_PYTHON) << endl << endl;
    cout << "R (csv)     = " << endl << format(R, Formatter::FMT_CSV   ) << endl << endl;
    cout << "R (numpy)   = " << endl << format(R, Formatter::FMT_NUMPY ) << endl << endl;
    cout << "R (c)       = " << endl << format(R, Formatter::FMT_C     ) << endl << endl;

    //opencv输出二维和三维格式
    Point2f p(1, 2);
    cout<<"Point (2D)="<<p<<endl;
    return 0;
}
