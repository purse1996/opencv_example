#include<iostream>
#include<sstream>
#include<opencv2/core.hpp>
#include<opencv2/core/utility.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>
using namespace std;
using namespace cv;

Mat& color_reduce(Mat &I, uchar *table);
Mat& iter_color_reduce(Mat &I, uchar *table);
Mat& table_define(Mat &table, int quan_val);
/*访问每个元素
Mat& ScanImageAndReduceRandomAccess(Mat& I, const uchar* const table)
{
    // accept only char type matrices
    CV_Assert(I.depth() == CV_8U);
    const int channels = I.channels();
    switch(channels)
    {
    case 1:
        {
            for( int i = 0; i < I.rows; ++i)
                for( int j = 0; j < I.cols; ++j )
                    I.at<uchar>(i,j) = table[I.at<uchar>(i,j)];
            break;
        }
    case 3:
        {
         Mat_<Vec3b> _I = I;
         for( int i = 0; i < I.rows; ++i)
            for( int j = 0; j < I.cols; ++j )
               {
                   _I(i,j)[0] = table[_I(i,j)[0]];
                   _I(i,j)[1] = table[_I(i,j)[1]];
                   _I(i,j)[2] = table[_I(i,j)[2]];
            }
         I = _I;
         break;
        }
    }
    return I;
*/
//quan_val是量化等级
int main(int argc, char *argv[])
{
    if(argc<3)
    {
        cout<<"not enough parameters"<<endl;
        return -1;
    }
    int div  =0;
    //如何从输入流中获取数字
    stringstream s;
    s << argv[2];
    s >> div;
    Mat I = imread(argv[1], IMREAD_COLOR);
    Mat J;
    uchar table[256];
    for (int i=0; i<256; i++)
    {

        table[i] = (uchar)(div * (i/div));
    }
    const int time = 100;
    double t;
    t = (double)getTickCount();
    //重复100次，计算损耗时间
    for(int i=0; i<time; i++)
    {
        Mat clone_i = I.clone();
        J = color_reduce(clone_i, table);
    }
    //getTickCount统计了耗费多少系统周期
    //getTickFrequency cpu运行周期的频率
    //1000得到的单位是ms
    t = 1000*((double)getTickCount() - t)/getTickFrequency();
    cout<<"scan color reduce"<<time<<"times"<<t<<"ms"<<endl<<endl;

    t = (double)getTickCount();
    //重复100次，计算损耗时间
    for(int i=0; i<time; i++)
    {
        Mat clone_i = I.clone();
        J = iter_color_reduce(clone_i, table);
    }
    //getTickCount统计了耗费多少系统周期
    //getTickFrequency cpu运行周期的频率
    //1000得到的单位是ms
    t = 1000*((double)getTickCount() - t)/getTickFrequency();
    cout<<"iterator color reduce"<<time<<"times"<<t<<"ms"<<endl<<endl;

    Mat newtable, out;
    table_define(newtable, div);
    t = (double)getTickCount();
    //重复100次，计算损耗时间
    for(int i=0; i<time; i++)
    {
        Mat clone_i = I.clone();
        LUT(clone_i, newtable, out);
    }
    //getTickCount统计了耗费多少系统周期
    //getTickFrequency cpu运行周期的频率
    //1000得到的单位是ms
    t = 1000*((double)getTickCount() - t)/getTickFrequency();
    cout<<"look up table color reduce"<<time<<"times"<<t<<"ms"<<endl<<endl;




 //   namedWindow("display window");
  //  imshow("display window", J);
    //waitKey();
    return 0;

}
Mat& color_reduce(Mat &I, uchar *table)
{

    int channels = I.channels();
    int nRows = I.rows;
    int nCols = I.cols*channels;
    if(I.isContinuous())
    {
        nCols *= nRows;
        nRows  = 1;
    }
    uchar *p;
    for(int i=0; i<nRows; i++)
    {
        p = I.ptr<uchar>(i);//获取第i行的指针
        for(int j=0; j<nCols;j++)
        {
            p[j] = table[p[j]];
        }

    }
    return I;

}

Mat & iter_color_reduce(Mat &I, uchar *table)
{
    int channels = I.channels();
    switch(channels)
    {
    case 1:
        {
            MatIterator_<uchar> it, end_l;
            //MatItertator是一个迭代器
            for(it=I.begin<uchar>(), end_l=I.end<uchar>(); it!=end_l; it++)
            {
                *it = table[*it];
            }
            break;
        }
    case 3:
        {
            MatIterator_<Vec3b> it, end_l;
            //vec3d是一种数据类型 3通道uchar类型
                for(it=I.begin<Vec3b>(), end_l=I.end<Vec3b>(); it!=end_l; it++)
                {
                    (*it)[0] = table[(*it)[0]];
                    (*it)[1] = table[(*it)[1]];
                    (*it)[2] = table[(*it)[2]];
                }
        }
    }
    return I;
}

Mat& table_define(Mat &table, int quan_val)
{
    table.create(1, 256, CV_8UC1);
    uchar *p = table.data;//获取首行首列的指针
    for(int i=0; i<256; i++)
    {
        p[i] = quan_val * (i/quan_val);
    }
    return table;

}
