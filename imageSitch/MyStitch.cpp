#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/stitching.hpp>
#include<opencv2/imgproc.hpp>
using namespace std;
using namespace cv;


int main()
{
    vector<Mat> imgs;
    img = imread("1.jpg");
    imgs.push_back(img);
    img = imread("2.jpg");
    imgs.push_back(img);
    img = imread("3.jpg");
    imgs.push_back(img);

    return 0;
}
