#include<opencv2/core.hpp>
#include<opencv2/imgcodecs.hpp>
#include<opencv2/highgui.hpp>

#include<iostream>
#include<string>
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    String imageName;
    if(argc>1)
    {
        imageName = argv[1];
    }
    Mat image;
    image = imread(imageName, IMREAD_COLOR);
    /*
    IMREAD_UNCHANGED (<0) loads the image as is (including the alpha channel if present)
    IMREAD_GRAYSCALE ( 0) loads the image as an intensity one
    IMREAD_COLOR (>0) loads the image in the RGB format
    */


    if(image.empty())
    {
        cout<<"couldn't open or find the image"<<std::endl;
        return -1;
    }
    else
    {
        namedWindow("display image", WINDOW_AUTOSIZE);
        imshow("display window", image);
        waitKey(0);
        return 0;
    }
}

