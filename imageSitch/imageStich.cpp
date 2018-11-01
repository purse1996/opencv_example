#include<opencv2/core.hpp>
#include<opencv2/stitching.hpp>
#include<opencv2/stitching.hpp/stitcher.hpp>
using namespace std;
using namespace cv;

int mian()
{
    Mat pano;
    Stitcher stitcher = Stitcher::createDefault(false);
    Stitcher::Status status = stitcher.stitch(imgs, pano);
    return 0;
}
