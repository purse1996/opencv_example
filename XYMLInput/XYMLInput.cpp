#include<opencv2/core.hpp>
#include<string>
#include<iostream>

using namespace std;
using namespace cv;

//定义存储在xml文件中自己的数据类型
class MyData
{
public:
    MyData() : A(0), X(0), id()
    {}
    explicit MyData(int) : A(97), X(CV_PI), id("mydata1234")
    {}
    void write(FileStorage& fs) const                        //为该类定义的写函数
    {
        fs << "{" << "A" << A << "X" << X << "id" << id << "}";
    }
    void read(const FileNode& node)                          //该类的写函数
    {
        A = (int)node["A"];
        X = (double)node["X"];
        id = (string)node["id"];
    }
public:
    int A;
    double X;
    string id;
};
//FileNode中将会调用这个Write函数进行自定义数据的写入
static void write(FileStorage& fs, const std::string&, const MyData& x)
{
    x.write(fs);
}
////FileNode中将会调用这个read函数进行自定义数据的读入
static void read(const FileNode& node, MyData& x, const MyData& default_value = MyData()){
    if(node.empty())
        x = default_value;
    else
        x.read(node);
}
// 重新定义<<操作符 从而使得<<可以操作对象
static ostream& operator<<(ostream& out, const MyData& m)
{
    out << "{ id = " << m.id << ", ";
    out << "X = " << m.X << ", ";
    out << "A = " << m.A << "}";
    return out;
}


int main(int argc, char **argv)
{

    string filename = "my.xml";
    FileStorage fs(filename, FileStorage::WRITE);
    fs << "iteration" <<100;
    // xml中第一种数据类型为element squence 在写入时 []位起始位置标志， <<位写入操作符
    fs << "strings" << "[";                              // text - string sequence
    fs << "image1.jpg" << "Awesomeness" << "../data/baboon.jpg";
    fs << "]";
    // xml第二种数据类型位mapping  类似与 key-value的结构 通过访问key来访问value
    fs << "mapping" <<"{";
    fs << "one" << 1 << "two" << 2 <<"}";

    //写入矩阵
    Mat R = Mat_<uchar>::eye(3, 3);
    fs << "R" << R;

    MyData m(1);
    fs << "Mydata" << m;
    // 完成写入后要释放
    fs.release();
    cout<<"write done"<<endl;

    cout<<endl<<"read"<<endl;

    fs.open(filename, FileStorage::READ);
    if(!fs.isOpened())
    {
        cerr<<"Failed to open"<<endl;
        return 1;
    }
    int itNr = (int) fs["iteration"];
    // 也可以写为 fs["iteration"] >> itNr;
    // FileStorage中数据存储在一个个FileNode当中
    FileNode s = fs["strings"];
    if(s.type()!=FileNode::SEQ)
    {
        cerr<<"strings is not a sequence"<<endl;
        return 1;
    }
    FileNodeIterator it=s.begin(), it_end = s.end();
    for(;it!=it_end; it++)
    {
        cout<<(string) *it <<endl;
    }

    s = fs["mapping"];
    cout<<"Two="<<(int) (s["two"])<<" "<<"one="<<(int)(s["one"])<<endl;

    // 将节点数据分别存储在矩阵和自定义类中
    Mat G;
    fs["R"] >> G;
    MyData data;
    fs["Mydata"] >> data;
    cout << endl<< "R = " << G << endl;
    cout << "MyData = " << endl << data << endl << endl;
    return 0;

}
