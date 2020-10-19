#include "ctfLK.h"
#include "LK.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

void saveMat(Mat& M, string s) {
    s += ".txt";
    FILE* pOut = fopen(s.c_str(), "w+");
    for (int i = 0; i < M.rows; i++) {
        for (int j = 0; j < M.cols; j++) {
            fprintf(pOut, "%lf", M.at<double>(i, j));
            if (j == M.cols - 1) fprintf(pOut, "\n");
            else fprintf(pOut, " ");
        }
    }
    fclose(pOut);
}
using namespace std;
using namespace cv;
int main()
{
    // 两帧图像
    Mat prevFrame;
    Mat nextFrame;
    resize(imread("mouse1.jpg", IMREAD_GRAYSCALE), prevFrame, Size(600, 600));
    resize(imread("mouse2.jpg", IMREAD_GRAYSCALE), nextFrame, Size(600, 600));


    vector<Point2f> featurePrev;
    Mat featureStatus, featureError;
    vector<Point2f> featurePyrLK;

    int maxCoutFeature = 100;      // 最大特征点数量
    double minDisFeature = 20;     // 特征点最小距离
    double qLevelFeature = 0.05;   // 质量水平
    // 提取特征点
    goodFeaturesToTrack(prevFrame, featurePyrLK, maxCoutFeature, qLevelFeature, minDisFeature);

    // 前帧图像标记特征点
    for (int i = 0; i < featurePyrLK.size(); i++)
    {
        circle(prevFrame, featurePyrLK[i], 3, Scalar(0), 1);
    }

    // 金字塔LK光流算法
    calcOpticalFlowPyrLK(prevFrame, nextFrame, featurePyrLK, featurePrev, featureStatus, featureError);

    // 标记特征点位移
    for (int i = 0; i < featurePrev.size(); i++)
    {
        circle(nextFrame, featurePrev[i], 3, Scalar(255), 1);
        if(featureStatus.at<bool>(i))
            line(nextFrame, featurePyrLK[i], featurePrev[i], Scalar(1));
    }

    // 输出位移情况
    for (int i = 0; i < featureStatus.rows; i++)
    {
        std::cout << "(" << featurePyrLK[i].x << "," << featurePyrLK[i].y << 
            ")->(" << featurePrev[i].x << "," << featurePrev[i].y << ")" << std::endl;
    }
    std::cout << "prev pts:" << featurePyrLK.size() << std::endl;;
    std::cout << "next pts:" << featurePrev.size() << std::endl;
    std::cout << "status size：" << featureStatus.rows << std::endl;

    imshow("img1", prevFrame);
    imshow("img2", nextFrame);
    waitKey(0);
    return 0;
}