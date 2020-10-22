#include "ctfLK.h"
#include "LK.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;
int main()
{
    VideoCapture capture;

    Mat frame, drawFrame;
    Mat prevFrame;
    Mat nextFrame;
    capture.open(0);

    capture.set(cv::CAP_PROP_FRAME_WIDTH, 4000);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 1880);
    int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);

    std::cout << width << "," << height;

    vector<Point2f> featurePrev;

    int maxCoutFeature = 200;      // 最大特征点数量
    double minDisFeature = 10;     // 特征点最小距离
    double qLevelFeature = 0.01;   // 质量水平
    vector<Point2f> featureNext;
    Mat featureStatus, featureError;

    int refreshCount = 100;
    capture >> frame;
    while(1)
    {
        // 两帧图像
        if (frame.empty())
            break;
        cvtColor(frame, prevFrame, COLOR_BGR2GRAY);
        capture >> frame;
        cvtColor(frame, nextFrame, COLOR_BGR2GRAY);
        frame.copyTo(drawFrame);
        if (frame.empty())
            break;

        auto countFeature = featureStatus.empty() ? 0 : std::count(featureStatus.begin<uchar>(), featureStatus.end<uchar>(), 1);

        if (--refreshCount <= 0 || countFeature < 150)
        {
            refreshCount = 100;
            goodFeaturesToTrack(prevFrame, featurePrev, maxCoutFeature, qLevelFeature, minDisFeature);
        }
        else
            featurePrev = featureNext;
        // 金字塔LK光流算法
        calcOpticalFlowPyrLK(prevFrame, nextFrame, featurePrev, featureNext, featureStatus, featureError);

        // 标记特征点位移
        for (int i = 0; i < featureNext.size(); i++)
        {
            circle(drawFrame, featureNext[i], 3, Scalar(255), 1);
            if (featureStatus.at<bool>(i))
                line(drawFrame, featurePrev[i], featureNext[i], Scalar(1));
        }
        putText(drawFrame, "Features:" + std::to_string(countFeature) + "/" + std::to_string(featurePrev.size()), Point(0, 40), FONT_HERSHEY_SIMPLEX, 0.5, countFeature < 150 ? Scalar(0, 0, 255) : Scalar(0, 255, 255));
        imshow("capture", drawFrame);
        waitKey(20);
    }
    //// 输出位移情况
    //for (int i = 0; i < featureStatus.rows; i++)
    //{
    //    std::cout << "(" << featurePyrLK[i].x << "," << featurePyrLK[i].y << 
    //        ")->(" << featurePrev[i].x << "," << featurePrev[i].y << ")" << std::endl;
    //}
    //std::cout << "prev pts:" << featurePyrLK.size() << std::endl;;
    //std::cout << "next pts:" << featurePrev.size() << std::endl;
    //std::cout << "status size：" << featureStatus.rows << std::endl;

    //imshow("img1", prevFrame);
    //imshow("img2", nextFrame);
    //waitKey(0);
    return 0;
}