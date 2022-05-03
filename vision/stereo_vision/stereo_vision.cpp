#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <pangolin/pangolin.h>


const std::string left_image_path = "../asset/left.png";
const std::string right_image_path = "../asset/right.png";


void show_point_cloud(
    const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>& point_cloud
)
{
    if(point_cloud.empty())
    {
        std::cerr << "" << std::endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 0.1, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));
    
    while (!pangolin::ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glPointSize(2);
        glBegin(GL_POINT);
        for (auto &p: point_cloud)
        {
            glColor3f(p[3], p[3], p[3]);
            glVertex3d(p[0], p[1], p[2]);
        }
        
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5ms
    }
}


int main(int argc, char** argv)
{
    // intrinsic param
    double fx, fy, cx, cy;
    // baseline
    double b = 0.573;

    cv::Mat left = cv::imread(left_image_path);
    cv::Mat right = cv::imread(right_image_path);

    if(left.empty() || right.empty())
    {
        std::cerr << "" << std::endl;
        return 1;
    }

    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32
    );

    cv::Mat disparity_sbgm, disparity;
    sgbm->compute(left, right, disparity_sbgm);
    disparity_sbgm.convertTo(disparity, CV_32F, 1.0/16.0f);

    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> point_cloud;

    for (size_t i = 0; i < left.rows; i++)
    {
        for (size_t j = 0; j < left.cols; j++)
        {
            if(disparity.at<float>(i, j) <= 0 ||
               disparity.at<float>(i, j) >= 96.0
            )
                continue;
            
            Eigen::Vector4d point(0, 0, 0, left.at<uchar>(i, j)/255.0);
            double x = (j - cx) / fx;
            double y = (i - cy) / fy;
            double depth = fx * b / disparity.at<float>(i, j);

            point[0] = x * depth;
            point[1] = y * depth;
            point[2] = depth;

            point_cloud.push_back(point);
        }
    }
    
    cv::imshow("disparity", disparity / 96.0);
    cv::waitKey(0);
    show_point_cloud(point_cloud);

    return 0;
}
