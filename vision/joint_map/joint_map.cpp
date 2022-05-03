#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>


typedef std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;
typedef Eigen::Matrix<double, 6, 1> Vector6d;


void show_point_cloud(
    const std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>>& point_cloud
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
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -0.1, 0.0)
    );
    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));
    
    while (!pangolin::ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p: point_cloud)
        {
            glColor3d(p[3] / 255.0, p[4] / 255.0, p[5] / 255.0);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);  // sleep 5ms
    }
}


int main(int argc, char** argv)
{
    std::vector<cv::Mat> color_images, depth_images;
    TrajectoryType poses;

    std::ifstream fin("../asset/pose.txt");
    if(!fin)
    {
        std::cerr << "" << std::endl;
        return 1;
    }

    // read poses
    std::stringstream color_ss, depth_ss;
    for (size_t i = 0; i < 5; i++)
    {
        color_ss << "../asset/color/" << i + 1 << ".png";
        depth_ss << "../asset/depth/" << i + 1 << ".png";
        color_images.push_back(cv::imread(color_ss.str()));
        depth_images.push_back(cv::imread(depth_ss.str()));
        color_ss.clear();
        depth_ss.clear();

        double data[7] = {0};
        for (auto &p: data)
            fin >> p;
        Sophus::SE3d pose(
            Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
            Eigen::Vector3d(data[0], data[1], data[2])
        );
        poses.push_back(pose);
    }

    double cx, cy, fx, fy;
    double depth_scale;
    std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> point_cloud;
    point_cloud.reserve(1000000);

    for (size_t i = 0; i < 5; i++)
    {
        cv::Mat color_image = color_images[i];
        cv::Mat depth_image = depth_images[i];
        Sophus::SE3d T = poses[i];

        for (size_t v = 0; v < color_image.rows; v++)
        {
            for (size_t u = 0; u < color_image.cols; u++)
            {
                uchar depth_value = depth_image.ptr<uchar>(v)[u];
                if (depth_value == 0) continue;
                Eigen::Vector3d point;
                point[2] = static_cast<double>(depth_value) / depth_scale;
                point[1] = point[2] * (v - cy) / fy;
                point[0] = point[2] * (u - cx) / fx;
                Eigen::Vector3d point_world = T * point;

                Vector6d p;
                p.head<3>() = point_world;
                // red
                p[3] = color_image.data[
                    v * color_image.step + u * color_image.channels() + 2];
                // green
                p[2] = color_image.data[
                    v * color_image.step + u * color_image.channels() + 1];
                // blue
                p[1] = color_image.data[
                    v * color_image.step + u * color_image.channels() + 0];
                point_cloud.push_back(p);
            }
        }
    }

    std::cout << "Point cloud size: " << point_cloud.size() << std::endl;
    show_point_cloud(point_cloud);

    return 0;
}
