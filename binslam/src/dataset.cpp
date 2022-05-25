#include "binslam/dataset.hpp"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>

namespace binslam
{

static std::string build_image_path(
    const std::string dataset_path,
    int dir_id, int image_id
)
{
    std::stringstream ss;
    ss << dataset_path << "/image_"
       << dir_id << "/" << std::setfill('0')
       << std::setw(6) << image_id << ".png";
    return ss.str();
}


Dataset::Dataset(const std::string &dataset_path)
    : dataset_path_(dataset_path) {}

bool Dataset::init()
{
    std::ifstream fin(dataset_path_ + "/calib.txt");
    if(!fin)
    {
        LOG(ERROR) << "Cannot find " << dataset_path_ << "/calib.txt";
        return false;
    }

    for (size_t i = 0; i < 4; i++)
    {
        char camera_name[3];
        for (size_t k = 0; k < 3; k++)
            fin >> camera_name[k];

        double projection_data[12];
        for (size_t k = 0; k < 12; k++)
            fin >> projection_data[k];
        
        Mat33 K;
        Vec3 t;
        K << projection_data[0], projection_data[1], projection_data[2],
             projection_data[4], projection_data[5], projection_data[6],
             projection_data[8], projection_data[9], projection_data[10];
        t << projection_data[3], projection_data[7], projection_data[11];
        t = K.inverse() * t;
        K = K * 0.5;

        Camera::Ptr new_camera(
            new Camera(
                K(0, 0), K(1, 1), K(0, 2), K(1, 2),
                t.norm(), Sophus::SE3d(Sophus::SO3d(), t)
            )
        );

        cameras_.push_back(new_camera);

        LOG(INFO) << "Camera " << i << " extrinsics: " << t.transpose();
    }

    fin.close();
    current_image_index_ = 0;
    return true;
}

Frame::Ptr Dataset::nextFrame()
{
    cv::Mat left_image, right_image;
    left_image = cv::imread(
        build_image_path(dataset_path_, 0, current_image_index_),
        cv::IMREAD_GRAYSCALE
    );
    right_image = cv::imread(
        build_image_path(dataset_path_, 1, current_image_index_),
        cv::IMREAD_GRAYSCALE
    );

    if(left_image.empty() || right_image.empty())
    {
        LOG(WARNING) << "Cannot find image at index " << current_image_index_;
        return nullptr;
    }

    cv::Mat left_image_resized, right_image_resized;
    cv::resize(left_image, left_image_resized,
        cv::Size(), 0.5, 0.5, cv::INTER_NEAREST
    );
    cv::resize(right_image, right_image_resized,
        cv::Size(), 0.5, 0.5, cv::INTER_NEAREST
    );

    auto new_frame = Frame::create();
    new_frame->left_image_ = left_image_resized;
    new_frame->right_image_ = right_image_resized;
    ++current_image_index_;
    return new_frame;
}

}
