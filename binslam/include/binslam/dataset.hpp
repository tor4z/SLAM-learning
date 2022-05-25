#pragma once

#include "binslam/common.hpp"
#include "binslam/frame.hpp"
#include "binslam/camera.hpp"

namespace binslam
{

class Dataset
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Dataset> Ptr;
    Dataset(const std::string &dataset_path);

    bool init();
    Frame::Ptr nextFrame();
    Camera::Ptr getCamera(int camera_id) const
    {
        return cameras_.at(camera_id);
    }

private:
    std::string dataset_path_;
    int current_image_index_ = 0;
    std::vector<Camera::Ptr> cameras_;
};

}
