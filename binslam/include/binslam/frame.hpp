#pragma once

#include "binslam/common.hpp"


namespace binslam
{

struct MapPoint;
struct Feature;

struct Frame
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frame> Ptr;

    unsigned long id_ = 0;
    unsigned long keyframe_id_ = 0;
    bool is_keyframe_ = false;
    double timestamp_;
    Sophus::SE3d pose_;
    std::mutex pose_mutex_;
    cv::Mat left_image_, right_image_;

    std::vector<std::shared_ptr<Feature>> left_features_;
    std::vector<std::shared_ptr<Feature>> right_features_;

    Frame() {}
    Frame(
        long id, double timestamp, const Sophus::SE3d &pose,
        const cv::Mat &left, const cv::Mat &right
    );
    Sophus::SE3d pose()
    {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        return pose_;
    }

    void setPose(const Sophus::SE3d &pose)
    {
        std::unique_lock<std::mutex> lck(pose_mutex_);
        pose_ = pose;
    }

    // set key frame and assign key frame id
    void setKeyFrame();

    // create frame with factory mode
    static std::shared_ptr<Frame> create();
};


}
