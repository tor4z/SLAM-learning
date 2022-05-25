#pragma once

#include "binslam/common.hpp"
#include "binslam/frame.hpp"
#include "binslam/mappoint.hpp"


namespace binslam
{

class Map
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Map> Ptr;
    typedef std::unordered_map<unsigned long, MapPoint::Ptr> LandmarksType;
    typedef std::unordered_map<unsigned long, Frame::Ptr> keyframesType;

    Map() {}

    void insertKeyFrame(Frame::Ptr frame);
    void insertMapPoint(MapPoint::Ptr map_point);

    LandmarksType getAllMapPoint()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return landmarks_;
    }

    keyframesType getAllKeyFrame()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return keyframes_;
    }

    LandmarksType getActiveMapPoints()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_landmarks_;
    }

    keyframesType getActiveKeyFrames()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return active_keyframes_;
    }

    void cleanMap();

private:
    void removeOldKeyFrame();

    std::mutex data_mutex_;
    // all landmarks
    LandmarksType landmarks_;
    // activated landmarks
    LandmarksType active_landmarks_;
    // all keyframe
    keyframesType keyframes_;
    // activated keyframes
    keyframesType active_keyframes_;

    Frame::Ptr current_frame_ = nullptr;

    // activated keyframe
    int num_active_keyframes_ = 7;
};

}
