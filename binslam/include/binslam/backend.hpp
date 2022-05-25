#pragma once

#include "binslam/common.hpp"
#include "binslam/frame.hpp"
#include "binslam/map.hpp"
#include "binslam/camera.hpp"

namespace binslam
{

class Map;

class Backend
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Backend> Ptr;

    Backend();
    void setCameras(Camera::Ptr left, Camera::Ptr right)
    {
        left_cam_ = left;
        right_cam_ = right;
    }

    void setMap(std::shared_ptr<Map> map)
    {
        map_ = map;
    }

    void updateMap();

    void stop();
private:
    void backendLoop();
    void optimize(Map::keyframesType &keyframes, Map::LandmarksType &landmarks);

    std::shared_ptr<Map> map_;
    std::thread backend_thread_;
    std::mutex data_mutex_;

    std::condition_variable map_update_;
    std::atomic<bool> backend_running_;

    Camera::Ptr left_cam_ = nullptr;
    Camera::Ptr right_cam_ = nullptr;
};

}
