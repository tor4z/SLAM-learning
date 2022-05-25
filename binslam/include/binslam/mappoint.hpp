#pragma once

#include "binslam/common.hpp"

namespace binslam
{

struct Frame;
struct Feature;

struct MapPoint
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<MapPoint> Ptr;
    unsigned long id_ = 0;
    bool is_outlier_ = false;
    // position in world
    Vec3 pos_ = Vec3::Zero();
    std::mutex data_mutex_;
    // being observed by feature matching algo
    int observed_times_ = 0;
    std::list<std::weak_ptr<Feature>> observations_;

    MapPoint() {}
    MapPoint(long id, const Vec3 &position);

    Vec3 pos()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return pos_;
    }

    void setPos(const Vec3 &pos)
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        pos_ = pos;
    }

    void addObservation(std::shared_ptr<Feature> feature)
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        observations_.push_back(feature);
        ++observed_times_;
    }

    void removeObservation(std::shared_ptr<Feature> feature);

    std::list<std::weak_ptr<Feature>> getObs()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        return observations_;
    }

    static MapPoint::Ptr create();
};

}
