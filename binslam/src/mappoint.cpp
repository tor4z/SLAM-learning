#include "binslam/mappoint.hpp"
#include "binslam/feature.hpp"

namespace binslam
{

MapPoint::MapPoint(
    long id, const Vec3 &position
): id_(id), pos_(position) {}

MapPoint::Ptr MapPoint::create()
{
    static long factory_id = 0;
    MapPoint::Ptr new_mappoint(new MapPoint);
    new_mappoint->id_ = factory_id++;
    return new_mappoint;
}

void MapPoint::removeObservation(std::shared_ptr<Feature> feature)
{
    std::unique_lock<std::mutex> lck(data_mutex_);
    for (auto it = observations_.begin(); it != observations_.end(); ++it)
    {
        if(it->lock() == feature)
        {
            observations_.erase(it);
            feature->map_point_.reset();
            --observed_times_;
            break;
        }
    }
}

}
