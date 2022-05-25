#include "binslam/map.hpp"
#include "binslam/feature.hpp"


namespace binslam
{

void Map::insertKeyFrame(Frame::Ptr frame)
{
    current_frame_ = frame;

    if(keyframes_.find(frame->keyframe_id_) == keyframes_.end())
    {
        // key frame not found
        keyframes_.insert(std::make_pair(frame->keyframe_id_, frame));
        active_keyframes_.insert(std::make_pair(frame->keyframe_id_, frame));
    }
    else
    {
        // frame already exist.
        keyframes_[frame->keyframe_id_] = frame;
        active_keyframes_[frame->keyframe_id_] = frame;
    }

    if(active_keyframes_.size() > num_active_keyframes_)
        removeOldKeyFrame();
}

void Map::insertMapPoint(MapPoint::Ptr map_point)
{
    if(landmarks_.find(map_point->id_) == landmarks_.end())
    {
        // landmark not found
        landmarks_.insert(std::make_pair(map_point->id_, map_point));
        active_landmarks_.insert(std::make_pair(map_point->id_, map_point));
    }
    else
    {
        // landmark already exist.
        landmarks_[map_point->id_] = map_point;
        active_landmarks_[map_point->id_] = map_point;
    }
}

void Map::removeOldKeyFrame()
{
    if(current_frame_ == nullptr) return;
    // find two keyframes closest and farthest to the current frame
    double max_dist = 0, min_dist = 99999;
    long max_kf_id = 0, min_kf_id = 0;
    auto Twc = current_frame_->pose().inverse();

    for (auto &kf: active_keyframes_)
    {
        if(kf.second == current_frame_) continue;
        auto dist = (kf.second->pose() * Twc).log().norm();
        if(dist > max_dist)
        {
            max_dist = dist;
            max_kf_id = kf.first;
        }
        if(dist < min_dist)
        {
            min_dist = dist;
            min_kf_id = kf.first;
        }
    }

    // threshold for closest keyframe
    const double min_dist_th = 0.2;
    Frame::Ptr frame_to_remove = nullptr;
    if(min_dist < min_dist_th)
    {
        // remove closest keyframe if exist
        frame_to_remove = keyframes_.at(min_kf_id);
    }
    else
    {
        // remove farthest keyframe
        frame_to_remove = keyframes_.at(max_kf_id);
    }

    LOG(INFO) << "Remove keyframe " << frame_to_remove->keyframe_id_;
    active_keyframes_.erase(frame_to_remove->keyframe_id_);

    for (auto feat: frame_to_remove->left_features_)
    {
        auto mp = feat->map_point_.lock();
        if(mp)
            mp->removeObservation(feat);
    }
    for (auto feat: frame_to_remove->right_features_)
    {
        auto mp = feat->map_point_.lock();
        if(mp)
            mp->removeObservation(feat);
    }

    cleanMap();
}

void Map::cleanMap()
{
    int removed_landmarks_cnt = 0;
    for (auto it = active_landmarks_.begin();
         it != active_landmarks_.end();)
    {
        if(it->second->observed_times_ == 0)
        {
            it = active_landmarks_.erase(it);
            ++removed_landmarks_cnt;
        }
        else
        {
            ++it;
        }
    }

    LOG(INFO) << "Remove " << removed_landmarks_cnt << "active landmarks.";
}

}
