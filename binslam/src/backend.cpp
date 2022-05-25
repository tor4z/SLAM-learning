#include "binslam/backend.hpp"
#include "binslam/algorithm.hpp"
#include "binslam/feature.hpp"
#include "binslam/g2o_types.hpp"
#include "binslam/map.hpp"
#include "binslam/mappoint.hpp"
#include <functional>


namespace binslam
{

Backend::Backend()
{
    backend_running_.store(true);
    backend_thread_ = std::thread(
        std::bind(
            &Backend::backendLoop, this
        )
    );
}

void Backend::updateMap()
{
    std::unique_lock<std::mutex> lck(data_mutex_);
    map_update_.notify_one();
}

void Backend::stop()
{
    backend_running_.store(false);
    map_update_.notify_one();
    backend_thread_.join();
}

void Backend::backendLoop()
{
    while (backend_running_.load())
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        map_update_.wait(lck);

        Map::keyframesType active_kfs = map_->getActiveKeyFrames();
        Map::LandmarksType active_landmarks = map_->getActiveMapPoints();
        optimize(active_kfs, active_landmarks);
    }
}

void Backend::optimize(
    Map::keyframesType &keyframes,
    Map::LandmarksType &landmarks
)
{
    using BlockSolverType = g2o::BlockSolver_6_3;
    using LinearSolverType = 
        g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType>;
    
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(
            g2o::make_unique<LinearSolverType>()
        )
    );

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    std::map<unsigned long, VertexPose*> vertices;
    unsigned long max_kf_id = 0;

    for (auto &keyframe: keyframes)
    {
        auto kf = keyframe.second;
        // camera vertex_pose
        VertexPose *vertex_pose = new VertexPose();
        vertex_pose->setId(kf->keyframe_id_);
        vertex_pose->setEstimate(kf->pose());
        optimizer.addVertex(vertex_pose);
        if(kf->keyframe_id_ > max_kf_id)
        {
            max_kf_id = kf->keyframe_id_;
        }
        vertices.insert({kf->keyframe_id_, vertex_pose});
    }

    std::map<unsigned long, VertexXYZ*> landmark_vertices;
    Mat33 K = left_cam_->K();
    Sophus::SE3d left_ext = left_cam_->pose();
    Sophus::SE3d right_ext = right_cam_->pose();

    int index = 1;
    double chi2_th = 5.991;
    std::map<EdgeProjection*, Feature::Ptr> edges_and_features;

    for (auto &landmark: landmarks)
    {
        if(landmark.second->is_outlier_) continue;

        unsigned long landmark_id = landmark.second->id_;
        auto observations = landmark.second->getObs();

        for (auto &obs: observations)
        {
            auto feat = obs.lock();
            if (feat == nullptr) continue;
            auto frame = feat->frame_.lock();
            if (feat->is_outlier_ || frame == nullptr) continue;

            EdgeProjection *edge = nullptr;
            if(feat->is_on_left_image_)
            {
                edge = new EdgeProjection(K, left_ext);
            }
            else
            {
                edge = new EdgeProjection(K, right_ext);
            }

            if(landmark_vertices.find(landmark_id) ==
               landmark_vertices.end()
            )
            {
                // not found
                VertexXYZ *v = new VertexXYZ;
                v->setEstimate(landmark.second->pos());
                v->setId(landmark_id + max_kf_id + 1);
                v->setMarginalized(true);
                landmark_vertices.insert({landmark_id, v});
                optimizer.addVertex(v);
            }

            auto rk = new g2o::RobustKernelHuber();
            rk->setDelta(chi2_th);

            edge->setId(index);
            edge->setVertex(0, vertices.at(frame->keyframe_id_));
            edge->setVertex(1, landmark_vertices.at(landmark_id));
            edge->setMeasurement(toVec2(feat->position_.pt));
            edge->setInformation(Mat22::Identity());
            edge->setRobustKernel(rk);
            edges_and_features.insert({edge, feat});

            optimizer.addEdge(edge);

            ++index;
        }
    }

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    int outlier_cnt = 0;
    int inlier_cnt = 0;
    int interation = 0;

    while (interation < 5)
    {
        outlier_cnt = 0;
        inlier_cnt = 0;

        for (auto &ef: edges_and_features)
        {
            if(ef.first->chi2() > chi2_th)
            {
                ++outlier_cnt;
            }
            else
            {
                ++inlier_cnt;
            }
        }

        double inlier_ratio = inlier_cnt / 
            static_cast<double>(inlier_cnt + outlier_cnt);
        
    }
    
}

}
