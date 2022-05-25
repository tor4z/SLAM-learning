#pragma once

#include "binslam/common.hpp"


namespace binslam
{

class Camera
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Camera> Ptr;

    Camera();
    Camera(
        double fx, double fy,
        double cx, double cy,
        double baseline,
        const Sophus::SE3d &pose
    ): fx_(fx), fy_(fy), cx_(cx), cy_(cy),
       baseline_(baseline), pose_(pose)
    {
        pose_inv_ = pose_.inverse();
    }

    Sophus::SE3d pose() { return pose_; }

    Mat33 K() const 
    {
        Mat33 k;
        k << fx_, 0,   cx_,
             0,   fy_, cy_,
             0,   0,   1;
        return k;
    }

    Vec3 world2camera(const Vec3 &p_w, const Sophus::SE3d &T_c_w);
    Vec3 camera2world(const Vec3 &p_c, const Sophus::SE3d &T_c_w);
    Vec2 camera2pixel(const Vec3 &p_c);
    Vec3 pixel2camera(const Vec2 &p_p, double depth = 1);
    Vec3 pixel2world(const Vec2 &p_p, const Sophus::SE3d &T_c_w, double depth = 1);
    Vec2 world2pixel(const Vec3 &p_w, const Sophus::SE3d &T_c_w);

private:
    // camera intrinsics
    double fx_ = 0,
           fy_ = 0,
           cx_ = 0,
           cy_ = 0,
           baseline_ = 0;

    // extrinsic, from stereo camera to single camera
    Sophus::SE3d pose_;
    Sophus::SE3d pose_inv_;
};

}
