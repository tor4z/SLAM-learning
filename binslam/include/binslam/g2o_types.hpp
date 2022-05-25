#pragma once

#include "binslam/common.hpp"

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

namespace binslam
{

class VertexPose: public g2o::BaseVertex<6, Sophus::SE3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    virtual void setToOriginImpl() override
    {
        _estimate = Sophus::SE3d();
    }

    virtual void oplusImpl(const double *update) override
    {
        Vec6 update_edge;
        update_edge << update[0], update[1], update[2],
                       update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(update_edge) * _estimate;
    }

    virtual bool read(std::istream &in) override
    {
        return true;
    }

    virtual bool write(std::ostream &out) const override
    {
        return true;
    }
};

class VertexXYZ: public g2o::BaseVertex<3, Vec3>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    virtual void setToOriginImpl() override
    {
        _estimate = Vec3::Zero();
    }

    virtual void oplusImpl(const double *update) override
    {
        _estimate[0] = update[0];
        _estimate[1] = update[1];
        _estimate[2] = update[2];
    }

    virtual bool read(std::istream &in) override
    {
        return true;
    }

    virtual bool write(std::ostream &out) const override
    {
        return true;
    }
};

class EdgeProjectionPoseOnly: public g2o::BaseUnaryEdge<2, Vec2, VertexPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeProjectionPoseOnly(const Vec3 &pos, const Mat33 &k)
        : pos3d_(pos), K_(k) {}

    virtual void computeError() override
    {
        const VertexPose *v = static_cast<VertexPose*>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Vec3 pos_pixel = K_ * (T * pos3d_);
        pos_pixel /= pos_pixel[2];
        _error = _measurement - pos_pixel.head<2>();
    }

    virtual void linearizeOplus() override
    {
        const VertexPose *v = static_cast<VertexPose*>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Vec3 pos_cam = T * pos3d_;
        double fx = K_(0, 0);
        double fy = K_(1, 1);
        double X = pos_cam[0];
        double Y = pos_cam[1];
        double Z = pos_cam[2];
        double Z_inv = 1.0 / (Z - 1e-18);
        double Z_inv2 = Z_inv * Z_inv;

        _jacobianOplusXi <<
            -fx * Z_inv, 0, fx * X * Z_inv2, fx * X * Y * Z_inv2,
            -fx - fx * X * X * Z_inv2, fx * Y * Z_inv, 0, -fy * Z_inv,
            fy * Y * Z_inv2, fy + fy * Y * Y * Z_inv2, -fy * X * Y * Z_inv2, -fy * X * Z_inv;
    }

    virtual bool read(std::istream &in) override
    {
        return true;
    }

    virtual bool write(std::ostream &out) const override
    {
        return true;
    }
private:
    Vec3 pos3d_;
    Mat33 K_;
};

class EdgeProjection: public g2o::BaseBinaryEdge<2, Vec2, VertexPose, VertexXYZ>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    
    EdgeProjection(const Mat33 &k, const Sophus::SE3d &cam_ext)
        : K_(k), cam_ext_(cam_ext) {}
    
    virtual void computeError() override
    {
        const VertexPose *v0 = static_cast<VertexPose*>(_vertices[0]);
        const VertexXYZ *v1 = static_cast<VertexXYZ*>(_vertices[1]);

        Sophus::SE3d T = v0->estimate();
        Vec3 pos_pixel = K_ * (cam_ext_ * (T * v1->estimate()));
        pos_pixel /= pos_pixel[2];
        _error = _measurement - pos_pixel.head<2>();
    }

    virtual void linearizeOplus() override
    {
        const VertexPose *v0 = static_cast<VertexPose*>(_vertices[0]);
        const VertexXYZ *v1 = static_cast<VertexXYZ*>(_vertices[1]);

        Sophus::SE3d T = v0->estimate();
        Vec3 pw = v1->estimate();
        Vec3 pos_cam = cam_ext_ * T * pw;

        double fx = K_(0, 0);
        double fy = K_(1, 1);
        double X = pos_cam[0];
        double Y = pos_cam[1];
        double Z = pos_cam[2];
        double Z_inv = 1.0 / (Z + 1e-18);
        double Z_inv2 = Z_inv * Z_inv;

        _jacobianOplusXi <<
            -fx * Z_inv, 0, fx * X * Z_inv2, fx * X * Y *Z_inv2,
            -fx - fx * X * X * Z_inv2, fx * Y * Z_inv, 0, -fy * Z_inv,
            fy * Y * Z_inv2, fy + fy * Y * Y * Z_inv2, -fy * X * Y *Z_inv2, -fy * X * Z_inv;

        _jacobianOplusXj =
            _jacobianOplusXi.block<2, 3>(0, 0) *
            cam_ext_.rotationMatrix() *
            T.rotationMatrix();
    }

    virtual bool read(std::istream &in) override
    {
        return true;
    }

    virtual bool write(std::ostream &out) const override
    {
        return true;
    }
private:
    Mat33 K_;
    Sophus::SE3d cam_ext_;
};

}
