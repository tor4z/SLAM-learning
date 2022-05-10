#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>


class CurveFittingVertex: public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    virtual void setToOriginImpl() override
    {
        _estimate << 0, 0, 0;
    }

    virtual void oplusImpl(const double* update) override
    {
        _estimate += Eigen::Vector3d(update);
    }

    virtual bool read(std::istream& in) {return true;}
    virtual bool write(std::ostream& out) const {return true;}
};


class CurveFittingEdge: public g2o::BaseUnaryEdge<1, double, CurveFittingVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    CurveFittingEdge(double x): BaseUnaryEdge(), x_(x) {}
    
    virtual void computeError() override
    {
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex*>(_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        _error(0, 0) = _measurement - std::exp(
            abc(0, 0) * x_ * x_ + abc(1, 0) * x_ + abc(2, 0)
        );
    }

    // compute jacobi matrix
    virtual void linearizeOplus() override
    {
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex*>(_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        double y = std::exp(
            abc[0] * x_ * x_ + abc[1] * x_ + abc[2]
        );
        _jacobianOplusXi[0] = -x_ * x_ * y;
        _jacobianOplusXi[1] = -x_ * y;
        _jacobianOplusXi[2] = -y;
    }

    virtual bool read(std::istream& in) {return true;}
    virtual bool write(std::ostream& out) const {return true;}
private:
    double x_;
};


int main(int argc, char** argv)
{
    double ar = 1.0, br = 2.0, cr = 1.0;
    double ae = 1.0, be = -1.0, ce = 5.0;
    double w_sigma = 1.0;
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng;
    int N = 100;

    std::vector<double> x_data, y_data;
    for (size_t i = 0; i < N; i++)
    {
        double x = i / 100.0;
        double y = std::exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma);
        x_data.push_back(x);
        y_data.push_back(y);
    }

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>())
    );
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // add vertex to graph
    CurveFittingVertex *v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(ae, be, ce));
    v->setId(0);
    optimizer.addVertex(v);

    // add edge to graph
    for (size_t i = 0; i < N; i++)
    {
        CurveFittingEdge *e = new CurveFittingEdge(x_data[i]);
        e->setId(i);
        e->setVertex(0, v);
        e->setMeasurement(y_data[i]);
        e->setInformation(
            Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma)
        ); // information matrix: covariance matrix
        optimizer.addEdge(e);
    }
    
    std::cout << "optimization start ..." << std::endl;
    std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start);
    std::cout << "optimization time cost: "
        << time_used.count()
        << "s" << std::endl;
    
    Eigen::Vector3d abc_estimate = v->estimate();
    std::cout << "Estimated model: "
        << abc_estimate.transpose() << std::endl;

    return 0;
}
