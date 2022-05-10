#include <iostream>
#include <chrono>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>



class CurveFittingCost
{
public:
    CurveFittingCost(double x, double y): x_(x), y_(y){}
    template<typename T>
    bool operator()(const T* const abc, T* residual) const
    {
        residual[0] = T(y_) - ceres::exp(
            abc[0] * T(x_) * T(x_) + abc[1] * T(x_) + abc[2]
        );
        return true;
    }
private:
    double x_, y_;
};



int main(int argc, char** argv)
{
    double ar = 1.0, br = 2.0, cr = 1.0;
    double ae = 2.0, be = -1.0, ce = 5.0;
    int N = 100;
    double w_sigma = 1.0;
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng;

    std::vector<double> x_data, y_data;
    for (int i = 0; i < N; i++)
    {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(
            exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma)
        );
    }

    double abc[3] = {ae, be, ce};

    ceres::Problem problem;
    for (size_t i = 0; i < N; i++)
    {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CurveFittingCost, 1, 3>(
                new CurveFittingCost(x_data[i], y_data[i])
            ),
            nullptr,       // kernel function
            abc            // param to estimate
        );
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);
    std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start);
    std::cout << "Time used: " << time_used.count() << "s" << std::endl;

    std::cout << summary.BriefReport() << std::endl;

    std::cout << "Estimation for a,b,c is:";
    for (auto it: abc) std::cout << it << " ";
    std::cout << std::endl;

    return 0;
}
