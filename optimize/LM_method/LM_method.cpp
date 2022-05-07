#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>


double f(const Eigen::Vector3d& param, double x)
{
    return exp(
        param[0] * x * x +
        param[1] * x +
        param[2]
    );
}


Eigen::Vector3d jacobi(
    const Eigen::Vector3d &param,
    double x
)
{
    Eigen::Vector3d jacobi_matrix;
    double ye = f(param, x);
    jacobi_matrix[0] = -x * x * ye;
    jacobi_matrix[1] = -x * ye;
    jacobi_matrix[2] = -ye;
    return jacobi_matrix;
}


void LMMethod(
    const std::vector<double> &x_data,
    const std::vector<double> &y_data,
    Eigen::Vector3d &param,
    size_t iterations,
    double lambda
)
{
    double cost = 0, last_cost = 0;
    double xi, yi, ye, error;
    double rho = 0;
    Eigen::Matrix3d H, I;
    Eigen::Vector3d B, J;
    Eigen::Vector3d dx;
    I = Eigen::Matrix3d::Identity();

    for (size_t iter = 0; iter < iterations; iter++)
    {
        H = Eigen::Matrix3d::Zero();
        B = Eigen::Vector3d::Zero();

        for (size_t i = 0; i < x_data.size(); i++)
        {
            yi = y_data[i];
            xi = x_data[i];
            ye = f(param, xi);
            error = yi - ye;
            J = jacobi(param, xi);

            H += J * J.transpose();
            B += -error * J;
            cost = error * error;
        }
        H = H + lambda * I;
        dx = H.ldlt().solve(B);
        param = param + dx;

        if(iter > 0 && cost >= last_cost)
        {
            std::cout << "cost=" << cost
                << ", last_cost=" << last_cost
                << std::endl;
            std::cout << "Break" << std::endl;
            break;
        }

        last_cost = cost;

        std::cout << "current cost: " << cost << std::endl;
        std::cout << "param: " << param.transpose() << std::endl;
        std::cout << "====" << std::endl;
    }

    std::cout << "------------" << std::endl;
    std::cout << "final cost: " << cost << std::endl;
    std::cout << "final param: " << param.transpose() << std::endl;
    std::cout << "------------" << std::endl;
}


int main(int argc, char** argv)
{
    Eigen::Vector3d param_r;
    param_r << 1.0, 2.0, 1.0;
    Eigen::Vector3d param_e;
    param_e << 2.0, -1.0, 5.0;
    double w_sigma = 1.0;
    cv::RNG rng;
    int N = 100;

    std::vector<double> x_data, y_data;
    double x;
    for (size_t i = 0; i < N; i++)
    {
        x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(
            f(param_r, x) +
            rng.gaussian(w_sigma * w_sigma)
        );
    }

    LMMethod(x_data, y_data, param_e, 20, 0.01);
    return 0;
}
