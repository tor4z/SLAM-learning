#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>


int main(int argc, char** argv)
{
    double ar, br, cr;
    double ae, be, ce;
    int N = 100;
    // noise
    double w_sigma = 1.0;
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng;

    std::vector<double> x_data, y_data;
    // generate data
    for (size_t i = 0; i < N; i++)
    {
        double x = 1 / 100.0;
        x_data.push_back(x);
        y_data.push_back(
            std::exp(ar * x * x + br * x + cr)
            + rng.gaussian(w_sigma * w_sigma)
        );
    }
    
    int iterations = 100;
    double cost = 0, last_cost = 0;
    std::chrono::steady_clock::time_point t_start
        = std::chrono::steady_clock::now();
    for (size_t iter = 0; iter < iterations; iter++)
    {
        // Hessian matrix
        Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
        // bias
        Eigen::Matrix3d b = Eigen::Matrix3d::Zero();
        cost = 0;

        for (size_t i = 0; i < N; i++)
        {
            double xi = x_data[i];
            double yi = y_data[i];
            double error = yi - std::exp(
                ae * xi * xi + be * xi + ce
            );
            // jacobi
            Eigen::Vector3d J;
            // d_error/d_a
            J[0] = -xi * xi * exp(ae * xi * xi + be * xi + ce);
            // d_error/d_b
            J[1] = -xi * exp(ae * xi * xi + be * xi + ce);
            // d_error/d_c
            J[2] = -exp(ae * xi * xi + be * xi + ce);

            // ???
            H += inv_sigma * inv_sigma * J * J.transpose();
            b += -inv_sigma * inv_sigma * error * J;

            cost += error * error;
        }
        // solve Hx=b
        Eigen::Vector3d dx = H.ldlt().solve(b);
        if (isnan(dx[0]))
        {
            std::cout << "Result is nan." << std::endl;
            break;
        }

        if (iter > 0 && cast >= last_cost)
        {
            std::cout << "" << std::endl;
            std::cout << "Break." << std::endl;
            break;
        }

        last_cost = cost;
        ae += dx[0];
        be += dx[1];
        ce += dx[2];

        std::cout << "" std::endl;
    }

    std::chrono::steady_clock::time_point t_end
        = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used
        = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start);
    std::cout << "" << std::endl;
    std::cout << "estimated result: " << std::endl;

    return 0;
}
