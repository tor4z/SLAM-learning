#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>


int main(int argc, char** argv)
{
    double ar = 1.0, br = 2.0, cr = 1.0;
    double ae = 2.0, be = -1.0, ce = 5.0;
    int N = 100;
    // noise
    double w_sigma = 1.0;
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng;

    std::vector<double> x_data, y_data;
    // generate data
    for (size_t i = 0; i < N; i++)
    {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(
            exp(ar * x * x + br * x + cr)
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
        Eigen::Vector3d b = Eigen::Vector3d::Zero();
        cost = 0;

        for (size_t i = 0; i < N; i++)
        {
            double xi = x_data[i];
            double yi = y_data[i];
            double ye = exp(ae * xi * xi + be * xi + ce);
            double error = yi - ye;
            // jacobi
            Eigen::Vector3d J;
            // d_error/d_a
            J[0] = -xi * xi * ye;
            // d_error/d_b
            J[1] = -xi * ye;
            // d_error/d_c
            J[2] = -ye;

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

        if (iter > 0 && cost >= last_cost)
        {
            std::cout << "cost: " << cost
                << ",laste cost: " << last_cost
                << std::endl;
            std::cout << "Break." << std::endl;
            break;
        }

        last_cost = cost;
        ae += dx[0];
        be += dx[1];
        ce += dx[2];

        std::cout << "total cost: " << cost << std::endl;
        std::cout << "estimated param: "
            << "ae: " << ae
            << ",be: " << be
            << ",ce: " << ce
            << std::endl;
    }

    std::chrono::steady_clock::time_point t_end
        = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used
        = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start);
    std::cout << "time used: " << time_used.count() << "s." << std::endl;
    std::cout << "estimated result: "
        << "ae: " << ae << ",be: " << be << ",ce: " << ce
        << std::endl;

    return 0;
}
