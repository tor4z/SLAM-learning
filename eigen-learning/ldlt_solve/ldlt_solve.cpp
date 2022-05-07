#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>


int main(int argc, char** argv)
{
    Eigen::Matrix3d H = Eigen::Matrix3d::Random();
    Eigen::Vector3d B = Eigen::Vector3d::Random();
    Eigen::Vector3d X = H.ldlt().solve(B);
    std::cout << X.transpose() << std::endl;
    return 0;
}
