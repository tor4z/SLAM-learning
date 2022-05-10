#include <iostream>
#include <Eigen/Core>


int main(int argc, char** argv)
{
    int array[10];
    for (size_t i = 0; i < 10; i++)
        array[i] = i;
    
    std::cout << Eigen::Map<Eigen::Matrix3i>(array) << std::endl;
    Eigen::Map<Eigen::Matrix3i>(array) *= 2;
    std::cout << Eigen::Map<Eigen::Matrix3i>(array) << std::endl;
    
    for (auto &it: array)
    {
        std::cout << it << " ";
    }
    std::cout << std::endl;

    return 0;
}
