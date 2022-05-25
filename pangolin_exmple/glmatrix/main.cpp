#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <Eigen/Core>


int main()
{
    Sophus::SE3d Twc = Sophus::SE3d(
        Sophus::SO3d(),
        Eigen::Matrix<double, 3, 1>::Zero()
    );
    pangolin::OpenGlMatrix(Twc.matrix());
    return 0;
}
