#include <iostream>
#include <ctime>
#include <Eigen/Core>
#include <Eigen/Dense>

#define MATRIX_SIZE 60


int main(int argc, char** argv)
{
    Eigen::Matrix<float, 2, 3> matrix_23;
    Eigen::Vector3d v_3d;
    Eigen::Matrix<float, 3, 1> vd_3d;

    Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_dynamic;
    Eigen::MatrixXd matrix_x;

    matrix_23 << 1, 2, 3, 4, 5, 6;

    std::cout << "matrix_23 from 1 to 6:\n" << matrix_23 << std::endl;

    std::cout << "print matrix_23:" << std::endl;
    for (size_t i = 0; i < 2; i++)
    {
        for (size_t j = 0; j < 3; j++)
            std::cout << matrix_23(i, j) << '\t';
        std::cout << std::endl;
    }
    
    v_3d << 3, 2, 1;
    vd_3d << 4, 5, 6;

    Eigen::Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;


    matrix_33 = Eigen::Matrix3d::Random();
    std::cout << "random matrix:\n" << matrix_33 << std::endl;
    std::cout << "random matrix transpose:\n" << matrix_33.transpose() << std::endl;
    std::cout << "sum of matrix: " << matrix_33.sum() << std::endl;
    std::cout << "trace:" << matrix_33.trace() << std::endl;
    std::cout << "time 10: \n" << 10 * matrix_33 << std::endl;
    std::cout << "inverse: \n" << matrix_33.inverse() << std::endl;
    std::cout << "det: " << matrix_33.determinant() << std::endl;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);
    std::cout << "eigen values:\n" << eigen_solver.eigenvalues() << std::endl;
    std::cout << "eigen vectors \n" << eigen_solver.eigenvectors() << std::endl;

    Eigen::Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN
        = Eigen::MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    
    // make sure semi-positive-definition
    matrix_NN = matrix_NN.transpose() * matrix_NN;
    Eigen::Matrix<double, MATRIX_SIZE, 1> v_Nd = Eigen::MatrixXd::Random(MATRIX_SIZE, 1);

    std::clock_t time_start = std::clock();
    
    Eigen::Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
    std::cout << "Time of normal invers is: "
        << 1000 * (std::clock() - time_start) / static_cast<double>(CLOCKS_PER_SEC) << "ms"
        << std::endl;
    std::cout << "x = " << x.transpose() << std::endl;

    time_start = std::clock();
    x = matrix_NN.ldlt().solve(v_Nd);
    std::cout << "Time of ldlt decomposition is "
        << 1000 * (std::clock() - time_start) / static_cast<double>(CLOCKS_PER_SEC)
        << "ms" << std::endl;
    std::cout << "x = " << x.transpose() << std::endl;

    return 0;
}
