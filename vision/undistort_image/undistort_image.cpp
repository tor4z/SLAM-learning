#include <iostream>
#include <string>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>


const std::string image_path = "../asset/distroted.png";


int main(int argc, char** argv)
{
    // distrot param
    double k1, k2, p1, p2;
    double fx, fy, cx, cy;

    cv::Mat image = cv::imread(image_path);
    int rows, cols;
    cv::Mat undistrot_image = cv::Mat(rows, cols, CV_8UC1);

    for (size_t v = 0; v < rows; v++)
    {
        for (size_t u = 0; u < cols; u++)
        {
            // map from (v, u) to (v_distroted, u_dsitroted)
            double x = (u - cx) / fx;
            double y = (v - cy) / fy;
            double r = std::sqrt(x * x + y * y);
            double x_distroted = x * (1 + k1 * r * r + k2 * std::pow(r, 4))
                                 + 2 * p1 * x * y
                                 + p2 * (r * r + 2 * x * x);
            double y_distroted = y * (1 + k1 * r * r + k2 * std::pow(r, 4))
                                 + 2 * p2 * x * y
                                 + p1 * (r * r + 2 * y * y);
            double u_distroted = fx * x_distroted + cx;
            double v_dsitroted = fy * y_distroted + cy;

            if(u_distroted >= 0 && v_dsitroted >= 0 &&
               u_distroted < cols && v_dsitroted < rows)
            {
                undistrot_image.at<uchar>(v, u) = image.at<uchar>(
                    static_cast<int>(v_dsitroted),
                    static_cast<int>(u_distroted)
                );
            } else {
                undistrot_image.at<uchar>(v, u) = 0;
            }
        }
    }

    cv::imshow("Distroted", image);
    cv::imshow("Undistroted", undistrot_image);
    cv::waitKey(0);

    return 0;
}
