#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <nmmintrin.h>
#include <opencv2/opencv.hpp>

// descriptor type
typedef std::vector<uint32_t> DescType;


const std::string image1_file = "../asset/1.png";
const std::string image2_file = "../asset/2.png";
const std::string orb_pattern_file = "../asset/orb_pattern.txt";


// compute discriptor
void compute_orb(
    const cv::Mat& image,
    std::vector<cv::KeyPoint>& key_point,
    std::vector<DescType>& descriptors)
{

}


int main(int argc, char** argv)
{
    cv::Mat image1 = cv::imread(image1_file,  cv::IMREAD_GRAYSCALE);
    cv::Mat image2 = cv::imread(image2_file,  cv::IMREAD_GRAYSCALE);

    if(image1.empty() || image2.empty())
    {
        std::cerr << "" << std::endl;
        return 1;
    }

    std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();
    std::vector<cv::KeyPoint> key_point1;
    cv::FAST(image1, key_point1, 40);
    std::vector<DescType> descriptor1;


    return 0;
}
