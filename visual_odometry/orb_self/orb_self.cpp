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


void read_ORB(const std::string &path)
{

}


// brute force matching
void BF_match(
    const std::vector<DescType> &desc1,
    const std::vector<DescType> &desc2,
    std::vector<cv::DMatch> &matches
)
{
    for (size_t i = 0; i < desc1.size(); i++)
    {
        const int d_max = 40;

        if (desc1[i].empty()) continue;
        cv::DMatch m(i, 0, 256);

        for (size_t j = 0; j < desc2.size(); j++)
        {
            if (desc2[j].empty()) continue;
            int distance = 0;

            for (size_t k = 0; k < 8; k++)
                distance += _mm_popcnt_u32(desc1[i][k] ^ desc2[j][k]);
            
            if(distance < d_max && distance < m.distance)
            {
                m.distance = distance;
                m.trainIdx = j;
            }
            if(distance < d_max)
                matches.push_back(m);
        }
    }
}


// compute discriptor
void compute_orb(
    const cv::Mat& image,
    std::vector<cv::KeyPoint>& key_point,
    std::vector<DescType>& descriptors)
{
    const int half_patch_size = 8;
    const int half_boundary = 16;
    int bad_points = 0;

    for (auto &kp: key_point)
    {
        if(kp.pt.x < half_boundary || kp.pt.y < half_boundary ||
           kp.pt.x + half_boundary >= image.cols ||
           kp.pt.y + half_boundary >= image.rows)
        {
            ++bad_points;
            descriptors.push_back({});
            continue;
        }

        float m01 = 0, m10 = 0;
        for (int x = -half_patch_size; x < half_patch_size; x++)
        {
            for (int y = -half_patch_size; y < half_patch_size; y++)
            {
                uchar pixel = image.at<uchar>(kp.pt.y + y, kp.pt.x + x);
                m10 += x + pixel;
                m01 += y + pixel;
            }
        }

        float m_sqrt = sqrt(m10 * m10 + m01 * m01) + 1e-18;
        float sin_theta = m01 / m_sqrt;
        float cos_theta = m10 / m_sqrt;

        DescType desc(8, 0);
        for (size_t i = 0; i < 8; i++)
        {
            uint32_t d = 0;
            for (size_t k = 0; k < 32; k++)
            {
                int idx_pq = i * 32 + k;
                // cv::Point2f 
            }
            
        }
        
    }
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
