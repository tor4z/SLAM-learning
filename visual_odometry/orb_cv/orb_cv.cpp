#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>


int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cout << "Invalid argument." << std::endl;
        return 1;
    }

    cv::Mat image1 = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat image2 = cv::imread(argv[2], cv::IMREAD_COLOR);
    assert(image1.data != nullptr && image2.data != nullptr);

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    cv::Ptr<cv::FeatureDetector> detector
        = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor
        = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher
        = cv::DescriptorMatcher::create("BruteForce_Hamming");

    std::chrono::steady_clock::time_point t_start
        = std::chrono::steady_clock::now();
    detector->detect(image1, keypoints1);
    detector->detect(image2, keypoints2);

    descriptor->compute(image1, keypoints1, descriptors1);
    descriptor->compute(image2, keypoints2, descriptors2);
    std::chrono::steady_clock::time_point t_end
        = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used
        = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start);
    std::cout << "Extract ORB cost " << time_used.count() << "s." << std::endl;

    cv::Mat image1Keypoints;
    cv::drawKeypoints(
        image1, keypoints1,
        image1Keypoints, cv::Scalar(255, 255, 255),
        cv::DrawMatchesFlags::DEFAULT
    );
    cv::namedWindow("image1Keypoints", cv::WINDOW_AUTOSIZE);
    cv::imshow("image1Keypoints", image1Keypoints);

    std::vector<cv::DMatch> matches;
    t_start = std::chrono::steady_clock::now();
    matcher->match(descriptors1, descriptors2, matches);
    t_end = std::chrono::steady_clock::now();
    time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start);
    std::cout << "Matches ORB cost " << time_used.count() << "s." << std::endl;

    auto min_max = std::minmax_element(
        matches.begin(), matches.end(),
        [](const cv::DMatch &m1, const cv::DMatch& m2) {
            return m1.distance < m2.distance;
        }
    );

    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    std::cout << "Min dist " << min_dist << std::endl;
    std::cout << "Max dist " << min_dist << std::endl;

    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < descriptors1.rows; i++)
    {
        if (matches[i].distance <= std::max(2 * min_dist, 30.0))
        {
            good_matches.push_back(matches[i]);
        }
    }
    
    cv::Mat imageMatch;
    cv::Mat imageGoodMatch;
    cv::drawMatches(
        image1, keypoints1,
        image2, keypoints2,
        matches, imageMatch
    );
    cv::drawMatches(
        image1, keypoints1,
        image2, keypoints2,
        good_matches, imageGoodMatch
    );

    cv::namedWindow("imageMatch", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("imageGoodMatch", cv::WINDOW_AUTOSIZE);
    cv::imshow("imageMatch", imageMatch);
    cv::imshow("imageGoodMatch", imageGoodMatch);
    cv::waitKey(0);

    return 0;
}
