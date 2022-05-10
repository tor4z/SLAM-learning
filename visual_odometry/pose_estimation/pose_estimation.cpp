#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>


cv::Point2d pixel2cam(
    const cv::Point2d &p,
    const cv::Mat &k
)
{
    return cv::Point2d(
        (p.x - k.at<double>(0, 2)) / k.at<double>(0, 0),
        (p.y - k.at<double>(1, 2)) / k.at<double>(1, 1)
    );
}


void find_feature_matches(
    const cv::Mat &image1,
    const cv::Mat &image2,
    std::vector<cv::KeyPoint> &keypoints1,
    std::vector<cv::KeyPoint> &keypoints2,
    std::vector<cv::DMatch> &matches
)
{
    cv::Mat descriptors1, descriptors2;
    std::vector<cv::DMatch> all_matches;
    double min_dist = 10000.0, max_dist = 0.0;
    double dist;

    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor_extractor
        = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> descriptor_matcher
        = cv::DescriptorMatcher::create("BruteForce-Hamming");

    detector->detect(image1, keypoints1);
    detector->detect(image2, keypoints2);

    descriptor_extractor->compute(image1, keypoints1, descriptors1);
    descriptor_extractor->compute(image2, keypoints2, descriptors2);

    descriptor_matcher->match(descriptors1, descriptors2, all_matches);

    // find max/min dist
    for (size_t i = 0; i < descriptors1.rows; i++)
    {
        dist = all_matches[i].distance;
        max_dist = std::max(dist, max_dist);
        min_dist = std::min(dist, min_dist);
    }
    
    // filter matches
    for (size_t i = 0; i < descriptors1.rows; i++)
    {
        if(all_matches[i].distance <= std::max(2 * min_dist, 30.0))
            matches.push_back(all_matches[i]);
    }
}


void pose_estimation_2d2d(
    std::vector<cv::KeyPoint> &keypoints1,
    std::vector<cv::KeyPoint> &keypoints2,
    std::vector<cv::DMatch> &matches,
    cv::Mat &R,
    cv::Mat &t
)
{
    cv::Mat k = (
        cv::Mat_<double>(3, 3) <<
        520.9, 0,     325.1,
        0,     521.0, 249.7,
        0,     0,     1
    );

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

    for (size_t i = 0; i < matches.size(); i++)
    {
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    cv::Mat fundamental_matrix;
    fundamental_matrix = cv::findFundamentalMat(
        points1, points2, cv::FM_8POINT
    );
    std::cout << "fundamental matrix:\n"  << fundamental_matrix << std::endl;

    cv::Point2d principal_point(325.1, 249.7);
    double focal_length = 521;
    cv::Mat essential_matrix;
    essential_matrix = cv::findEssentialMat(
        points1, points2,
        focal_length,
        principal_point
    );
    std::cout << "essential matrix:\n" << essential_matrix << std::endl;

    cv::Mat homography_matrix;
    homography_matrix = cv::findHomography(
        points1, points2,
        cv::RANSAC, 3
    );
    std::cout << "homography matrix:\n" << homography_matrix << std::endl;

    cv::recoverPose(
        essential_matrix,
        points1,
        points2,
        R,
        t,
        focal_length,
        principal_point
    );
    std::cout << "R is:\n" << R << std::endl;
    std::cout << "t is:\n" << t << std::endl;
}



const std::string image1_path = "../asset/1.png";
const std::string image2_path = "../asset/2.png";


int main(int argc, char** argv)
{
    cv::Mat image1 = cv::imread(
        image1_path,
        cv::IMREAD_COLOR
    );
    cv::Mat image2 = cv::imread(
        image2_path,
        cv::IMREAD_COLOR
    );
    assert(!image1.empty() && ! image2.empty());

    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    std::vector<cv::DMatch> matches;

    find_feature_matches(
        image1, image2,
        keypoints1, keypoints2,
        matches
    );
    std::cout << "found " << matches.size() << " pair match points." << std::endl;

    cv::Mat R, t;
    pose_estimation_2d2d(keypoints1, keypoints2, matches, R, t);

    cv::Mat t_x = (
        cv::Mat_<double>(3, 3) <<
            0,                  -t.at<double>(2, 0),  t.at<double>(1, 0),
            t.at<double>(2, 0),  0,                  -t.at<double>(0, 0),
           -t.at<double>(1, 0),  t.at<double>(0, 0),  0
    );
    std::cout << "t^R=\n" << t_x * R << std::endl;

    cv::Mat k = (
        cv::Mat_<double>(3, 3) <<
            520.9, 0,     325.1,
            0,     521.0, 249.7,
            0,     0,     1
    );
    for (auto &m: matches)
    {
        cv::Point2d pt1 = pixel2cam(keypoints1[m.queryIdx].pt, k);
        cv::Point2d pt2 = pixel2cam(keypoints2[m.trainIdx].pt, k);
        cv::Mat y1 = (
            cv::Mat_<double>(3, 1) <<
                pt1.x, pt1.y, 1
        );
        cv::Mat y2 = (
            cv::Mat_<double>(3, 1) <<
                pt2.x, pt2.y, 1
        );

        cv::Mat d = y2.t() * t_x * R * y1;
        std::cout << "epipolar constraint: " << d << std::endl;
    }

    return 0;
}
