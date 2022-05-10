#include <iostream>
#include <algorithm>
#include <string>
#include <opencv2/opencv.hpp>


cv::Point2f pixel2cam(
    const cv::Point2d &p,
    const cv::Mat &K
)
{
    return cv::Point2f(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
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
    double max_dist = 0.0, min_dist = 10000.0;
    double dist;

    cv::Ptr<cv::DescriptorExtractor> descriptor_extractor = cv::ORB::create();
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> descriptor_matcher
        = cv::DescriptorMatcher::create("BruteForce-Hamming");
    
    detector->detect(image1, keypoints1);
    detector->detect(image2, keypoints2);

    descriptor_extractor->compute(image1, keypoints1, descriptors1);
    descriptor_extractor->compute(image2, keypoints2, descriptors2);

    descriptor_matcher->match(descriptors1, descriptors2, all_matches);

    for (size_t i = 0; i < descriptors1.rows; i++)
    {
        dist = all_matches[i].distance;
        max_dist = std::max(max_dist, dist);
        min_dist = std::min(min_dist, dist);
    }
    
    std::cout << "max dist: " << max_dist << std::endl;
    std::cout << "min dist: " << min_dist << std::endl;

    for (size_t i = 0; i < descriptors1.rows; i++)
    {
        if(all_matches[i].distance <= std::max(2 * min_dist, 30.0))
            matches.push_back(all_matches[i]);
    }
}

void pose_estimation_2d2d(
    const std::vector<cv::KeyPoint> &keypoints1,
    const std::vector<cv::KeyPoint> &keypoints2,
    const std::vector<cv::DMatch> &matches,
    cv::Mat &R, cv::Mat &t
)
{
    cv::Mat K = (
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
    
    cv::Point2d principal_point(325.1, 249.7);
    int focal_length = 512;
    cv::Mat essential_matrix;
    essential_matrix = cv::findEssentialMat(
        points1,
        points2,
        focal_length,
        principal_point
    );

    cv::recoverPose(
        essential_matrix,
        points1,
        points2,
        R, t,
        focal_length,
        principal_point
    );
}


void triangulation(
    const std::vector<cv::KeyPoint> &keypoints1,
    const std::vector<cv::KeyPoint> &keypoints2,
    const std::vector<cv::DMatch> &matches,
    const cv::Mat &R,
    const cv::Mat &t,
    std::vector<cv::Point3d> &points
)
{
    cv::Mat T1 = (cv::Mat_<float>(3, 4) << 
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0
    );
    cv::Mat T2 = (cv::Mat_<float>(3, 4) <<
        R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
    );
    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        520.9, 0,     325.1,
        0,     521.0, 249.7,
        0,     0,     1
    );
    std::vector<cv::Point2f> pts1, pts2;

    for (auto &m: matches)
    {
        pts1.push_back(pixel2cam(keypoints1[m.queryIdx].pt, K));
        pts2.push_back(pixel2cam(keypoints2[m.trainIdx].pt, K));
    }

    cv::Mat pts4d;
    cv::triangulatePoints(T1, T2, pts1, pts2, pts4d);

    for (size_t i = 0; i < pts4d.cols; i++)
    {
        cv::Mat x = pts4d.col(i);
        x /= x.at<float>(3, 0);
        cv::Point3d p(
            x.at<float>(0, 0),
            x.at<float>(1, 0),
            x.at<float>(2, 0)
        );
        points.push_back(p);
    }
}


inline cv::Scalar get_color(float depth)
{
    float up_th = 15, low_th = 5;
    float th_range = up_th - low_th;

    depth = std::min(depth, up_th);
    depth = std::max(depth, low_th);
    return cv::Scalar(
        255 * depth / th_range,
        0,
        255 * (1 - depth / th_range)
    );
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

    assert(!image1.empty() && !image2.empty());

    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    std::vector<cv::DMatch> matches;

    find_feature_matches(image1, image2, keypoints1, keypoints2, matches);
    std::cout << "Found " << matches.size() << " matched pairs." << std::endl;

    cv::Mat R, t;
    std::vector<cv::Point3d> points;
    pose_estimation_2d2d(keypoints1, keypoints2, matches, R, t);
    triangulation(keypoints1, keypoints2, matches, R, t, points);

    cv::Mat K = (cv::Mat_<double>(3, 3) <<
        520.9, 0,     325.1,
        0,     521.0, 249.7,
        0,     0,     1
    );
    cv::Mat image1_plot = image1.clone();
    cv::Mat image2_plot = image2.clone();

    for (size_t i = 0; i < matches.size(); i++)
    {
        float depth1 = points[i].z;
        std::cout << "depth: " << depth1 << std::endl;
        cv::Point2f pt1_cam = pixel2cam(keypoints1[matches[i].queryIdx].pt, K);
        cv::circle(
            image1_plot,
            keypoints1[matches[i].queryIdx].pt,
            2,
            get_color(depth1),
            2
        );

        cv::Mat pt_trans = R * (cv::Mat_<double>(3, 1) << 
            points[i].x, points[i].y, points[i].z
        ) + t;
        float depth2 = pt_trans.at<double>(2, 0);
        cv::circle(
            image2_plot,
            keypoints2[matches[i].trainIdx].pt,
            2,
            get_color(depth2),
            2
        );
    }
    
    cv::imshow("image1", image1_plot);
    cv::imshow("image2", image2_plot);
    cv::waitKey(0);

    return 0;
}
