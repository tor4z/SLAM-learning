#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <cmath>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


const int boarder = 20;
const int width = 640;
const int height = 480;
const double fx = 481.2;
const double fy = -480.0;
const double cx = 319.5;
const double cy = 239.5;
const int ncc_window_size = 3;
const int ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1);
const double min_cov = 0.1;
const double max_cov = 10.0;


inline double getBilinearInterpolatedValue(
    const cv::Mat &image,
    const Eigen::Vector2d &pt
)
{
    uchar *d = &image.data[
        // row                            col
        int(pt(1, 0) * image.step + int(pt(0, 0)))
    ];
    double xx = pt(0, 0) - floor(pt(0, 0));
    double yy = pt(1, 0) - floor(pt(1, 0));
    return (
        (1 - xx) * (1 - yy) * static_cast<double>(d[0]) +
        xx * (1 - yy) * static_cast<double>(d[1]) +
        (1 - xx) * yy * static_cast<double>(d[image.step]) +
        xx * yy * static_cast<double>(d[image.step + 1])
    ) / 255.0;
}


inline Eigen::Vector3d px2cam(const Eigen::Vector2d &px)
{
    return Eigen::Vector3d(
        (px(0, 0) - cx) / fx,
        (px(1, 0) - cy) / fy,
        1
    );
}


inline Eigen::Vector2d cam2px(const Eigen::Vector3d &cam)
{
    return Eigen::Vector2d(
        cam(0, 0) * fx / cam(2, 0) + cx,
        cam(1, 0) * fy / cam(2, 0) + cy
    );
}


inline bool inside(const Eigen::Vector2d &pt)
{
    return pt(0, 0) >= boarder &&
           pt(1, 0) >= boarder &&
           pt(0, 0) + boarder < width &&
           pt(1, 0) + boarder < height;
}


bool readDatasetFiles(
    const std::string &path,
    std::vector<std::string> &color_image_files,
    std::vector<Sophus::SE3d> &poses,
    cv::Mat &ref_depth
)
{
    std::ifstream fin(
        path +
        "/first_200_frames_traj_over_table_input_sequence.txt"
    );

    if(!fin) return false;
    std::string image_name;

    while (!fin.eof())
    {
        // format: image_name rx ty tz qx qy qz qw
        fin >> image_name;
        double data[7];
        for (double &d: data) fin >> d;
        color_image_files.push_back(
            path + "/images/" + image_name
        );
        poses.push_back(
            Sophus::SE3d(
                Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
                Eigen::Vector3d(data[0], data[1], data[2])
            )
        );
        if(!fin.good())
        {
            std::cerr << "fin.good() is false." << std::endl;
            break;
        }
    }
    fin.close();

    fin.open(path + "/depthmaps/scene_000.depth");
    if(!fin) return false;
    ref_depth = cv::Mat(height, width, CV_64F);
    double depth = 0;
    for (size_t i = 0; i < height; i++)
    {
        for (size_t j = 0; j < width; j++)
        {
            fin >> depth;
            ref_depth.ptr<double>(i)[j] = depth / 100.0;
        }
    }

    fin.close();
    return true;
}


bool updateDepthFilter(
    const Eigen::Vector2d &pt_ref,
    const Eigen::Vector2d &pt_curr,
    const Sophus::SE3d &T_C_R,
    const Eigen::Vector2d &epipolar_direction,
    cv::Mat &depth,
    cv::Mat &depth_cov2
)
{
    Sophus::SE3d T_R_C = T_C_R.inverse();
    Eigen::Vector3d f_ref = px2cam(pt_ref);
    // std::cout << "f_ref: " << f_ref << std::endl;
    f_ref.normalize();
    
    Eigen::Vector3d f_curr = px2cam(pt_curr);
    // std::cout << "f_curr: " << f_curr << std::endl;
    f_curr.normalize();

    Eigen::Vector3d t = T_R_C.translation();
    Eigen::Vector3d f2 = T_R_C.so3() * f_curr;
    Eigen::Vector2d b = Eigen::Vector2d(t.dot(f_ref), t.dot(f2));
    Eigen::Matrix2d A;

    A(0, 0) = f_ref.dot(f_ref);
    A(0, 1) = -f_ref.dot(f2);
    A(1, 0) = -A(0, 1);
    A(1, 1) = -f2.dot(f2);

    Eigen::Vector2d ans = A.inverse() * b;
    Eigen::Vector3d xm = ans[0] * f_ref;
    Eigen::Vector3d xn = t + ans[1] * f2;
    Eigen::Vector3d p_esti = (xm + xn) / 2.0;
    double depth_estimation = p_esti.norm();

    Eigen::Vector3d p = f_ref * depth_estimation;
    Eigen::Vector3d a = p - t;

    double t_norm = t.norm();
    double a_norm = a.norm();
    double alpha = acos(f_ref.dot(t) / t_norm);
    double beta = acos(-a.dot(t) / (a_norm * t_norm));
    Eigen::Vector3d f_curr_prime = px2cam(pt_curr + epipolar_direction);    
    f_curr_prime.normalize();

    double beta_prime = acos(f_curr_prime.dot(-t) / t_norm);
    double gamma = M_PI - alpha - beta_prime;
    double p_prime = t_norm * sin(beta_prime) / sin(gamma);
    double d_cov = p_prime - depth_estimation;
    double d_cov2 = d_cov * d_cov;

    double mu = depth.ptr<double>(static_cast<int>(pt_ref(1, 0)))
        [static_cast<int>(pt_ref(0, 0))];
    double sigma2 = depth_cov2.ptr<double>(static_cast<int>(pt_ref(1, 0)))
        [static_cast<int>(pt_ref(0, 0))];
    
    double mu_fuse = (d_cov2 * mu + sigma2 * depth_estimation) / (sigma2 + d_cov2);
    double sigma_fuse2 = (sigma2 * d_cov2) / (sigma2 + d_cov2);

    depth.ptr<double>(static_cast<int>(pt_ref(1, 0)))
        [static_cast<int>(pt_ref(0, 0))] = mu_fuse;
    depth_cov2.ptr<double>(static_cast<int>(pt_ref(1, 0)))
        [static_cast<int>(pt_ref(0, 0))] = sigma_fuse2;

    return true;
}


double NCC(
    const cv::Mat &ref,
    const cv::Mat &curr,
    const Eigen::Vector2d &pt_ref,
    const Eigen::Vector2d &pt_curr
)
{
    double mean_ref = 0;
    double mean_curr = 0;
    std::vector<double> ref_values, curr_values;
    for (int i = -ncc_window_size; i < ncc_window_size; i++)
    {
        for (int j = -ncc_window_size; j < ncc_window_size; j++)
        {
            double ref_value = static_cast<double>(
                ref.ptr<uchar>(static_cast<int>(j + pt_ref(1, 0)))
                    [static_cast<int>(i + pt_ref(0, 0))]
            ) / 255.0;
            mean_ref += ref_value;
            double curr_value = getBilinearInterpolatedValue(
                curr, pt_curr + Eigen::Vector2d(j, i)
            );
            mean_curr += curr_value;

            ref_values.push_back(ref_value);
            curr_values.push_back(curr_value);
        }
    }

    mean_ref /= ncc_area;
    mean_curr /= ncc_area;
    
    double numerator = 0;
    double demoniator1 = 0;
    double demoniator2 = 0;
    for (size_t i = 0; i < ref_values.size(); i++)
    {
        double n = (ref_values[i] - mean_ref) * (curr_values[i] - mean_curr);
        numerator += n;
        demoniator1 += (ref_values[i] - mean_ref) * (ref_values[i] - mean_ref);
        demoniator2 += (curr_values[i] - mean_curr) * (curr_values[i] - mean_curr);
    }
    return numerator / sqrt(demoniator1 * demoniator2 + 1e-10);
    
}


bool epipolarSearch(
    const cv::Mat &ref,
    const cv::Mat &curr,
    const Sophus::SE3d &T_C_R,
    const Eigen::Vector2d &pt_ref,
    const double &depth_mu,
    const double &depth_cov,
    Eigen::Vector2d &pt_curr,
    Eigen::Vector2d &epipolar_direction
)
{
    Eigen::Vector3d f_ref = px2cam(pt_ref);
    f_ref.normalize();
    
    Eigen::Vector3d p_ref = f_ref * depth_mu;

    Eigen::Vector2d px_mean_curr = cam2px(T_C_R * p_ref);
    double d_min = depth_mu - 3 * depth_cov;
    double d_amx = depth_mu + 3 * depth_cov;
    if(d_min < 0.1) d_min = 0.1;

    Eigen::Vector2d px_min_curr = cam2px(T_C_R * (f_ref * d_min));
    Eigen::Vector2d px_max_curr = cam2px(T_C_R * (f_ref * d_amx));

    Eigen::Vector2d  epipolar_line = px_max_curr - px_min_curr;
    epipolar_direction = epipolar_line;
    epipolar_direction.normalize();
    
    double half_length = 0.5 * epipolar_line.norm();
    if(half_length > 1000) half_length = 100;

    double best_ncc = -1.0;
    Eigen::Vector2d best_px_curr;
    double ncc;

    for (double i = -half_length; i < half_length; i+=0.7) // += sqrt(2)
    {
        Eigen::Vector2d px_curr = px_mean_curr + i * epipolar_direction;
        if(!inside(px_curr)) continue;

        ncc = NCC(ref, curr, pt_ref, px_curr);
        if(ncc > best_ncc)
        {
            // std::cout << "best ncc: " << ncc << std::endl;
            best_ncc = ncc;
            best_px_curr = px_curr;
        }
    }

    // std::cout << "ncc=" << ncc << ", best_ncc=" << best_ncc << std::endl;
    if(best_ncc < 0.85f)
        return false;
    pt_curr = best_px_curr;
    return true;
}


void showEpipolarMatch(
    const cv::Mat &ref,
    const cv::Mat &curr,
    const Eigen::Vector2d &px_ref,
    const Eigen::Vector2d &px_curr
)
{
    cv::Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, cv::COLOR_GRAY2BGR);
    cv::cvtColor(curr, curr_show, cv::COLOR_GRAY2BGR);

    cv::circle(
        ref_show,
        cv::Point2f(px_ref(0, 0), px_ref(1, 0)),
        5,
        cv::Scalar(0, 0, 255),
        2
    );
    cv::circle(
        curr_show,
        cv::Point2f(px_curr(0, 0), px_curr(1, 0)),
        5,
        cv::Scalar(0, 0, 255),
        2
    );

    cv::imshow("ref", ref_show);
    cv::imshow("curr", curr_show);
    cv::waitKey(1);
}


void update(
    const cv::Mat &ref,
    const cv::Mat &curr,
    const Sophus::SE3d &T_C_R,
    cv::Mat &depth,
    cv::Mat &depth_cov2
)
{
    for (size_t i = boarder; i < width - boarder; i++)
    {
        for (size_t j = boarder; j < height - boarder; j++)
        {
            if(depth_cov2.ptr<double>(j)[i] < min_cov ||
               depth_cov2.ptr<double>(j)[i] > max_cov)
               continue;
            
            Eigen::Vector2d pt_curr;
            Eigen::Vector2d epipolar_direction;

            bool ret = epipolarSearch(
                ref,
                curr,
                T_C_R,
                Eigen::Vector2d(i, j),
                depth.ptr<double>(j)[i],
                sqrt(depth_cov2.ptr<double>(j)[i]),
                pt_curr,
                epipolar_direction
            );

            if(!ret)
            {
                // std::cout << "epipolar search failed." << std::endl;
                continue;
            }

            // showEpipolarMatch(
            //     ref,
            //     curr,
            //     Eigen::Vector2d(i, j),
            //     pt_curr
            // );

            updateDepthFilter(
                Eigen::Vector2d(i, j),
                pt_curr,
                T_C_R,
                epipolar_direction,
                depth,
                depth_cov2
            );
        }
    }
}


void plotDepth(
    const cv::Mat &depth_truth,
    const cv::Mat &depth_estimate
)
{
    cv::imshow("depth truth", depth_truth);
    cv::imshow("depth estimate", depth_estimate);
    cv::imshow("depth error", depth_truth - depth_estimate);
    cv::waitKey(1);
}


void evaluatedDepth(
    const cv::Mat &depth_truth,
    const cv::Mat &depth_estimate
)
{
    float total_depth_error = 0;
    float total_depth_error_sq = 0;
    int cnt_depth_data = 0;
    for (int i = boarder; i < depth_truth.rows - boarder; i++)
    {
        for (int j = boarder; j < depth_truth.cols - boarder; j++)
        {
            float error = depth_truth.ptr<float>(i)[j] -
                           depth_estimate.ptr<float>(i)[j];
            total_depth_error += error;
            total_depth_error_sq += error * error;
            ++cnt_depth_data;
            // std::cout << "total: " << total_depth_error << std::endl;
            // if (std::isnan(total_depth_error))
            //     break;
        }
    }

    float ave_depth_error = total_depth_error / cnt_depth_data;
    float ave_depth_error_sq = total_depth_error_sq / cnt_depth_data;
    std::cout << "average squared error: " << ave_depth_error_sq
        << ", average error: " << ave_depth_error
        << ", cnt: " << cnt_depth_data
        << ", total error: " << total_depth_error
        << ", total square error: " << total_depth_error_sq
        << std::endl;
}



void showEpipolarLine(
    const cv::Mat &ref,
    const cv::Mat &curr,
    const Eigen::Vector2d &px_ref,
    const Eigen::Vector2d &px_min_curr,
    const Eigen::Vector2d &px_max_curr
)
{
    cv::Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, cv::COLOR_GRAY2BGR);
    cv::cvtColor(curr, curr_show, cv::COLOR_GRAY2BGR);

    cv::circle(
        ref_show,
        cv::Point2f(px_ref(0, 0), px_ref(1, 0)),
        5,
        cv::Scalar(0, 255, 0),
        2
    );
    cv::circle(
        curr_show,
        cv::Point2f(px_min_curr(0, 0), px_min_curr(1, 0)),
        5,
        cv::Scalar(0, 255, 0),
        2
    );
    cv::circle(
        curr_show,
        cv::Point2f(px_max_curr(0, 0), px_max_curr(1, 0)),
        5,
        cv::Scalar(0, 255, 0),
        2
    );
    cv::line(
        curr_show,
        cv::Point2f(px_min_curr(0, 0), px_min_curr(1, 0)),
        cv::Point2f(px_max_curr(0, 0), px_max_curr(1, 0)),
        cv::Scalar(0, 255, 0),
        1
    );

    cv::imshow("ref", ref_show);
    cv::imshow("curr", curr_show);
    cv::waitKey(1);
}


const std::string remode_test_data = "/home/tor/Data/test_data";


int main(int argc, char** argv)
{
    std::vector<std::string> color_image_files;
    std::vector<Sophus::SE3d> poses_TWC;
    cv::Mat ref_depth;

    bool ret = readDatasetFiles(
        remode_test_data,
        color_image_files,
        poses_TWC,
        ref_depth
    );

    if(!ret)
    {
        std::cerr << "Failed to read files" << std::endl;
        return 1;
    }
    std::cout << "read total " << color_image_files.size()
        << " files" << std::endl;
    
    cv::Mat ref = cv::imread(
        color_image_files[0],
        cv::IMREAD_GRAYSCALE
    );
    Sophus::SE3d pose_ref_TWC = poses_TWC[0];
    double init_depth = 3.0;
    double init_cov2 = 3.0;
    cv::Mat depth(height, width, CV_64F, init_depth);
    cv::Mat depth_cov2(height, width, CV_64F, init_cov2);

    for (size_t i = 1; i < color_image_files.size(); i++)
    {
        std::cout << "== loop " << i << "/"
            << color_image_files.size() << " ==="
            << std::endl;

        cv::Mat curr = cv::imread(
            color_image_files[i],
            cv::IMREAD_GRAYSCALE
        );
        if(curr.empty()) continue;
        Sophus::SE3d pose_curr_TWC = poses_TWC[i];
        Sophus::SE3d pose_T_C_R = pose_curr_TWC.inverse() * pose_ref_TWC;
        update(ref, curr, pose_T_C_R, depth, depth_cov2);
        evaluatedDepth(ref_depth, depth);
        // plotDepth(ref_depth, depth);
        // cv::imshow("image", curr);
        // cv::waitKey(1);
    }

    std::cout << "estimation return, saving depth map ..." << std::endl;
    cv::imwrite("depth.png", depth);
    std::cout << "done." << std::endl;

    return 0;
}
