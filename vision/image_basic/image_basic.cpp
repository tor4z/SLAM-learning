#include <iostream>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>


int main(int argc, char**  argv)
{
    if(argc != 2)
    {
        std::cerr << "" << std::endl;
        return 1;
    }
    cv::Mat image;
    image = cv::imread(argv[1]);
    if(image.empty())
    {
        std::cerr << "" << std::endl;
        return 1;
    }
    // print image information
    std::cout << "" << std::endl;
    cv::imshow("Image", image);
    cv::waitKey(0);

    if(image.type() != CV_8UC1 && image.type() != CV_8UC3)
    {
        std::cerr << "" << std::endl;
        return 1;
    }

    std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();
    size_t nc = image.cols * image.channels();
    for (size_t i = 0; i < image.rows; i++)
    {
        uchar* row_ptr = image.ptr<uchar>(i);
        for (size_t j = 0; j < nc; j++)
            uchar data = row_ptr[j];
    }
    std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = 
        std::chrono::duration_cast<std::chrono::duration<double>>(t_end - t_start);
    // report timed used for scanning pixels
    std::cout << "" << std::endl;

    cv::Mat another_image = image;
    another_image(cv::Rect(0, 0, 100, 100)).setTo(255);
    cv::imshow("Image", image);
    cv::waitKey(0);

    // clone
    cv::Mat clone_image = image.clone();
    clone_image(cv::Rect(0, 0, 100, 100)).setTo(255);
    cv::imshow("Image", image);
    cv::imshow("Clone_Image", clone_image);
    cv::waitKey(0);

    cv::destroyAllWindows();
    return 0;
}
