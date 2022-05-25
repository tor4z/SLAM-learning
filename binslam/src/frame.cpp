#include "binslam/frame.hpp"


namespace binslam
{

Frame::Frame(
    long id, double timestamp, const Sophus::SE3d &pose,
    const cv::Mat &left, const cv::Mat &right
): id_(id), timestamp_(timestamp), pose_(pose),
   left_image_(left), right_image_(right) {}


Frame::Ptr Frame::create()
{
    static long factory_id = 0;
    Frame::Ptr new_frame(new Frame);
    new_frame->id_ = factory_id++;
    return new_frame;
}

void Frame::setKeyFrame()
{
    static long keyframe_factory_id = 0;
    is_keyframe_ = true;
    keyframe_id_ = keyframe_factory_id++;
}

}
