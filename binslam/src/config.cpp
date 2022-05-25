#include "binslam/config.hpp"


namespace binslam
{

bool Config::setParamFile(const std::string &filename)
{
    if(config_ == nullptr)
        config_ = std::make_shared<Config>(new Config());
    config_->file_ = cv::FileStorage(filename.c_str(), cv::FileStorage::READ);

    if(!config_->file_.isOpened())
    {
        LOG(ERROR) << "parameter file " << filename << " not found.";
        config_->file_.release();
        return false;
    }
    return true;
}


Config::~Config()
{
    if(file_.isOpened())
        file_.release();
}

}
