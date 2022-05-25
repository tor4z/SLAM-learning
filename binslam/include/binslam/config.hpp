#pragma once



#include "binslam/common.hpp"


namespace binslam
{

class Config
{
public:
    ~Config();
    static bool setParamFile(const std::string &filename);
    template<typename T>
    static T get(const std::string &key)
    {

    }
private:
    static std::shared_ptr<Config> config_;
    cv::FileStorage file_;
    Config(){}
};


}