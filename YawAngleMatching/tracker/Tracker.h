#pragma once

#ifndef ILC_API
#define ILC_API __declspec( dllexport )
#endif

#include "common.h"

namespace robot 
{
    class ILC_API Tracker
    {
    public:
        Tracker() : thr_fb( 30 ) {};
        void track( const cv::Mat im_prev, 
            const cv::Mat im_gray, 
            const std::vector<cv::Point2f> & points_prev,
            std::vector<cv::Point2f> & points_tracked, 
            std::vector<unsigned char> & status );

    private:
        float thr_fb;
    };

}
