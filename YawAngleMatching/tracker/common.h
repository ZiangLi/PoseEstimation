#pragma once

#ifndef ILC_API
#define ILC_API __declspec( dllexport )
#endif

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

namespace robot
{
    ILC_API float median( std::vector<float> & A );
    ILC_API cv::Point2f rotate( const cv::Point2f v, const float angle );

    template<class T>
    ILC_API int sgn( T x )
    {
        return x >= 0 ? 1 : -1;
    }
}
