#pragma once

#ifndef ILC_API
#define ILC_API __declspec( dllexport )
#endif

#include "common.h"

namespace robot 
{
    class ILC_API Fusion
    {
    public:
        void preferFirst( const std::vector<cv::Point2f> & firstPoints,
            const std::vector<int> & firstClasses,
            const std::vector<cv::Point2f> & secondPoints, 
            const std::vector<int> & secondClasses,
            std::vector<cv::Point2f> & fusedPoints, 
            std::vector<int> & fusedClasses );
    };

}
