#pragma once

#include "common.h"
#include "Consensus.h"
#include "Fusion.h"
#include "Matcher.h"
#include "Tracker.h"

#include <opencv2/features2d/features2d.hpp>

#ifndef ILC_API
#define ILC_API __declspec( dllexport )
#endif

namespace robot
{
    class ILC_API FaceTracker
    {
    public:
        FaceTracker() : str_detector( "FAST" ), str_descriptor( "BRISK" ) {};
        void initialize( const cv::Mat im_gray, const cv::Rect rect );
        void processFrame( const cv::Mat im_gray );

        Fusion fusion;
        Matcher matcher;
        Tracker tracker;
        Consensus consensus;

        std::string str_detector;
        std::string str_descriptor;

        std::vector<cv::Point2f> points_active;
        cv::RotatedRect bb_rot;
        bool is_failed;

    private:
        cv::Ptr<cv::FeatureDetector> detector;
        cv::Ptr<cv::DescriptorExtractor> descriptor;

        cv::Size2f size_initial;
        std::vector<int> classes_active;
        float theta;

        cv::Mat im_prev;
    };
}