#pragma once

#ifndef ILC_API
#define ILC_API __declspec( dllexport )
#endif

#include "common.h"

#include <opencv2/core/core.hpp>
#include <string>

ILC_API void screenLog( cv::Mat im_draw, const std::string text);
ILC_API cv::Rect getRect(const cv::Mat im, const std::string win_name);
