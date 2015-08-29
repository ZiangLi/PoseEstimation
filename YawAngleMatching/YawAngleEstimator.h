#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

enum FeatureType
{
	USE_BRISK    = 0,
	USE_SURF     = 1,
	USE_LANDMARK = 2,
	USE_ORB      = 3
};

class YawAngleEstimator
{
public:
	//Default construct function
	YawAngleEstimator(int AngleNum = 5, FeatureType Feature = USE_BRISK);

	//To get template pictures from one folder,
	virtual void init(string PicFilename);

	//Construct Yaw-Template Index from assigned feature
	virtual void train();

	//Estimate head pose from last few frames 
	virtual void Estimate(int FrameNum = 3);

private:
	float				 YawAngle;
	int					 AngleNum;
	vector<Mat>			 YawTemplate;
	vector<Mat>		     Frames;
	vector<flann::Index> YawIndex;
	FeatureType Feature;
};