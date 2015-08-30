#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <vector>
#include <deque>
#include <string>
#include <fstream>

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
	YawAngleEstimator(int FrameNum = 3, FeatureType Feature = USE_BRISK);

	//Default destruct function
	~YawAngleEstimator();

	//To get template pictures from one folder
	virtual void init(const string PicFilename, int AngleNum = 5);

	//Construct Yaw-Template Index from assigned feature
	virtual void train();

	//Estimate head pose from last few frames 
	virtual void Estimate(Mat& CurrentFrame, float& CurrentAngle);

	//Release Memory Manually
	virtual void release();

private:

	float*				 VoteWeight;
	float*				 FinalVote;
	int					 AngleNum;
	int					 FrameNum;
	int                  AngleIndex;
	vector<Mat>			 YawTemplate;
	vector<float>        Angle;
	deque<float*>		 FramesVote;
	vector<flann::Index> YawIndex;
	FeatureType Feature;
};