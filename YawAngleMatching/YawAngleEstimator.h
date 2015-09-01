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
	USE_SIFT	 = 1,
	USE_LANDMARK = 2,
	USE_ORB      = 3
};

class YawAngleEstimator
{
public:
	//Default construct function
	YawAngleEstimator(int FrameNum = 3, FeatureType Feature = USE_BRISK, bool useIndex = true);

	//Default destruct function
	~YawAngleEstimator();

	//To get template pictures from one folder
	virtual void init(const string PicFilename, int AngleNum = 5);

	//Construct Yaw-Template Index from assigned feature
	virtual void train();

	//Estimate head pose from last few frames 
	virtual bool Estimate(Mat& CurrentFrame, float& CurrentAngle);

	//Release Memory Manually
	virtual void release();

	void BFmatch(Mat& CurrentDescriptors, float* CurrentVote);

	void Indexmatch(Mat& CurrentDescriptors, float* CurrentVote);


private:

	float*				 VoteWeight;
	float*				 FinalVote;
	bool				 useIndex;
	int					 AngleNum;
	int					 FrameNum;
	int                  AngleIndex;
	vector<Mat>			 YawTemplate;
	vector<float>        Angle;
	deque<float*>		 FramesVote;
	vector<flann::Index> YawIndex;
	vector<BFMatcher>    matchers;
	FeatureType Feature;
};

//TODO
//flannIndex 训练
//无输入时队列弹出
//vote 相同时的策略
//平滑结果
//landmark 正负号
//深度信息，其他特征