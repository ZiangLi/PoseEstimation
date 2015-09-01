#include "YawAngleEstimator.h"

YawAngleEstimator::YawAngleEstimator(int _FrameNum , FeatureType _Feature ) :FrameNum(_FrameNum),Feature(_Feature)
{
	VoteWeight = new float[_FrameNum];
	float sum=0.0;
	for (int i = 1; i <= _FrameNum; i++)
	{
		sum += i;
	}
	for (int i = 1; i <= _FrameNum; i++)
	{
		VoteWeight[i-1] = (float(i)/ sum);
	}
	AngleIndex = -1;
}
YawAngleEstimator::~YawAngleEstimator()
{
	delete VoteWeight;
	delete FinalVote;
	for (int i = 0; i < FrameNum; i++)
	{
		delete FramesVote[i];
	}
}

void YawAngleEstimator::init(const string PicFilename,int _AngleNum )
{
	printf("YawAngleEstimator:init\n");

	AngleNum = _AngleNum;
	FinalVote = new float[_AngleNum];
	for (int i = 0; i < _AngleNum; i++)
		FinalVote[i] = 0;

	YawTemplate.reserve(_AngleNum);
	Angle.reserve(_AngleNum);
	YawIndex.reserve(_AngleNum);

	ifstream fs;
	for (int i = 0; i < 5; i++)
	{
		int j = 0;
		Rect ROI;
		float Angletemp;
		char buffer[16];
		float coordinate[4];
		sprintf(buffer, "%d", i + 1);

		fs.open(PicFilename + "\\" + buffer + ".mk");
		while (j<4)
		{
			fs >> coordinate[j];
			j++;
		}
		fs >> Angletemp;
		Angle.push_back(Angletemp);
		fs.close();

		ROI.x = coordinate[0];
		ROI.y = coordinate[1];
		ROI.width = coordinate[2];
		ROI.height = coordinate[3];

		Mat srcTemp = imread(PicFilename + "\\" + buffer + ".jpg");
		Mat dstTemp = srcTemp(ROI);
		if (!dstTemp.data)
		{
			printf("Error for read!");
			return;
		}
		imshow(buffer, dstTemp);
		waitKey(1000);

		YawTemplate.push_back(dstTemp);
	}
}

//Detect keypoints and compute descriptors
void descriptImg(const vector<Mat>& ImgTmp, Feature2D* Detector, Feature2D* Extractor,
				vector<vector<KeyPoint>>& kp, vector<Mat>& descriptors)
{
	for (int i = 0; i < ImgTmp.size(); i++)
	{
		Detector->detect(ImgTmp[i], kp[i]);
		Extractor->compute(ImgTmp[i], kp[i],descriptors[i]);
	}
}

//Select one feature to extract
//TODO:landmark extractor
void featureExtract(const vector<Mat>& ImgTmp, vector<vector<KeyPoint>>& kp, 
					vector<Mat>& descriptors,FeatureType Feature)
{
	printf("YawAngleEstimator:featureExtract\n");

	switch (Feature)
	{
	case USE_BRISK:
	{
		BRISK* detector = new BRISK();
		descriptImg(ImgTmp, detector, detector, kp, descriptors);
		delete detector;
	}
	break;
	case USE_SIFT:
	{
		SIFT* detector = new SIFT();
		SIFT* extractor = new SIFT();
		descriptImg(ImgTmp, detector, extractor, kp, descriptors);
		delete detector;
		delete extractor;
	}
	break;
	case USE_ORB:
	{
		ORB* detector = new ORB();
		ORB* extractor = new ORB();
		descriptImg(ImgTmp, detector, extractor, kp, descriptors);
		delete detector;
		delete extractor;
	}
	case USE_LANDMARK://TODO
		break;
	}
}

void YawAngleEstimator::train()
{
	printf("YawAngleEstimator:train\n");
	vector<vector<KeyPoint>> kp(AngleNum,vector<KeyPoint>());
	vector<Mat> descriptors(AngleNum,Mat());
	featureExtract(YawTemplate, kp, descriptors, Feature);

	//build Index with Lsh and Hamming distance
	for (int i = 0; i < AngleNum; i++)
	{
		flann::Index tempIndex;
		if (Feature == USE_SIFT)
		{
			tempIndex.build(descriptors, flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_L2);
		}
		else
		{
			tempIndex.build(descriptors[i], flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);
		}
		YawIndex.push_back(tempIndex);
	}
}

bool YawAngleEstimator::Estimate(Mat& CurrentFrame,float& CurrentAngle)
{
	printf("YawAngleEstimator:estimate\n");

	Mat TempFrame = CurrentFrame.clone();
	if (TempFrame.channels() == 3)
		cvtColor(TempFrame, TempFrame, CV_BGR2GRAY);

	vector<Mat> MatchingImg(1, TempFrame);
	vector<vector<KeyPoint>> CurrentKp(1, vector<KeyPoint>());
	vector<Mat> CurrentDescriptors(1,Mat());
	float* CurrentVote = new float[AngleNum];
	float maxVote=0;
	//Extract current feature
	featureExtract(MatchingImg, CurrentKp, CurrentDescriptors, Feature);

	Mat matchIndex(CurrentDescriptors[0].rows, 2, CV_32SC1), matchDistance(CurrentDescriptors[0].rows, 2, CV_32FC1);

	//Lowe's Algorithm to caculate vote for each angletemplate
	for (int i = 0; i < AngleNum; i++)
	{
		CurrentVote[i] = 0;
		cout << CurrentDescriptors[0] << endl;
		YawIndex[i].knnSearch(CurrentDescriptors[0], matchIndex, matchDistance, 2,flann::SearchParams());
		for (int j = 0; j < matchDistance.rows; j++)
		{
			if (matchDistance.at<float>(j,0) < 0.6*matchDistance.at<float>(j,1))
				++CurrentVote[i];
		}
	}
	
	if (FramesVote.size() < FrameNum)
		FramesVote.push_back(CurrentVote);
	else//caculate FinalVote for each angle from last frame's vote
	{
		float* ptrTemp = FramesVote.front();
		FramesVote.pop_front();
		FramesVote.push_back(CurrentVote);
		delete ptrTemp;

		for (int i = 0; i < FrameNum; i++)
			for (int j = 0; j < AngleNum; j++)
			{
				FinalVote[j] = FramesVote[i][j] * VoteWeight[i];
				if (FinalVote[j]>maxVote)
				{
					maxVote = FinalVote[j];
					AngleIndex = j;
				}
			}
	}
	if (AngleIndex == -1) 
		return false;
	else
	{
		CurrentAngle = Angle[AngleIndex];
		printf("The max vote of angle is: %d\n", maxVote);
		return true;
	}
}

void YawAngleEstimator::release()
{
	delete VoteWeight;
	delete FinalVote;

	YawTemplate.swap(vector<Mat>());
	YawIndex.swap(vector<flann::Index>());
	Angle.swap(vector<float>());
	for (int i = 0; i < FrameNum; i++)
	{
		delete FramesVote[i];
	}
}