#include "YawAngleEstimator.h"

YawAngleEstimator::YawAngleEstimator(int _FrameNum = 3, FeatureType _Feature = USE_BRISK) :FrameNum(_FrameNum),Feature(_Feature)
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

void YawAngleEstimator::init(const string PicFilename,int _AngleNum = 5)
{
	AngleNum = _AngleNum;
	FinalVote = new float[_AngleNum];
	for (int i = 0; i < _AngleNum; i++)
		FinalVote[i] = 0;

	YawTemplate.reserve(_AngleNum);
	Angle.reserve(_AngleNum);

	ifstream fs;
	for (int i = 0; i < 5; i++)
	{
		int j = 0;
		Rect ROI;
		char buffer[16];
		float coordinate[4];
		sprintf(buffer, "%d", i + 1);

		fs.open(PicFilename + "\\" + buffer + ".mk");
		while (j<4)
		{
			fs >> coordinate[j];
			j++;
		}
		fs >> Angle[i];
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
	switch (Feature)
	{
	case USE_BRISK:
	{
		BRISK* detector = new BRISK();
		descriptImg(ImgTmp, detector, detector, kp, descriptors);
		delete detector;
	}
	break;
	case USE_SURF:
	{
		SURF* detector = new SURF(300);
		SURF* extractor = new SURF();
		descriptImg(ImgTmp, detector, extractor, kp, descriptors);
		delete detector;
		delete extractor;
	}
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
	vector<vector<KeyPoint>> kp;
	vector<Mat> descriptors;
	featureExtract(YawTemplate, kp, descriptors, Feature);

	//build Index with Lsh and Hamming distance
	for (int i = 0; i < AngleNum; i++)
		YawIndex[i].build(descriptors[i], flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);
}

void YawAngleEstimator::Estimate(Mat& CurrentFrame)
{
	vector<Mat> MatchingImg(1, CurrentFrame);
	vector<vector<KeyPoint>> CurrentKp;
	vector<Mat> CurrentDescriptors;
	float* CurrentVote = new float[AngleNum];
	int index;
	float maxVote=0;
	//Extract current feature
	featureExtract(MatchingImg, CurrentKp, CurrentDescriptors, Feature);

	Mat matchIndex(CurrentDescriptors[0].rows, 2, CV_32SC1), matchDistance(CurrentDescriptors[0].rows, 2, CV_32FC1);

	//Lowe's Algorithm to caculate vote for each angletemplate
	for (int i = 0; i < AngleNum; i++)
	{
		CurrentVote[i] = 0;
		YawIndex[i].knnSearch(CurrentDescriptors[0], matchIndex, matchDistance, 2,flann::SearchParams());
		for (int j = 0; j < matchDistance.rows; j++)
		{
			if (matchDistance.at<float>[j][0] < 0.6*matchDistance.at<float>[j][0])
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

}

float YawAngleEstimator::outputAngle()
{
	return Angle[AngleIndex];
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