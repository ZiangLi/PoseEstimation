#include "YawAngleEstimator.h"

ofstream fss("Results.txt");

YawAngleEstimator::YawAngleEstimator(int _FrameNum, FeatureType _Feature, bool _useIndex) 
	:FrameNum(_FrameNum), Feature(_Feature), useIndex(_useIndex)
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
	matchers.reserve(_AngleNum);

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
		Mat dstTemp = srcTemp(ROI).clone();
		if (!dstTemp.data)
		{
			printf("Error for read!");
			return;
		}
		//imshow(buffer, dstTemp);
		waitKey(1000);
		cvtColor(dstTemp, dstTemp, CV_BGR2GRAY);

		waitKey(1);
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
		BRISK* detector = new BRISK(30);
		BRISK* extractor = new BRISK();
		descriptImg(ImgTmp, detector, extractor, kp, descriptors);
		delete detector;
		delete extractor;
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

	if (useIndex)
	{
		//build Index with Lsh and Hamming distance
		for (int i = 0; i < AngleNum; i++)
		{
			flann::Index tempIndex;
			if (Feature == USE_SIFT)
			{
				tempIndex.build(descriptors[i], flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_L2);
			}
			else
			{
				tempIndex.build(descriptors[i], flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);
			}
			YawIndex.push_back(tempIndex);
		}
	}
	else
	{
		//build BFMathers
		for (int i = 0; i < AngleNum; i++)
		{
			//record
			fss<<"\nKeypoints number of template "<< i <<" is "<< descriptors[i].rows << endl;

			BFMatcher tempMatcher;
			vector<Mat> train_des(1, descriptors[i]);
			tempMatcher.add(train_des);
			tempMatcher.train();
			matchers.push_back(tempMatcher);
		}
	}
}

void YawAngleEstimator::Indexmatch(Mat& CurrentDescriptors, float* CurrentVote)
{
	//Lowe's Algorithm to caculate vote for each angletemplate
	for (int i = 4; i < AngleNum; i++)
	{
		//rows:numbers of keypoints
		Mat matchIndex(CurrentDescriptors.rows, 2, CV_32SC1), matchDistance(CurrentDescriptors.rows, 2, CV_32FC1);

		CurrentVote[i] = 0;

		//cout << CurrentDescriptors[0] << endl;

		YawIndex[i].knnSearch(CurrentDescriptors, matchIndex, matchDistance, 2, flann::SearchParams());
		for (int j = 0; j < matchDistance.rows; j++)
		{
			if (matchDistance.at<float>(j, 0) < 0.6*matchDistance.at<float>(j, 1))
				++CurrentVote[i];
		}
	}
}
void YawAngleEstimator::BFmatch(Mat& CurrentDescriptors, float* CurrentVote)
{
	for (int i = 0; i < AngleNum; i++)
	{
		vector<vector<DMatch>>   matches;
		matchers[i].knnMatch(CurrentDescriptors, matches, 2);
		CurrentVote[i] = 0;

		vector<DMatch> goodMatches;
		for (int j = 0; j < matches.size(); j++)
		{
			//printf("The Nearest  Two Distance of point %d for template %d is: ", j, i);
			//cout << matches[j][0].distance <<","<< matches[j][1].distance << endl;
			if (matches[j][0].distance<380 && matches[j][0].distance < 0.85*matches[j][1].distance)
				++CurrentVote[i];
		}

		//printf("The Current Number of Vote for template %d is %f\n", i, CurrentVote[i]);
	}
}

//TODO:FlannIndexEstimate Debug
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

	//draw Keypoints on frame and record
	drawKeypoints(CurrentFrame, CurrentKp[0], CurrentFrame);
	fss <<"\nKeypoints number of CurrentFrame is "<<CurrentDescriptors[0].rows<<endl;

		printf("Estimate::The Number of keypoints is:%d\n", CurrentDescriptors[0].rows);

	if (useIndex)
		Indexmatch(CurrentDescriptors[0], CurrentVote);
	else
		BFmatch(CurrentDescriptors[0], CurrentVote);

	if (FramesVote.size() < FrameNum)
		FramesVote.push_back(CurrentVote);
	else//caculate FinalVote for each angle from last frame's vote
	{
		float* ptrTemp = FramesVote.front();
		FramesVote.pop_front();
		FramesVote.push_back(CurrentVote);
		delete ptrTemp;

		for (int i = 0; i < FrameNum; i++)
		{
			for (int j = 0; j < AngleNum; j++)
			{
				FinalVote[j] = FramesVote[i][j] * VoteWeight[i];
				
				if (FinalVote[j] > maxVote)
				{
					maxVote = FinalVote[j];
					AngleIndex = j;
				}
			}
		}

		for (int i = 0; i < AngleNum; i++)
		{
			cout << "Final Vote for template " << i << " is " << FinalVote[i] << endl;
			
			//record
			fss << "Final Vote for template " << i << " is " << FinalVote[i] << endl;
		}
	}
	if (AngleIndex == -1)
	{
		printf("Cannot estimate right now!\n");
		return false;
	}
	else
	{
		CurrentAngle = Angle[AngleIndex];
		printf("The max vote of angle is: %f\n", maxVote);
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