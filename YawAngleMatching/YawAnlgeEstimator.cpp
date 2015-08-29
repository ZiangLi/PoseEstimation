#include "YawAngleEstimator.h"

YawAngleEstimator::YawAngleEstimator(int _AngleNum = 5, FeatureType _Feature = USE_BRISK) :AngleNum(_AngleNum), Feature(_Feature)
{
	YawTemplate.reserve(_AngleNum);
}

void YawAngleEstimator::init(const string PicFilename)
{
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

void YawAngleEstimator::Estimate(Mat& CurrentFrame, int FrameNum)
{
	if (Frames.size() < FrameNum)
		Frames.push_back(CurrentFrame);

	vector<vector<KeyPoint>> kp;
	vector<Mat> descriptors;
	featureExtract(Frames, kp, descriptors, Feature);
}