#include <iostream>
#include "cxlibface.hpp"
#include "YawAngleEstimator.h"
#include "tracker\FaceTracker.h"

using namespace std;
using namespace cv;
using namespace robot;

const int   MAX_face_numBER = 20;

#define CAMERA

//convert self-defined RectItem to regular Rect in center
void selectCenterRect(Mat& Frame, CvRectItem* FaceRegion, int num, Rect& FD)
{
	int centerX = Frame.cols / 2;
	int centerY = Frame.rows / 2;
	int minDistance = 10000;
	int index = num + 1;
	for (int i = 0; i < num; i++)
	{

		int distance = abs(FaceRegion[i].rc.x - centerX) + abs(FaceRegion[i].rc.y - centerX);
		if (distance < minDistance)
		{
			minDistance = distance;
			index = i;
		}
	}
	FD = Rect(FaceRegion[index].rc);
	rectangle(Frame, FD, Scalar(0, 0, 255), 2, 8);
}

//Face Detect with facelib
int faceDetect(Mat& Frame, Rect& FaceRect)
{
	CxlibFaceDetector*		 face_detector = new CxlibFaceDetector();
	CvRectItem				 FaceRegion[MAX_face_numBER];

	face_detector->init();
	IplImage* FrameTemp = &IplImage(Frame);
	face_detector->SetFaceDetectionSizeRange(FrameTemp);
	face_detector->SetFaceDetectionROI(FrameTemp, 0.8);
	int nFacenum = face_detector->detect(FrameTemp, FaceRegion, MAX_face_numBER);
	delete face_detector;

	if (nFacenum != 0)
	{
		selectCenterRect(Frame, FaceRegion, nFacenum, FaceRect);
	}
	return nFacenum;
}

int main()
{
	
#ifdef CAMERA
	VideoCapture cap;
	cap.open(0);
#endif

	YawAngleEstimator* estimator = new YawAngleEstimator(3,USE_SIFT,false);
	float YawAngle = 0.0f;
	estimator->init("pic_base\\ziang_0829");
	estimator->train();

	bool hasFace = false;

	FaceTracker *ft = new FaceTracker();
	ft->consensus.estimate_rotation = false;
	ft->consensus.estimate_scale = true;

	while (1)
	{
		Mat frame, faceROI;
		Mat gray_frame;
		Rect FaceRect;

#ifdef CAMERA
		cap >> frame;
#endif

#ifndef CAMERA
		frame = imread("1.jpeg");
		faceROI = frame;
#endif
		cvtColor(frame, gray_frame, CV_BGR2GRAY);

		if (!hasFace)
		{
			int facenum=faceDetect(frame, FaceRect);

			if (facenum==0)
			{
				imshow("FaceRegion", frame);
				waitKey(1);
				continue;
			}
			else
			{
				hasFace = true;
				ft->initialize(gray_frame, FaceRect);
			}
		}
		else
		{
			ft->processFrame(gray_frame);

			FaceRect = ft->bb_rot.boundingRect();
			faceROI = frame(FaceRect);
			estimator->Estimate(faceROI, YawAngle);			
			rectangle(frame, FaceRect, Scalar(0, 255, 0), 2, 8);

			char buffer[32];
			sprintf(buffer, "Face YawAngle: %d", (int)YawAngle);
			putText(frame, buffer, FaceRect.tl() - Point(0, 8), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 255));
		}
#ifdef RECORD
		{
			char buffer[32];
			static int i = 0;
			sprintf(buffer, "%d.jpg", i++);
			imwrite(buffer, frame);
		}
#endif
		imshow("FaceRegion", frame);

		waitKey(1);

	}
	estimator->release();

	delete estimator;
	delete ft;

	return 0;
}