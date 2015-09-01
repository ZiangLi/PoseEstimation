#include <iostream>
#include "cxlibface.hpp"
#include "YawAngleEstimator.h"

using namespace std;
using namespace cv;

const int   MAX_face_numBER = 20;
Rect FaceRect;
#define CAMERA

//convert self-defined RectItem to regular Rect
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

void faceDetect(Mat& Frame, Mat& faceROI)
{
	CxlibFaceDetector*		 face_detector = new CxlibFaceDetector();
	CvRectItem				 FaceRegion[MAX_face_numBER];
	//Rect                     FaceRect;
	face_detector->init();

	faceROI = Mat();
	IplImage* FrameTemp = &IplImage(Frame);
	face_detector->SetFaceDetectionSizeRange(FrameTemp);
	face_detector->SetFaceDetectionROI(FrameTemp, 0.8);
	int nFacenum = face_detector->detect(FrameTemp, FaceRegion, MAX_face_numBER);
	delete face_detector;

	if (nFacenum == 0) return;

	selectCenterRect(Frame, FaceRegion, nFacenum, FaceRect);
	faceROI = Frame(FaceRect);
}

int  main()
{

#ifdef CAMERA
	VideoCapture cap;
	cap.open(0);
#endif

	YawAngleEstimator* estimator = new YawAngleEstimator(3,USE_SIFT,false);
	float YawAngle = 0.0f;
	estimator->init("pic_base\\ziang_0829");
	estimator->train();

	while (1)
	{
		Mat frame, faceROI;
#ifdef CAMERA
		cap >> frame;
#endif

#ifndef CAMERA
		frame = imread("1.jpeg");
		faceROI = frame;
#endif

		faceDetect(frame, faceROI);


		//imwrite("origin.jpg", frame);
		//imwrite("ROI.jpg", faceROI);

		if (faceROI.data)
		{
			char buffer[32];
			estimator->Estimate(faceROI, YawAngle);
			sprintf(buffer, "Face YawAngle %d", (int)YawAngle);
			putText(frame, buffer, FaceRect.tl()-Point(10,5), CV_FONT_HERSHEY_COMPLEX,0.6, Scalar(255, 0, 0));
		}
		imshow("FaceRegion", frame);

		waitKey(1);

	}
	estimator->release();
	delete estimator;

	return 0;
}