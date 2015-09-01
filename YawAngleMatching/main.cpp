#include <iostream>
#include "cxlibface.hpp"
#include "YawAngleEstimator.h"

using namespace std;
using namespace cv;

const int   MAX_face_numBER = 20;

//convert self-defined RectItem to regular Rect
void selectCenterRect(Mat& Frame,CvRectItem* FaceRegion, int num, Rect& FD)
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
	Rect                     FaceRect;
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

int main()
{
	VideoCapture cap;
	cap.open(0);
	Mat frame,faceROI;
	YawAngleEstimator* estimator = new YawAngleEstimator();
	float YawAngle=0.0f;
	estimator->init("pic_base\\ziang_0829");
	estimator->train();
	int count = 0;
	while (1)
	{
		cap >> frame;
		faceDetect(frame, faceROI);
		if (count < 300)
		{
			imshow("FaceRegion", frame);
		}
		else
		{
			faceDetect(frame, faceROI);

			imwrite("origin.jpg", frame);
			imwrite("ROI.jpg", faceROI);

			if (faceROI.data)
			{
				estimator->Estimate(faceROI, YawAngle);

				printf("Face YawAngle is %f бу", YawAngle);
			}
			imshow("FaceRegion", frame);
		}
		waitKey(1);
		count++;
	}
	delete estimator;
	
	return 0;
}