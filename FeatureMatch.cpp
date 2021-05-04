
#include "pch.h"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

int main()
{
	Mat right = imread("right.jpg", 0);
	Mat right_c = imread("right.jpg");
	Mat left = imread("left.jpg", 0);
	Mat left_c = imread("left.jpg");
	Mat out, descriptor1, descriptor2;
	Ptr<FeatureDetector> detector = BRISK::create(100);
	vector<KeyPoint> keyPoints1, keyPoints2;
	vector<DMatch> matches, matches2;

	detector->detectAndCompute(right,Mat(), keyPoints1, descriptor1);
	detector->detectAndCompute(left, Mat(), keyPoints2, descriptor2);

	Ptr<BFMatcher> matcher = BFMatcher::create(NORM_HAMMING);
	matcher->match(descriptor2, descriptor1, matches);

	for (auto& m : matches) {
		if (m.distance < 100) {
			matches2.push_back(m);
		}
	}

	drawMatches(left_c, keyPoints2, right_c, keyPoints1, matches2, out);

	imshow("Output", out);
	waitKey();
	return 0;

}
