// assignment3.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

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

// 프로그램 실행: <Ctrl+F5> 또는 [디버그] > [디버깅하지 않고 시작] 메뉴
// 프로그램 디버그: <F5> 키 또는 [디버그] > [디버깅 시작] 메뉴

// 시작을 위한 팁: 
//   1. [솔루션 탐색기] 창을 사용하여 파일을 추가/관리합니다.
//   2. [팀 탐색기] 창을 사용하여 소스 제어에 연결합니다.
//   3. [출력] 창을 사용하여 빌드 출력 및 기타 메시지를 확인합니다.
//   4. [오류 목록] 창을 사용하여 오류를 봅니다.
//   5. [프로젝트] > [새 항목 추가]로 이동하여 새 코드 파일을 만들거나, [프로젝트] > [기존 항목 추가]로 이동하여 기존 코드 파일을 프로젝트에 추가합니다.
//   6. 나중에 이 프로젝트를 다시 열려면 [파일] > [열기] > [프로젝트]로 이동하고 .sln 파일을 선택합니다.
