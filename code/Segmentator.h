#ifndef SEGMENTATOR_H
#define SEGMENTATOR_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <map>
#include <unordered_map>


class Segmentator
{
public:
	Segmentator();
	~Segmentator();
	
	void Canny(cv::Mat& image, cv::Mat& output,  std::vector<cv::Vec4i>& edges);



private:
	void NonMaximumSuppression();
	void ComputeNonMaximumSuppressedGradientsProbabilitiesAndMean();
	void ComputeHighGradientThreshold();
	void DoubleThresholding();
	void RemoveUnexpectedEdges();
	void GetEdgesCoordinates();
	void RemoveSmallGap(int gap);
	void EdgesLinking(float percent);

	void CleanUp();
	

	cv::Mat image_;
	cv::Mat gradients_;
	cv::Mat nonMaximumSuppressed_;
	cv::Mat output_;
	
	std::unordered_map<float, float> gradientsProbabilities_;
	std::unordered_map<float, float> gradientsMeanBelow_;
	std::unordered_map<float, float> gradientsMeanAbove_;

	//Edge coord are represented as Xleft, Y, Xright, Y
	std::vector<cv::Vec4i> edges_;

	float highThreshold_;

	int nonSuppresedGradientCount_;
};


#endif