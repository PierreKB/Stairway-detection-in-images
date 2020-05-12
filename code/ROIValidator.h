#ifndef ROI_VALIDATOR
#define ROI_VALIDATOR

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <map>
#include <unordered_map>


class ROIValidator
{
public:
	ROIValidator(int rows, int cols);
	~ROIValidator();
	
	bool Evaluate(std::vector<cv::Vec4i>& edges);

	std::vector<cv::Vec4i> edges();

private:
	void LIS(std::vector<cv::Vec4i>& edges);
	void ComputeVanishingPoint();

	cv::Mat output_;

	std::vector<cv::Vec4i> edges_;
	cv::Vec2f vanishingPoint_;

	int rows_;
	int cols_;
};


#endif