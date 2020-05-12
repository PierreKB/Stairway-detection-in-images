#include "ROIValidator.h"



ROIValidator::ROIValidator(int rows, int cols)
	:rows_(rows), cols_(cols)
{}

ROIValidator::~ROIValidator()
{}


bool ROIValidator::Evaluate(std::vector<cv::Vec4i>& edges)
{
	LIS(edges);
	ComputeVanishingPoint();


	if (vanishingPoint_[1] < 0 && vanishingPoint_[1] > (-2 * cols_))
	{
		std::cout << "It's a staircase." << std::endl;
		return true;
	}
	else
	{
		std::cout << "It's not a staircase." << std::endl;
		return false;
	}
}


void ROIValidator::LIS(std::vector<cv::Vec4i>& edges)
{
	//Search for the longest increasing subsequence of the horizontal edges

	auto lowerThan = [](cv::Vec4i a, cv::Vec4i b)
	{
		return (a[0] > b[0] && a[2] < b[2] && a[1] < b[1]);
	};

	int n = edges.size();
	std::vector<int> d(n, 1), p(n, -1);

	for (int i = 0; i < n; i++) 
	{
		for (int j = 0; j < i; j++) 
		{
			if (lowerThan(edges[j], edges[i]) && d[i] < d[j] + 1)
			{
				d[i] = d[j] + 1;
				p[i] = j;
			}
		}
	}

	int ans = d[0], pos = 0;

	for (int i = 1; i < n; i++) 
	{
		if (d[i] > ans) 
		{
			ans = d[i];
			pos = i;
		}
	}

	while (pos != -1) 
	{
		edges_.push_back(edges[pos]);
		pos = p[pos];
	}

	reverse(edges_.begin(), edges_.end());

	output_ = cv::Mat(rows_, cols_, CV_8UC1, cv::Scalar(0));

	for (auto it = edges_.begin(); it != edges_.end(); ++it)
	{
		cv::Point pt1((*it)[0], (*it)[1]);
		cv::Point pt2((*it)[2], (*it)[3]);

		cv::line(output_, pt1, pt2, cv::Scalar(255));
	}

	cv::imshow("LIS", output_);
}



void ROIValidator::ComputeVanishingPoint()
{
	std::sort(edges_.begin(), edges_.end(), [](cv::Vec4i a, cv::Vec4i b) {
		if (a[1] == b[1]) { return a[0] < b[0]; }

		return a[1] < b[1];
		});

	cv::Vec4i min = edges_[0];
	cv::Vec4i max = edges_[edges_.size() - 1];

	//Compute the parameters of the line that will serve to compute the vanishing point

	float leftLineA = (min[1] - max[1]) / (min[0] - max[0]);
	float rightLineA = (min[1] - max[1]) / (min[2] - max[2]);

	float leftLineB = min[1] - (leftLineA * min[0]);
	float rightLineB = min[1] - (rightLineA * min[2]);

	vanishingPoint_[0] = (rightLineB - leftLineB) / (leftLineA - leftLineB);
	vanishingPoint_[1] = leftLineA * vanishingPoint_[0] + leftLineA;
}

std::vector<cv::Vec4i> ROIValidator::edges() { return edges_; }