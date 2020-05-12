#include "Segmentator.h"



Segmentator::Segmentator()
	: nonSuppresedGradientCount_(0), highThreshold_(0.0f)
{}


Segmentator::~Segmentator()
{}


void Segmentator::Canny(cv::Mat& image, cv::Mat& output, std::vector<cv::Vec4i>& edges)
{
	image.copyTo(image_);
	nonMaximumSuppressed_ = cv::Mat(image_.rows, image_.cols, CV_32FC1, cv::Scalar(0));
	output_ = cv::Mat(image_.rows, image_.cols, CV_8UC1, cv::Scalar(0));

	cv::GaussianBlur(image_, image_, cv::Size(9, 9), 4, 0);
	cv::cvtColor(image_, image_, cv::ColorConversionCodes::COLOR_BGR2GRAY);
	cv::Sobel(image_, gradients_, CV_32FC1, 0, 1, 5);
	gradients_ = abs(gradients_);


	NonMaximumSuppression();
	ComputeNonMaximumSuppressedGradientsProbabilitiesAndMean();
	ComputeHighGradientThreshold();
	DoubleThresholding();
	RemoveUnexpectedEdges();

	GetEdgesCoordinates();
	RemoveSmallGap(3);
	GetEdgesCoordinates();
	EdgesLinking(0.1);
	RemoveUnexpectedEdges();

	GetEdgesCoordinates();

	CleanUp();

	edges = edges_;
	output = output_;
}



void Segmentator::NonMaximumSuppression()
{
	float* zeros = std::vector<float>(gradients_.cols, 0).data();

	for (int i = 0; i < gradients_.rows; ++i)
	{
		float* gradients = gradients_.ptr<float>(i);
		float* output = nonMaximumSuppressed_.ptr<float>(i);

		float* aboveGradients = i > 0 ? gradients_.ptr<float>(i - 1) : zeros;
		float* belowGradients = i < gradients_.rows - 1 ? gradients_.ptr<float>(i + 1) : zeros;


		for (int j = 0; j < gradients_.cols; ++j)
		{	
			if (gradients[j] > aboveGradients[j] && gradients[j] > belowGradients[j])
			{
				output[j] = gradients[j];
				++nonSuppresedGradientCount_;
			}
		}

	}

	cv::Mat display;

	cv::convertScaleAbs(nonMaximumSuppressed_, display);
	cv::imshow("NonMaximunSuppession", display);
}

void Segmentator::ComputeNonMaximumSuppressedGradientsProbabilitiesAndMean()
{
	std::map<float, int> gradientsCount;

	//Count the number of times each gradient value appears in the image
	for (int i = 0; i < nonMaximumSuppressed_.rows; ++i)
	{
		float* gradients = nonMaximumSuppressed_.ptr<float>(i);

		for (int j = 0; j < nonMaximumSuppressed_.cols; ++j)
		{
			if (gradients[j] != 0.0f)
			{
				auto it = gradientsCount.find(gradients[j]);

				gradientsCount[gradients[j]] = it == gradientsCount.end() ?
					1 : gradientsCount[gradients[j]] + 1;
			}
		}

	}


	//Compute the probability that the gradient of a pixel is below a certain threshold
	gradientsProbabilities_.reserve(gradientsCount.size());

	for (auto it1 = gradientsCount.begin(); it1 != gradientsCount.end(); ++it1)
	{
		gradientsProbabilities_[it1->first] = 0.0f;

		for (auto it2 = gradientsCount.begin(); ; ++it2)
		{
			gradientsProbabilities_[it1->first] += it2->second;

			if (it2 == it1) { break; }
		}

		gradientsProbabilities_[it1->first] /= nonSuppresedGradientCount_;
	}


	//Compute the mean of pixel gradient below and above a certain threshold
	gradientsMeanBelow_.reserve(gradientsCount.size());
	gradientsMeanAbove_.reserve(gradientsCount.size());

	for (auto it1 = gradientsCount.begin(); it1 != gradientsCount.end(); ++it1)
	{
		gradientsMeanBelow_[it1->first] = 0.0f;
		int nBelow = 0;

		for (auto it2 = gradientsCount.begin(); ; ++it2)
		{
			gradientsMeanBelow_[it1->first] += (it2->second * it2->first);

			nBelow += it2->second;

			if (it2 == it1) { break; }
		}

		gradientsMeanBelow_[it1->first] /= (float)nBelow;
	}


	for (auto it1 = gradientsCount.begin(); it1 != gradientsCount.end(); ++it1)
	{
		gradientsMeanAbove_[it1->first] = 0.0f;
		int nAbove = 0;

		for (auto it2 = std::next(it1); it2 != gradientsCount.end(); ++it2)
		{
			gradientsMeanAbove_[it1->first] += (it2->second * it2->first);

			nAbove += it2->second;
		}

		gradientsMeanAbove_[it1->first] = nAbove != 0 ? gradientsMeanAbove_[it1->first] / (float)nAbove : 0;
	}
}


void Segmentator::ComputeHighGradientThreshold()
{
	//Find a threshold for the image gradient with the Otsu's method

	float betweenClassVariance = 0.0f;
	float max = 0.0f;
	
	for (auto prob : gradientsProbabilities_)
	{
		betweenClassVariance = (prob.second * (1 - prob.second)) * pow(gradientsMeanBelow_[prob.first] - gradientsMeanAbove_[prob.first], 2);

		if (betweenClassVariance > max)
		{
			max = betweenClassVariance;
			highThreshold_ = prob.first;
		}
	}
}


void Segmentator::DoubleThresholding()
{
	float lowThreshold = highThreshold_ / 2.0f;

	cv::Canny(image_, output_, lowThreshold, highThreshold_, 5);

	//Remove  straigth horizontal edges
	for (int i = 0; i < output_.rows; ++i)
	{
		uchar* data = output_.ptr<uchar>(i);

		for (int j = 0; j < output_.cols; ++j)
		{
			if (j == 0)
			{
				if (data[j + 1] == 0) { data[j] = 0; }
			}
			else if (j > 0)
			{
				if (data[j + 1] == 0 && data[j - 1] == 0) { data[j] = 0; }
			}
			else if (j == output_.rows - 1)
			{
				if (data[j - 1] == 0) { data[j] = 0; }
			}
		}
	}
	
	cv::imshow("Double Thresholding", output_);
}


void Segmentator::GetEdgesCoordinates()
{
	///Fing all edges coordiantes in the image$
	edges_.clear();
	int edgeLength = 0;
	int edgeNumber = 0;


	for (int i = 0; i < output_.rows; ++i)
	{
		uchar* data = output_.ptr<uchar>(i);

		for (int j = 0; j < output_.cols; ++j)
		{
			if (data[j] == 255)
			{
				if (j == output_.cols - 1)
				{
					edges_[edgeNumber][2] = j;

					edgeLength = 0;
					++edgeNumber;

					continue;
				}

				if (edgeLength == 0) { edges_.push_back(cv::Vec4i(j, i, 0, i)); }
				++edgeLength;
			}
			else
			{
				if (j > 0 && data[j - 1] == 255)
				{
					edges_[edgeNumber][2] = j - 1;
					++edgeNumber;
				}

				edgeLength = 0;
			}
			
		}
	}
}

void Segmentator::RemoveSmallGap(int gap)
{
	//Fill small breaks and gaps in the image
	std::vector<cv::Vec4i> edges;
	edges.reserve(edges_.size());
	edges = edges_;

	std::sort(edges_.begin(), edges_.end(), [](cv::Vec4i a, cv::Vec4i b) {
		if (a[1] == b[1]) { return a[0] < b[0]; }

		return a[1] < b[1];
		});


	//Linking process: if the gap between 2 edges is <= 3 pixel 
	
	for (int i = 0; i < edges_.size() - 1; ++i)
	{
		for (int j = i + 1; j < edges_.size(); ++j)
		{
			if (edges_[i][2] < edges_[j][0])
			{
				if (edges_[j][1] - edges_[i][1] <= gap && edges_[j][0] - edges_[i][2] <= gap)
				{
					edges.push_back(cv::Vec4i(edges_[i][0], edges_[i][1], edges_[j][2], edges_[i][1]));
				}
			}
		}
	}

	edges.push_back(edges_[edges_.size() - 1]);


	output_ = cv::Mat(output_.size(), CV_8UC1, cv::Scalar(0));

	for(auto it = edges.begin(); it != edges.end(); ++it)
	{
		cv::Point pt1((*it)[0], (*it)[1]);
		cv::Point pt2((*it)[2], (*it)[3]);

		cv::line(output_, pt1, pt2, cv::Scalar(255));
	}
	
	cv::imshow("Small gap removal", output_);
}


void Segmentator::EdgesLinking(float percent)
{
	//Link the horizontal edge on the image based on a threshold
	std::vector<cv::Vec4i> edges;
	int heigthThreshold = 0;

	edges.reserve(edges_.size());
	edges = edges_;

	//Fing the average height of the longest HEs
		//Sortby length DESC
	std::sort(edges_.begin(), edges_.end(), [](cv::Vec4i a, cv::Vec4i b) {
		return (a[2] - a[0]) < (b[2] - b[0]);
		});
	

	for (int i = 0; i < (edges_.size() * percent); ++i)
	{
		heigthThreshold += (output_.size().height - edges_[i][1]);
	}

	heigthThreshold /= (edges_.size() * percent);
	heigthThreshold /= 3;

	std::cout << heigthThreshold << std::endl;

	std::sort(edges_.begin(), edges_.end(), [](cv::Vec4i a, cv::Vec4i b) {
		return a[1] < b[1];
		});

	for (int i = 0; i < edges_.size() - 1; ++i)
	{
		for (int j = i + 1; j < edges_.size(); ++j)
		{
			if (edges_[i][2] < edges_[j][0] && (edges_[j][1] - edges_[i][1]) <= heigthThreshold)
			{
				edges.push_back(cv::Vec4i(edges_[i][0], edges_[j][1], edges_[j][2], edges_[j][1]));
			}
			else if (edges_[j][2] < edges_[i][0] && (edges_[j][1] - edges_[i][1]) < heigthThreshold)
			{
				edges.push_back(cv::Vec4i(edges_[j][0], edges_[j][1], edges_[i][2], edges_[j][1]));
			}
		}
	}

	edges.push_back(edges_[edges_.size() - 1]);

	output_ = cv::Mat(output_.size(), CV_8UC1, cv::Scalar(0));

	for (auto it = edges.begin(); it != edges.end(); ++it)
	{
		cv::Point pt1((*it)[0], (*it)[1]);
		cv::Point pt2((*it)[2], (*it)[3]);

		cv::line(output_, pt1, pt2, cv::Scalar(255));
	}

	cv::imshow("Edge Lnking", output_);
}



void Segmentator::RemoveUnexpectedEdges()
{
	int edgeLength = 0;
	int edgeNumber = 0;
	double edgeLengthMean = 0.0;

	std::vector<cv::Vec4i> edges;

	for (int i = 0; i < output_.rows; ++i)
	{
		uchar* data = output_.ptr<uchar>(i);

		for (int j = 0; j < output_.cols; ++j)
		{
			if (data[j] == 255)
			{
				if (edgeLength == 0) { edges.push_back(cv::Vec4i(j, i, 0, i)); }
				++edgeLength;
			}
			else
			{
				edgeLengthMean += edgeLength;

				if (j > 0 && data[j - 1] == 255)
				{
					edges[edgeNumber][2] = j - 1;
					++edgeNumber;
				}

				edgeLength = 0;
			}
			if (data[j] == 255 && j == output_.cols - 1)
			{
				edgeLengthMean += edgeLength;

				edges[edgeNumber][2] = j;

				edgeLength = 0;
				++edgeNumber;
			}
		}
	}

	edgeLengthMean /= edgeNumber;

	double edgeThreshold = 0.0;

	for (auto edge : edges)
	{
		edgeThreshold += pow(((double)edge[2] - (double)edge[0] - edgeLengthMean), 2);
	}

	edgeThreshold = sqrt(edgeThreshold / edgeNumber);

	//Remove edge that are less than the minimun threshold
	edgeLength = 0;

	for (int i = 0; i < output_.rows; ++i)
	{
		uchar* data = output_.ptr<uchar>(i);

		for (int j = 0; j < output_.cols; ++j)
		{
			if (data[j] == 255)
			{
				if (edgeLength == 0) { edges.push_back(cv::Vec4i(j, i, 0, i)); }
				++edgeLength;
			}
			else
			{
				if (j > 0 && data[j - 1] == 255)
				{
					if (edgeLength < edgeThreshold)
					{
						int pixelDeleted = 0;
						int k = j - 1;

						while (pixelDeleted < edgeLength)
						{
							data[k] = 0;

							++pixelDeleted;
							--k;
						}
					}
				}

				edgeLength = 0;
			}

			if (data[j] == 255 && j == output_.cols - 1)
			{
				if (edgeLength < edgeThreshold)
				{
					int pixelDeleted = 0;
					int k = j;

					while (pixelDeleted < edgeLength)
					{
						data[k] = 0;

						++pixelDeleted;
						--k;
					}
				}
			}
		}
	}

	cv::imshow("Unexpected edge removal", output_);
}



void Segmentator::CleanUp()
{
	std::sort(edges_.begin(), edges_.end(), [](cv::Vec4i a, cv::Vec4i b) {
		return a[1] < b[1];
		});

	std::vector<cv::Vec4i> edges;

	for (auto it1 = edges_.begin(); it1 != edges_.end() - 1; ++it1)
	{
		edges.push_back(*it1);

		auto it2 = it1 + 1;

		if ((*it2)[1] == (*it1)[1] + 1) { continue; }
		if ((*it2) == (*it1)) { continue; }

		edges.push_back(*it2);
	}

	edges_ = edges;

	output_ = cv::Mat(output_.size(), CV_8UC1, cv::Scalar(0));

	for (auto it = edges_.begin(); it != edges_.end(); ++it)
	{
		cv::Point pt1((*it)[0], (*it)[1]);
		cv::Point pt2((*it)[2], (*it)[3]);

		cv::line(output_, pt1, pt2, cv::Scalar(255));
	}

	cv::imshow("CleanUp", output_);
}