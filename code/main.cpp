#include "Segmentator.h"
#include "ROIValidator.h"


int main(int argc, char* argv[])
{
	if (argc < 2)
	{
		std::cout << "Please provide a file." << std::endl;
	}

	cv::Mat image = cv::imread(argv[1]);

	Segmentator segmentator;
	ROIValidator validator(image.rows, image.cols);

	cv::Mat output;
	std::vector<cv::Vec4i> edges;

	segmentator.Canny(image, output, edges);


	if (validator.Evaluate(edges))
	{
		std::sort(edges.begin(), edges.end(), [](cv::Vec4i a, cv::Vec4i b) {
			return a[1] < b[1];
			});
		
	
		int consecutiveHeigthMean2 = 0.0f;

		auto validatorEdges = validator.edges();

		for (int i = 0; i < validatorEdges.size() - 1; ++i)
		{
			consecutiveHeigthMean2 += (validatorEdges[i + 1][1] - validatorEdges[i][1]);
		}
		consecutiveHeigthMean2 /= (validatorEdges.size());

	
		
		std::cout << "Number of stairs: " << (validatorEdges[validatorEdges.size() - 1][1] - validatorEdges[0][1]) / consecutiveHeigthMean2 << std::endl;
	}
	

	cv::waitKey(0);
}