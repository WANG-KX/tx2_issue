#include <cstdlib>
#include <deque>
#include <ctime>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include <device_image.cuh>
#include <pixel_cost.cuh>

#include <wta/wta.cuh>
#include <sgm/sgm.cuh>

using namespace std;
using namespace cv;


__global__ void SAD_aggregation(DeviceImage<float> *left_devptr, DeviceImage<float> *right_devptr, DeviceImage<PIXEL_COST> *cost_devptr)
{
  const int x = blockIdx.x;
  const int y = blockIdx.y;
  const int depth = threadIdx.x;
  const int width = left_devptr->width;
  const int height = right_devptr->height;

  if(x >= width - 1 || y >= height - 1 || x <= 0 || y <= 0)
  	return;

  if(x - 1 - depth< 0)
  {
    (cost_devptr->atXY(x,y)).set_cost(depth, 999.0f);
    return;
  }

  float cost = 0.0f;
  for(int j = 0; j < 3; j++)
  	for(int i = 0; i < 3; i++)
  	{
  		cost += fabs(left_devptr->atXY(x + i - 1, y + j - 1) - right_devptr->atXY(x + i - 1 - depth, y + j - 1));
  	}
  (cost_devptr->atXY(x,y)).set_cost(depth, cost);
}

void download_depth(DeviceImage<float> &depth, string window_name);

int main()
{
	//initial data
	string left_image_path = string("../urban1_left.pgm");
	string right_image_path = string("../urban1_right.pgm");
	Mat left_mat = imread(left_image_path, CV_LOAD_IMAGE_GRAYSCALE);
	Mat right_mat = imread(right_image_path, CV_LOAD_IMAGE_GRAYSCALE);

	//construct the data
	int width = left_mat.cols;
	int height = left_mat.rows;
	printf("the input image size: %d x %d.\n", width, height);
	Mat left_float_mat = cv::Mat::zeros(height, width, CV_32FC1);
	Mat right_float_mat = cv::Mat::zeros(height, width, CV_32FC1);
	left_mat.convertTo(left_float_mat, CV_32F, 1.0f/255.0f);
	right_mat.convertTo(right_float_mat, CV_32F, 1.0f/255.0f);

	//cuda data
	DeviceImage<float> left_image(width, height);
	DeviceImage<float> right_image(width, height);
	left_image.setDevData(reinterpret_cast<float*>(left_float_mat.data));
	right_image.setDevData(reinterpret_cast<float*>(right_float_mat.data));

	DeviceImage<PIXEL_COST> image_cost(width, height);
	image_cost.zero();

	//cost aggregation
	dim3 cost_block;
	dim3 cost_grid;
	cost_block.x = DEPTH_NUM;
	cost_grid.x = width;
	cost_grid.y = height;
	SAD_aggregation<<<cost_grid, cost_block>>>(left_image.dev_ptr, right_image.dev_ptr, image_cost.dev_ptr);
  cudaDeviceSynchronize();

  //wta
  DeviceImage<float> wta_depth(width, height);
  wta_depth.zero();
  wta(image_cost, wta_depth);

  //sgm
  DeviceImage<float> sgm_depth(width, height);
  sgm_depth.zero();
  sgm(image_cost, sgm_depth);

  download_depth(wta_depth, string("wta_depth"));	
  download_depth(sgm_depth, string("sgm_depth"));

  cv::waitKey(0);
}

void download_depth(DeviceImage<float> &depth, string window_name)
{
	int width = depth.width;
	int height = depth.height;

	float* depth_ptr = (float*) malloc(width * height * sizeof(float));

	depth.getDevData(depth_ptr);

	Mat depth_mat = cv::Mat::zeros(height, width, CV_32FC1);
	for(int i = 0; i < height; i++)
		for(int j = 0; j < width; j++)
		{
			depth_mat.at<float>(i,j) = depth_ptr[i * width + j];
		}

  cv::Mat adjMap;
  float min = 0;
  float max = DEPTH_NUM;
  depth_mat.convertTo(adjMap, CV_8UC1, 255 / (max-min), -min);
  cv::Mat falseColorsMap;
  cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_RAINBOW);
  cv::imshow(window_name, adjMap);
	free(depth_ptr);
}