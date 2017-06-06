#include <wta/wta.cuh>

__global__ void wta_kernel(DeviceImage<PIXEL_COST> *cost_devptr, DeviceImage<float> *depth_devptr)
{
  const int x = blockIdx.x;
  const int y = blockIdx.y;
  const int depth = threadIdx.x;

  __shared__ float cost[DEPTH_NUM];
  __shared__ float min_cost[DEPTH_NUM];
  __shared__ float min_index[DEPTH_NUM];
  min_cost[depth] = cost[depth] = (cost_devptr->atXY(x,y)).get_cost(depth);
  min_index[depth] = depth;
  __syncthreads();

  for(int i = DEPTH_NUM/2; i > 0; i /= 2)
  {
  	if( depth < i && min_cost[depth + i] < min_cost[depth] )
  	{
  		min_cost[depth] = min_cost[depth + i];
  		min_index[depth] = min_index[depth + i];
  	}
  	__syncthreads();
  }

  //sub pixel depth
  if(depth == 0)
  {
  	int min = min_index[0];
  	if(min == 0 || min == DEPTH_NUM - 1)
  	  depth_devptr->atXY(x,y) = min;
  	else
  	{
  		float pre_cost = cost[min - 1];
  		float pro_cost = cost[min + 1];
  		float a = pre_cost - 2.0f * min_cost[0] + pro_cost;
      float b = - pre_cost + pro_cost;
      depth_devptr->atXY(x,y) = (float) min - b / (2.0f * a);
  	}
  }
}

void wta(DeviceImage<PIXEL_COST> &cost, DeviceImage<float> &depth)
{
	int width = cost.width;
	int height = cost.height;

	dim3 wta_block;
	dim3 wta_grid;
	wta_block.x = DEPTH_NUM;
	wta_grid.x = width;
	wta_grid.y = height;

  struct timeval start, end;
  gettimeofday(&start,NULL);
	wta_kernel<<<wta_grid, wta_block>>>(cost.dev_ptr, depth.dev_ptr);
  cudaDeviceSynchronize();
  gettimeofday(&end,NULL);
  float time_use = (end.tv_sec-start.tv_sec) * 1000.0 + (end.tv_usec-start.tv_usec) / 1000.0f;
  printf("wta cost: %lf ms.\n",time_use);
}
