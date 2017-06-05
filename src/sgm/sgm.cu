#include <sgm/sgm.cuh>

void sgm(DeviceImage<PIXEL_COST> &cost, DeviceImage<float> &depth)
{
	int width = cost.width;
	int height = cost.height;

	DeviceImage<PIXEL_COST> sgm_cost(width, height);
	sgm_cost.zero();

	dim3 sgm_row_block;
	dim3 sgm_row_grid;
	sgm_row_block.x = DEPTH_NUM;
	sgm_row_grid.x = height;

	dim3 sgm_col_block;
	dim3 sgm_col_grid;
	sgm_col_block.x = DEPTH_NUM;
	sgm_col_grid.x = width;
  
  std::clock_t start_time = std::clock();

  sgm_cost_row_kernel<<<sgm_row_grid, sgm_row_block>>>(true, cost.dev_ptr, sgm_cost.dev_ptr);
  sgm_cost_row_kernel<<<sgm_row_grid, sgm_row_block>>>(false, cost.dev_ptr, sgm_cost.dev_ptr);
  sgm_cost_col_kernel<<<sgm_col_grid, sgm_col_block>>>(true, cost.dev_ptr, sgm_cost.dev_ptr);
  sgm_cost_col_kernel<<<sgm_col_grid, sgm_col_block>>>(false, cost.dev_ptr, sgm_cost.dev_ptr);
  cudaDeviceSynchronize();

  dim3 depth_filter_block;
  dim3 depth_filter_grid;
  depth_filter_block.x = DEPTH_NUM;
  depth_filter_grid.x = width;
  depth_filter_grid.y = height;
  sgm_filter<<<depth_filter_grid, depth_filter_block>>>(sgm_cost.dev_ptr, depth.dev_ptr);
  cudaDeviceSynchronize();

  printf("sgm cost: %lf ms.\n",(std::clock()-start_time)/(double)CLOCKS_PER_SEC*1000);
}

__global__ void sgm_cost_row_kernel(bool to_left, DeviceImage<PIXEL_COST> *cost_devptr, DeviceImage<PIXEL_COST> *sgm_cost_devptr)
{
  const int width = cost_devptr->width;
  const int height = cost_devptr->height;
  const int depth_id = threadIdx.x;

  int y = blockIdx.x;
  int x, delta_x;
  if(to_left)
  {
    x = 0;
    delta_x = 1;
  }
  else
  {
    x = width - 1;
    delta_x = -1;
  }

  __shared__ float last_cost[DEPTH_NUM], last_cost_min[DEPTH_NUM];;
  __shared__ float this_cost[DEPTH_NUM];

  last_cost_min[depth_id] = last_cost[depth_id] = 0.0;
  __syncthreads();

  for( ; x < width && x >= 0; x += delta_x)
  {
    float* my_add_ptr = (sgm_cost_devptr->atXY(x,y)).cost_ptr(depth_id);
    this_cost[depth_id] = (cost_devptr->atXY(x,y)).get_cost(depth_id);
    __syncthreads();
    for(int i = DEPTH_NUM/2; i > 0; i /= 2)
    {
      if(depth_id < i && last_cost_min[depth_id + i] < last_cost_min[depth_id])
      {
        last_cost_min[depth_id] = last_cost_min[depth_id + i];
      }
      __syncthreads();
    }

    float value = min(last_cost_min[0] + sgm_P2, last_cost[depth_id]);
    if(depth_id > 0)
      value = min(value, last_cost[depth_id - 1] + sgm_P1);
    if(depth_id < DEPTH_NUM - 1)
      value = min(value, last_cost[depth_id + 1] + sgm_P1);

    value = this_cost[depth_id] + value - last_cost_min[0];
    atomicAdd(my_add_ptr, value);
    __syncthreads();
    last_cost_min[depth_id] = last_cost[depth_id] = value;
  }
}

__global__ void sgm_cost_col_kernel(bool to_down, DeviceImage<PIXEL_COST> *cost_devptr, DeviceImage<PIXEL_COST> *sgm_cost_devptr)
{
  const int width = cost_devptr->width;
  const int height = cost_devptr->height;
  const int depth_id = threadIdx.x;

  int x = blockIdx.x;
  int y, delta_y;
  if(to_down)
  {
    y = 0;
    delta_y = 1;
  }
  else
  {
    y = height - 1;
    delta_y = -1;
  }

  __shared__ float last_cost[DEPTH_NUM], last_cost_min[DEPTH_NUM];;
  __shared__ float this_cost[DEPTH_NUM];

  last_cost_min[depth_id] = last_cost[depth_id] = 0.0;
  __syncthreads();

  for( ; y < height && y >= 0; y += delta_y)
  {
    float* my_add_ptr = (sgm_cost_devptr->atXY(x,y)).cost_ptr(depth_id);
    this_cost[depth_id] = (cost_devptr->atXY(x,y)).get_cost(depth_id);
    __syncthreads();
    for(int i = DEPTH_NUM/2; i > 0; i /= 2)
    {
      if(depth_id < i && last_cost_min[depth_id + i] < last_cost_min[depth_id])
      {
        last_cost_min[depth_id] = last_cost_min[depth_id + i];
      }
      __syncthreads();
    }

    float value = min(last_cost_min[0] + sgm_P2, last_cost[depth_id]);
    if(depth_id > 0)
      value = min(value, last_cost[depth_id - 1] + sgm_P1);
    if(depth_id < DEPTH_NUM - 1)
      value = min(value, last_cost[depth_id + 1] + sgm_P1);

    value = this_cost[depth_id] + value - last_cost_min[0];
    atomicAdd(my_add_ptr, value);
    __syncthreads();
    last_cost_min[depth_id] = last_cost[depth_id] = value;
  }
}

__global__ void sgm_filter(DeviceImage<PIXEL_COST> *sgm_cost_devptr, DeviceImage<float> *depth_devptr)
{
	 const int x = blockIdx.x;
  const int y = blockIdx.y;
  const int depth = threadIdx.x;

  __shared__ float cost[DEPTH_NUM];
  __shared__ float min_cost[DEPTH_NUM];
  __shared__ float min_index[DEPTH_NUM];
  min_cost[depth] = cost[depth] = (sgm_cost_devptr->atXY(x,y)).get_cost(depth);
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