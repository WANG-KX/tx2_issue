#include <cstdlib>
#include <deque>
#include <ctime>
#include <iostream>

#include <cuda_runtime.h>
#include <device_image.cuh>
#include <pixel_cost.cuh>

#define sgm_P1 1.0
#define sgm_P2 4.0

__global__ void sgm_cost_row_kernel(bool to_left, DeviceImage<PIXEL_COST> *cost_devptr, DeviceImage<PIXEL_COST> *sgm_cost_devptr);
__global__ void sgm_cost_col_kernel(bool to_down, DeviceImage<PIXEL_COST> *cost_devptr, DeviceImage<PIXEL_COST> *sgm_cost_devptr);
__global__ void sgm_filter(DeviceImage<PIXEL_COST> *sgm_cost_devptr, DeviceImage<float> *depth_devptr);
void sgm(DeviceImage<PIXEL_COST> &cost, DeviceImage<float> &depth);