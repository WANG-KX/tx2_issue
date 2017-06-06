#include <cstdlib>
#include <deque>
#include <sys/time.h>
#include <iostream>

#include <cuda_runtime.h>
#include <device_image.cuh>
#include <pixel_cost.cuh>

__global__ void wta_kernel(DeviceImage<PIXEL_COST> *cost_devptr, DeviceImage<float> *depth_devptr);
void wta(DeviceImage<PIXEL_COST> &cost, DeviceImage<float> &depth);