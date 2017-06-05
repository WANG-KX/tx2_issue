
#ifndef DEVICE_IMAGE_CUH
#define DEVICE_IMAGE_CUH

#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_exception.cuh>

template<typename ElementType>
struct DeviceImage
{
  __host__
  DeviceImage(size_t width, size_t height)
    : width(width),
      height(height)
  {
    cudaError err = cudaMalloc(
          &data,
          height*width*sizeof(ElementType));
    if(err != cudaSuccess)
      throw CudaException("DeviceImage: unable to allocate pitched memory.", err);

    err = cudaMalloc(
          &dev_ptr,
          sizeof(*this));
    if(err != cudaSuccess)
      throw CudaException("DeviceImage: cannot allocate device memory to store image parameters.", err);

    err = cudaMemcpy(
          dev_ptr,
          this,
          sizeof(*this),
          cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
      throw CudaException("DeviceImage: cannot copy image parameters to device memory.", err);
  }

  __device__
  ElementType & operator()(size_t x, size_t y)
  {
    return atXY(x, y);
  }

  __device__
  const ElementType & operator()(size_t x, size_t y) const
  {
    return atXY(x, y);
  }

  __device__
  ElementType & atXY(size_t x, size_t y)
  {
    return data[y*width+x];
  }

  __device__
  const ElementType & atXY(size_t x, size_t y) const
  {
    return data[y*width+x];
  }

  /// Upload aligned_data_row_major to device memory
  __host__
  void setDevData(const ElementType * aligned_data_row_major)
  {
    const cudaError err = cudaMemcpy(
          data,
          aligned_data_row_major,
          height*width*sizeof(ElementType),
          cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
      throw CudaException("Image: unable to copy data from host to device.", err);
  }

  /// Download the data from the device memory to aligned_data_row_major, a preallocated array in host memory
  __host__
  void getDevData(ElementType* aligned_data_row_major) const
  {

    const cudaError err = cudaMemcpy(
          aligned_data_row_major,
          data,
          height*width*sizeof(ElementType),
          cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
    {
      throw CudaException("Image: unable to copy data from device to host.", err);
    }
  }

  __host__
  ~DeviceImage()
  {
    cudaError err = cudaFree(data);
    if(err != cudaSuccess)
      throw CudaException("Image: unable to free allocated memory.", err);
    err = cudaFree(dev_ptr);
    if(err != cudaSuccess)
      throw CudaException("Image: unable to free allocated memory.", err);
  }

  __host__
  cudaChannelFormatDesc getCudaChannelFormatDesc() const
  {
    return cudaCreateChannelDesc<ElementType>();
  }

  __host__
  void zero()
  {
    const cudaError err = cudaMemset(
          data,
          0,
          height*width*sizeof(ElementType));
    if(err != cudaSuccess)
      throw CudaException("Image: unable to zero.", err);
  }

  __host__
  DeviceImage<ElementType> & operator=(const DeviceImage<ElementType> &other_image)
  {
    if(this != &other_image)
    {
      assert(width  == other_image.width &&
             height == other_image.height);
      const cudaError err = cudaMemcpy(
          data,
          other_image.data,
          height*width*sizeof(ElementType),
          cudaMemcpyDeviceToDevice);
      if(err != cudaSuccess)
        throw CudaException("Image, operator '=': unable to copy data from another image.", err);
    }
    return *this;
  }

  // fields
  size_t width;
  size_t height;
  ElementType * data;
  DeviceImage<ElementType> *dev_ptr;
};

#endif // DEVICE_IMAGE_CUH
