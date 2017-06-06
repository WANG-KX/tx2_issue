set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
cmake_minimum_required(VERSION 2.4)
project(sgm)
find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )

include_directories(
  src
)

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -lineinfo
    #-gencode=arch=compute_61,code=sm_61 #for TITAN XP
    -gencode=arch=compute_62,code=sm_62 #for tx2
    #-gencode=arch=compute_53,code=sm_53 #for tx1
    )

cuda_add_executable(
    sgm
    src/main.cu src/wta/wta.cu src/sgm/sgm.cu )

target_link_libraries(sgm ${OpenCV_LIBS})