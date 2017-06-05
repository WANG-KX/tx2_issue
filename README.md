This code performs badly on TX2 for reason we do not know.
to run the code, just:

git clone https://github.com/WANG-KX/tx2_issue.git


cd tx2_issue

mkdir build

cd build

cmake ..

make

./sgm

the performance will print out and the result will show on the screen.

All the CmakeLists are the same except the arch code in line 15, in which we set TITAN XP:61 TX2: 62, TX1: 53.
In TX2, before we run the code, we already set: nvpmodel -m 0 and ./jetson_clocks.sh as discribed in https://devtalk.nvidia.com/default/topic/1011551/jetson-tx2/cuda-performance-issue-on-tx2/

Below is the performance for reference:

TITAN XP
the input image size: 1344 x 391.
wta cost 4.095000 ms.
sgm cost: 37.640000 ms.

TX2
the input image size: 1344 x 391.
wta cost 41.571000 ms.
sgm cost: 514.919000 ms.

TX1
the input image size: 1344 x 391.
wta cost 18.380000 ms.
sgm cost: 287.463000 ms.

