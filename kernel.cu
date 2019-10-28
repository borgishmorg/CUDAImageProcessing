#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>
#include <algorithm>

using namespace std;
using uchar = unsigned char;

__global__ void kernel(uchar* data, uchar* new_data, unsigned height, unsigned width) {
	float matr[3][3] =
	{
		{0.11111f, 0.11111f, 0.11111f},
		{0.11111f, 0.11111f, 0.11111f},
		{0.11111f, 0.11111f, 0.11111f}
	};
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < height && j < width) {
		float resB = 0.f, resG = 0.f, resR = 0.f;
		for (int di : {-1, 0, 1}) {
			for (int dj : {-1, 0, 1}) {
				int ni = max(0, min(i + di, (int)height - 1));
				int nj = max(0, min(j + dj, (int)width - 1));
				resB += (char)(matr[1 + di][1 + dj] * data[(ni * width + nj) * 3]);
				resG += (char)(matr[1 + di][1 + dj] * data[(ni * width + nj) * 3 + 1]);
				resR += (char)(matr[1 + di][1 + dj] * data[(ni * width + nj) * 3 + 2]);
			}
		}
		new_data[(i * width + j) * 3] = (char) resB;
		new_data[(i * width + j) * 3 + 1] = (char) resG;
		new_data[(i * width + j) * 3 + 2] = (char) resR;
	}
}

int main(){
	ifstream in("picture.bmp", ios::in|ios::binary);
	ofstream out("new_picture.bmp", ofstream::binary);
	
	uchar *picture, *new_picture;

	cudaSetDeviceFlags(cudaDeviceMapHost);
	cudaHostAlloc(&picture, 50*1024*1024, cudaHostAllocMapped);
	cudaHostAlloc(&new_picture, 50*1024*1024, cudaHostAllocMapped);

	int len = 0;

	while (in.read((char*)picture+len, 1)) new_picture[len] = picture[len], len++;
	
	unsigned begin = *(unsigned*)(picture + 10);
	unsigned width = *(unsigned*)(picture + 18);
	unsigned height = *(unsigned*)(picture + 22);

	uchar* data = picture + begin;
	uchar* new_data = new_picture + begin;

	dim3 block(32, 32), numBlock((height+31)/32, (width+31)/32);

	kernel <<<numBlock, block>>> (data, new_data, height, width);

	cudaDeviceSynchronize();
	for (int i = 0; i < len; i++)
		out << new_picture[i];
	
    return 0;
}
