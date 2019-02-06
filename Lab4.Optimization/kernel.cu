#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <Windows.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

// Швидкість доступу і використання памяті

// Оптимальна память, низька швидкість
__device__ int3 dataNO[512];
__global__ void initDataNO()
{
	int idx = threadIdx.x;
	dataNO[idx] = make_int3(idx, idx, idx);
};

// Залишкова память, висока швидкість
__device__ int4 dataO[512];
__global__ void initDataO()
{
	int idx = threadIdx.x;
	dataO[idx] = make_int4(idx, idx, idx, 0);
};

// Оптимальна память, низька швидкість
struct vector3bg {
	float x;
	float y;
	float z;
};

// Залишкова память, висока швидкість
struct __align__(16) vector3sm {
	float x;
	float y;
	float z;
};





struct __align__(16) vec3 {
	float x;
	float y;
	float z;
};

__device__ vec3 data[512];

// не ефективна робота з памяттю
__global__ void initArrNO()
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	data[idx].x = idx;
	data[idx].y = idx * 2;
	data[idx].z = idx * 3;
};

__device__ float x[512];
__device__ float y[512];
__device__ float z[512];

// ефективна робота з памяттю
__global__ void initArrO() {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	x[idx] = idx;
	y[idx] = idx * 2;
	z[idx] = idx * 3;
};


int main() {
	printf("%i\n", sizeof(vector3bg));
	printf("%i\n", sizeof(vector3sm));
	initDataNO << <1, 512 >> >();
	initDataO << <1, 512 >> >();
	return 0;
};