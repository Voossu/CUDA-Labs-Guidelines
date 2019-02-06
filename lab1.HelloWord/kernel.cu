#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cuda.h>

__global__ void kernel() {
	printf("I am from %d block, %d thread (global index: %d)\n", blockIdx.x, threadIdx.x, blockIdx.x * blockDim.x + threadIdx.x);
}

int main(int agrc, char* argv[]) {
	printf("Now cuda say:\n");
	kernel<<<2,2>>>();
	cudaDeviceSynchronize();
	return 0;
}
