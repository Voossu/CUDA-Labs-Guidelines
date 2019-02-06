#include "cuda_runtime.h"
#include "stdio.h"

int main(int argc, char* argv[]) {

	int deviceCount;
	cudaDeviceProp deviceProp;

	cudaGetDeviceCount(&deviceCount);

	printf("Device count: %d\n\n", deviceCount);

	for (int i = 0; i < deviceCount; i++) {
		cudaGetDeviceProperties(&deviceProp, i);

		printf("Device %d name: %s\n", i + 1, deviceProp.name);
		printf("Total global memory: %zu\n", deviceProp.totalGlobalMem);
		printf("Shared memory per block: %zu\n", deviceProp.sharedMemPerBlock);
		printf("Registers per block: %d\n", deviceProp.regsPerBlock);
		printf("Warp size: %d\n", deviceProp.warpSize);
		printf("Memory pitch: %zu\n", deviceProp.memPitch);
		printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
		printf("Max threads dimensions: x = %d, y = %d, z = %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
		printf("Max grid size: x = %d, y = %d, z = %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
		printf("Clock rate: %d\n", deviceProp.clockRate);
		printf("Total constant memory: %zu\n", deviceProp.totalConstMem);
		printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
		printf("Texture alignment: %zu\n", deviceProp.textureAlignment);
		printf("Device overlap: %d\n", deviceProp.deviceOverlap);
		printf("Multiprocessor count: %d\n", deviceProp.multiProcessorCount);
		printf("Kernel execution timeout enabled: %s\n\n", deviceProp.kernelExecTimeoutEnabled ? "true" : "false");
	}

	return 0;
}
