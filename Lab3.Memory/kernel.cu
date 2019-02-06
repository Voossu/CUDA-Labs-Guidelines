#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <Windows.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

// ���������� ����� ����������
#define CUDA_DEBUG
// ��������� ����������� ����������
#ifdef CUDA_DEBUG
#define CUDA_CHECK_ERROR(err) \
 if (err != cudaSuccess) { \
  printf("Cuda error: %s\n", cudaGetErrorString(err)); \
  printf("Error in file: %s, line: %i\n", __FILE__, __LINE__); \
 }
#else
#define CUDA_CHECK_ERROR(err)
#endif

// ������� �������������� ������� ��� ������������ ��������� ���'��
// * inputMatrix - �������� �� ������� �������
// * outputMatrix - �������� �� ������� ���������
// * width - ������ ������� ������� (���� � ������ �������-����������)
// * height - ������ ������� ������� (���� � ������ �������-����������)
__global__ void transposeMatrixGlobal(float* inputMatrix, float* outputMatrix, int width, int height) {
	// ���������� ������� �������
	int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	if ((xIndex < width) && (yIndex < height)) {
		// ˳����� ������ �������� ����� ������� ������� 
		int inputIdx = xIndex + width * yIndex;
		// ˳����� ������ �������� ������� �������-����������
		int outputIdx = yIndex + height * xIndex;
		// ������������ �������� 
		outputMatrix[outputIdx] = inputMatrix[inputIdx];
	}
}

#define BLOCK_DIM 16

#define BLOCK_DIM 16
// ������� �������������� ������� � ������������� ���������� ���'��
// * inputMatrix - �������� �� ������� �������
// * outputMatrix - �������� �� ������� ���������
// * width - ������ ������� ������� (���� � ������ �������-����������)
// * height - ������ ������� ������� (���� � ������ �������-����������)
__global__ void transposeMatrixShared(float* inputMatrix, float* outputMatrix, int width, int height) {
	__shared__ float temp[BLOCK_DIM][BLOCK_DIM];
	// ���������� ������� �������
	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	if ((xIndex < width) && (yIndex < height)) {
		// �������� ������ �������� ������ �������� �������  
		int idx = yIndex * width + xIndex;
		//�������� �������� �������� �������
		temp[threadIdx.y][threadIdx.x] = inputMatrix[idx];
	}
	//�������������� ��� ���� � �����
	__syncthreads();
	xIndex = blockIdx.y * blockDim.y + threadIdx.x;
	yIndex = blockIdx.x * blockDim.x + threadIdx.y;
	if ((xIndex < height) && (yIndex < width)) {
		// �������� ������ �������� ������ �������� �������  
		int idx = yIndex * height + xIndex;
		//�������� �������� �������� �������
		outputMatrix[idx] = temp[threadIdx.x][threadIdx.y];
	}
}

// ������� �������������� �������, ��� ���������� �� CPU
__host__ void transposeMatrixCPU(float *inputMatrix, float *outputMatrix, int width, int height) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			outputMatrix[x * height + y] = inputMatrix[y * width + x];
		}
	}
}

// ������� ��������� ������� �� �����
__host__ void printMatrixToFile(char* fileName, float* matrix, int width, int height) {
	FILE *file = fopen(fileName, "wt");
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			fprintf(file, "%.0f\t", matrix[y * width + x]);
		}
		fprintf(file, "\n");
	}
	fclose(file);
}

#define GPU_SLOW 1
#define GPU_FAST 2
#define CPU 3

// ʳ������ ���������������� �����
#define ITERATIONS 20

__host__ int main() {

 // ������ � ������ �������
 int width = 2048, height = 1536;
 // ����� ������ ��� ���������� �������
 int matrixSize = width * height;
 // ʳ������ ����� �� ����� �������
 int byteSize = matrixSize * sizeof(float);

 //�������� ������ ��� ������� �� �����
 float* inputMatrix = new float[matrixSize];
 float* outputMatrix = new float[matrixSize];

 //��������� �������� ������� �������
 for (int i = 0; i < matrixSize; i++)
  inputMatrix[i] = i;

 // �������� ����� ���������� �������������� �������
 printf("Select compute mode: 1 - Slow GPU, 2 - Fast GPU, 3 - CPU\n");
 int mode;
 scanf("%i", &mode);

 // �������� ������� ������� � ����
 printMatrixToFile("before.txt", inputMatrix, width, height);

 // ���� ������������� ����� CPU
 if (mode == CPU) {
  int start = GetTickCount();
  for (int i = 0; i < ITERATIONS; i++) {
   transposeMatrixCPU(inputMatrix, outputMatrix, width, height);
  }
  // �������� ��� ��������� ������� �� CPU (� ����������)
  printf("CPU compute time: %i\n", GetTickCount() - start);
 }
 // � ��� ���������� �� GPU
 else {
  float *devInputMatrix, *devOutputMatrix;
  // �������� ��������� ���'��� ��� ��������� ����� �� �������
  CUDA_CHECK_ERROR(cudaMalloc((void**)&devInputMatrix, byteSize));
  CUDA_CHECK_ERROR(cudaMalloc((void**)&devOutputMatrix, byteSize));
  // ������� ������� ������� � ����� �� ������
  CUDA_CHECK_ERROR(cudaMemcpy(devInputMatrix, inputMatrix, byteSize, cudaMemcpyHostToDevice));
  // ������������ ������� ����
  dim3 gridSize = dim3(width / BLOCK_DIM, height / BLOCK_DIM, 1);
  dim3 blockSize = dim3(BLOCK_DIM, BLOCK_DIM, 1);

  cudaEvent_t start, stop;
  // ��������� ��䳿 ��� ������������ � ����� ���� ������ GPU
  CUDA_CHECK_ERROR(cudaEventCreate(&start));
  CUDA_CHECK_ERROR(cudaEventCreate(&stop));
  //�������� ����� �������� �� GPU
  cudaEventRecord(start, 0);
  // ��������������� ������� ��� �������� ���'��
  if (mode == GPU_SLOW) {
   for (int i = 0; i < ITERATIONS; i++) {
    transposeMatrixGlobal<<<gridSize, blockSize>>>(devInputMatrix, devOutputMatrix, width, height);
   }
  }
  // ��������������� ������� � ��������� ���'����
  else if (mode == GPU_FAST) {
   for (int i = 0; i < ITERATIONS; i++) {
    transposeMatrixShared<<<gridSize, blockSize>>>(devInputMatrix, devOutputMatrix, width, height);
   }
  }
  // ³�������� ��������� ����������
  cudaEventRecord(stop, 0);
  // �������������� � �������� ��������� ����������
  cudaEventSynchronize(stop);
  // ����������� ��� ������ GPU
  float time = 0;
  cudaEventElapsedTime(&time, start, stop);
  
  // �������� ��� ���������� � �������
  printf("GPU compute time: %.0f\n", time);

  // ������� ��������� � ������� �� ����
  CUDA_CHECK_ERROR(cudaMemcpy(outputMatrix, devOutputMatrix, byteSize, cudaMemcpyDeviceToHost));

  // ������� ������� �� ��������
  CUDA_CHECK_ERROR(cudaFree(devInputMatrix));
  CUDA_CHECK_ERROR(cudaFree(devOutputMatrix));
  CUDA_CHECK_ERROR(cudaEventDestroy(start));
  CUDA_CHECK_ERROR(cudaEventDestroy(stop));
 }

 // �������� �������-��������� � ����
 printMatrixToFile("after.txt", outputMatrix, height, width);

 // ������� ���'��� �� ����
 delete[] inputMatrix, outputMatrix;

 return 0;
} 
