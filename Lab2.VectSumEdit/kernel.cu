#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <Windows.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define BLOCK 512

#include <iostream>

// ������� ��������� ���� �������
__global__ void addVector(float* left, float* right, float* result, int size) {
	// �������� id ������� �����.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// ����������� ���������.
	if (idx < size) result[idx] = left[idx] + right[idx];
}

__host__ int main(int argc, const char *argv[]) {

	if (argc != 3 || strcmp("-r", argv[1]) != 0) {
		printf(" Available command line options:\n");
		printf("   -r <vector length> - generation of random vectors\n");
		printf("   -h - help information\n");
		return !(argc == 2 && strcmp("-h", argv[1]));
	}
	
	int vec_size = atoi(argv[2]);
	if (vec_size < 1) {
		printf(" Invalid vector size.\n");
		return 1;
	}

	// �������� ���'��� �� �������
	float *vec1 = new float[vec_size], *vec2 = new float[vec_size], *vec3 = new float[vec_size];
	// ����������� �������� �������
	srand(time(NULL));
	for (int i = 0; i < vec_size; i++) {
		vec1[i] = rand();
		vec2[i] = rand();
	}

	// ��������� �� ���'��� ���������
	float *devVec1, *devVec2, *devVec3;

	// �������� ���'��� ��� ������� �� ��������
	cudaMalloc((void**)&devVec1, sizeof(float) * vec_size);
	cudaMalloc((void**)&devVec2, sizeof(float) * vec_size);
	cudaMalloc((void**)&devVec3, sizeof(float) * vec_size);

	double total_time = GetTickCount();

	// ������� ��� � ���'��� ���������
	cudaMemcpy(devVec1, vec1, sizeof(float) * vec_size, cudaMemcpyHostToDevice);
	cudaMemcpy(devVec2, vec2, sizeof(float) * vec_size, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	// ��������� ��䳿 ��� ������������ � ����� ���� ������ GPU
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	//�������� ����� �������� �� GPU
	cudaEventRecord(start, 0);

	// �������� ������ ������� ����
	addVector<<<BLOCK,(int)(vec_size/BLOCK)+1>>>(devVec1, devVec2, devVec3, vec_size);

	cudaEventRecord(stop, 0);
	// �������������� � �������� ��������� ����������
	cudaEventSynchronize(stop);
	// ����������� ��� ������ GPU
	float gpu_time = 0;
	cudaEventElapsedTime(&gpu_time, start, stop);
	
	// ҳ���� ����� �������� ��������� ����������
	cudaMemcpy(vec3, devVec3, sizeof(float) * vec_size, cudaMemcpyDeviceToHost);

	total_time = GetTickCount() - total_time;

	// ���������� ����������
	for (int i = 0; i < vec_size; i++) printf("Element #%i: %.2f + %.2f = %.1f\n", i, vec1[i], vec2[i], vec3[i]);
	//
	printf("GPU compute time: %.10f\n", gpu_time);
	printf("Total time: %.10f\n", total_time);

	// ������������ �������
	// ��������� ��䳿
	cudaEventDestroy(stop);
	// ���������� ���'�� �� ��������
	cudaFree(devVec1);
	cudaFree(devVec2);
	cudaFree(devVec3);
	// ���������� ������ ������� ��������
	delete[] vec1, vec2, vec3;

}