#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define SIZE 512

// ������� ��������� ���� �������
__global__ void addVector(float* left, float* right, float* result) {
	// �������� id ������� �����.
	int idx = blockIdx.x;
	// ����������� ���������.
	result[idx] = left[idx] + right[idx];
}

__host__ int main() {

	// �������� ���'��� �� �������
	float *vec1 = new float[SIZE], *vec2 = new float[SIZE], *vec3 = new float[SIZE];
	// ����������� �������� �������
	for (int i = 0; i < SIZE; i++) vec1[i] = vec2[i] = i;

	// ��������� �� ���'��� ���������
	float *devVec1, *devVec2, *devVec3;
	// �������� ���'��� ��� ������� �� ��������
	cudaMalloc((void**)&devVec1, sizeof(float) * SIZE);
	cudaMalloc((void**)&devVec2, sizeof(float) * SIZE);
	cudaMalloc((void**)&devVec3, sizeof(float) * SIZE);
	// ������� ��� � ���'��� ���������
	cudaMemcpy(devVec1, vec1, sizeof(float) * SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(devVec2, vec2, sizeof(float) * SIZE, cudaMemcpyHostToDevice);

	// ����� ����, �� ���������������
	dim3 gridSize = dim3(SIZE, 1, 1);
	// ����� �����, �� ���������������
	dim3 blockSize = dim3(1, 1, 1);
	// �������� ������ ������� ����
	addVector << <gridSize, blockSize >> >(devVec1, devVec2, devVec3);

	// �������� ��䳿
	cudaEvent_t syncEvent;
	// ��������� ����
	cudaEventCreate(&syncEvent);
	// �������� ����
	cudaEventRecord(syncEvent, 0);
	// ����������� ����
	cudaEventSynchronize(syncEvent);
	// ҳ���� ����� �������� ��������� ����������
	cudaMemcpy(vec3, devVec3, sizeof(float) * SIZE, cudaMemcpyDeviceToHost);

	// ���������� ����������
	for (int i = 0; i < SIZE; i++) printf("Element #%i: %.1f\n", i, vec3[i]);
	// ������������ �������
	// ��������� ��䳿
	cudaEventDestroy(syncEvent);
	// ���������� ���'�� �� ��������
	cudaFree(devVec1);
	cudaFree(devVec2);
	cudaFree(devVec3);
	// ���������� ������ ������� ��������
	delete[] vec1, vec2, vec3;

}