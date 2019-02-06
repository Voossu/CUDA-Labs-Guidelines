#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <Windows.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define BLOCK 512

#include <iostream>

// Функція складання двох векторів
__global__ void addVector(float* left, float* right, float* result, int size) {
	// Отримати id поточної нитки.
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// Розраховуємо результат.
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

	// Виділяємо пам'ять під вектора
	float *vec1 = new float[vec_size], *vec2 = new float[vec_size], *vec3 = new float[vec_size];
	// Ініціалізіруем значення векторів
	srand(time(NULL));
	for (int i = 0; i < vec_size; i++) {
		vec1[i] = rand();
		vec2[i] = rand();
	}

	// Покажчики на пам'ять відеокарти
	float *devVec1, *devVec2, *devVec3;

	// Виділяємо пам'ять для векторів на відеокарті
	cudaMalloc((void**)&devVec1, sizeof(float) * vec_size);
	cudaMalloc((void**)&devVec2, sizeof(float) * vec_size);
	cudaMalloc((void**)&devVec3, sizeof(float) * vec_size);

	double total_time = GetTickCount();

	// Копіюємо дані в пам'ять відеокарти
	cudaMemcpy(devVec1, vec1, sizeof(float) * vec_size, cudaMemcpyHostToDevice);
	cudaMemcpy(devVec2, vec2, sizeof(float) * vec_size, cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	// Створюємо події для синхронізації і виміру часу роботи GPU
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	//Отмечаем старт расчетов на GPU
	cudaEventRecord(start, 0);

	// Виконуємо виклик функції ядра
	addVector<<<BLOCK,(int)(vec_size/BLOCK)+1>>>(devVec1, devVec2, devVec3, vec_size);

	cudaEventRecord(stop, 0);
	// Синхронізуються з моментом закінчення розрахунків
	cudaEventSynchronize(stop);
	// Розраховуємо час роботи GPU
	float gpu_time = 0;
	cudaEventElapsedTime(&gpu_time, start, stop);
	
	// Тільки тепер отримуємо результат розрахунку
	cudaMemcpy(vec3, devVec3, sizeof(float) * vec_size, cudaMemcpyDeviceToHost);

	total_time = GetTickCount() - total_time;

	// Результати розрахунку
	for (int i = 0; i < vec_size; i++) printf("Element #%i: %.2f + %.2f = %.1f\n", i, vec1[i], vec2[i], vec3[i]);
	//
	printf("GPU compute time: %.10f\n", gpu_time);
	printf("Total time: %.10f\n", total_time);

	// Вивільняються ресурсів
	// Видалення події
	cudaEventDestroy(stop);
	// Вивільнення пам'яті на відеокарті
	cudaFree(devVec1);
	cudaFree(devVec2);
	cudaFree(devVec3);
	// Вивільнення памьяті основної програми
	delete[] vec1, vec2, vec3;

}