#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define SIZE 512

// Функція складання двох векторів
__global__ void addVector(float* left, float* right, float* result) {
	// Отримати id поточної нитки.
	int idx = blockIdx.x;
	// Розраховуємо результат.
	result[idx] = left[idx] + right[idx];
}

__host__ int main() {

	// Виділяємо пам'ять під вектора
	float *vec1 = new float[SIZE], *vec2 = new float[SIZE], *vec3 = new float[SIZE];
	// Ініціалізіруем значення векторів
	for (int i = 0; i < SIZE; i++) vec1[i] = vec2[i] = i;

	// Покажчики на пам'ять відеокарти
	float *devVec1, *devVec2, *devVec3;
	// Виділяємо пам'ять для векторів на відеокарті
	cudaMalloc((void**)&devVec1, sizeof(float) * SIZE);
	cudaMalloc((void**)&devVec2, sizeof(float) * SIZE);
	cudaMalloc((void**)&devVec3, sizeof(float) * SIZE);
	// Копіюємо дані в пам'ять відеокарти
	cudaMemcpy(devVec1, vec1, sizeof(float) * SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(devVec2, vec2, sizeof(float) * SIZE, cudaMemcpyHostToDevice);

	// Розмір гріду, що використовується
	dim3 gridSize = dim3(SIZE, 1, 1);
	// Розмір блоку, що використовується
	dim3 blockSize = dim3(1, 1, 1);
	// Виконуємо виклик функції ядра
	addVector << <gridSize, blockSize >> >(devVec1, devVec2, devVec3);

	// Обробник події
	cudaEvent_t syncEvent;
	// Створюємо подію
	cudaEventCreate(&syncEvent);
	// Записуємо подію
	cudaEventRecord(syncEvent, 0);
	// Синхронізуємо подію
	cudaEventSynchronize(syncEvent);
	// Тільки тепер отримуємо результат розрахунку
	cudaMemcpy(vec3, devVec3, sizeof(float) * SIZE, cudaMemcpyDeviceToHost);

	// Результати розрахунку
	for (int i = 0; i < SIZE; i++) printf("Element #%i: %.1f\n", i, vec3[i]);
	// Вивільняються ресурсів
	// Видалення події
	cudaEventDestroy(syncEvent);
	// Вивільнення пам'яті на відеокарті
	cudaFree(devVec1);
	cudaFree(devVec2);
	cudaFree(devVec3);
	// Вивільнення памьяті основної програми
	delete[] vec1, vec2, vec3;

}