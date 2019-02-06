#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <Windows.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

// Визначення змінної середовища
#define CUDA_DEBUG
// Виведення діагностичної інформації
#ifdef CUDA_DEBUG
#define CUDA_CHECK_ERROR(err) \
 if (err != cudaSuccess) { \
  printf("Cuda error: %s\n", cudaGetErrorString(err)); \
  printf("Error in file: %s, line: %i\n", __FILE__, __LINE__); \
 }
#else
#define CUDA_CHECK_ERROR(err)
#endif

// Функція транспонування матриці без використання глобальної пам'яті
// * inputMatrix - покажчик на вихідну матрицю
// * outputMatrix - покажчик на матрицю результат
// * width - ширина вихідної матриці (вона ж висота матриці-результату)
// * height - висота вихідної матриці (вона ж ширина матриці-результату)
__global__ void transposeMatrixGlobal(float* inputMatrix, float* outputMatrix, int width, int height) {
	// Розрахунок індексів матриці
	int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
	int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
	if ((xIndex < width) && (yIndex < height)) {
		// Лінійний індекс елемента рядки вихідної матриці 
		int inputIdx = xIndex + width * yIndex;
		// Лінійний індекс елемента стовпця матриці-результату
		int outputIdx = yIndex + height * xIndex;
		// Встановлення елементу 
		outputMatrix[outputIdx] = inputMatrix[inputIdx];
	}
}

#define BLOCK_DIM 16

#define BLOCK_DIM 16
// Функція транспонування матриці з використанням колективної пам'яті
// * inputMatrix - покажчик на вихідну матрицю
// * outputMatrix - покажчик на матрицю результат
// * width - ширина вихідної матриці (вона ж висота матриці-результату)
// * height - висота вихідної матриці (вона ж ширина матриці-результату)
__global__ void transposeMatrixShared(float* inputMatrix, float* outputMatrix, int width, int height) {
	__shared__ float temp[BLOCK_DIM][BLOCK_DIM];
	// Розрахунок індексів матриці
	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	if ((xIndex < width) && (yIndex < height)) {
		// Линейный индекс элемента строки исходной матрицы  
		int idx = yIndex * width + xIndex;
		//Копируем элементы исходной матрицы
		temp[threadIdx.y][threadIdx.x] = inputMatrix[idx];
	}
	//Синхронизируем все нити в блоке
	__syncthreads();
	xIndex = blockIdx.y * blockDim.y + threadIdx.x;
	yIndex = blockIdx.x * blockDim.x + threadIdx.y;
	if ((xIndex < height) && (yIndex < width)) {
		// Линейный индекс элемента строки исходной матрицы  
		int idx = yIndex * height + xIndex;
		//Копируем элементы исходной матрицы
		outputMatrix[idx] = temp[threadIdx.x][threadIdx.y];
	}
}

// Функція транспонування матриці, яка виконується на CPU
__host__ void transposeMatrixCPU(float *inputMatrix, float *outputMatrix, int width, int height) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			outputMatrix[x * height + y] = inputMatrix[y * width + x];
		}
	}
}

// Функція виведення матриці на екран
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

// Кількість навантажувальних циклів
#define ITERATIONS 20

__host__ int main() {

 // Ширина і висота матриці
 int width = 2048, height = 1536;
 // Розмір масиву для збереження матриці
 int matrixSize = width * height;
 // Кількість байтів що займає матриця
 int byteSize = matrixSize * sizeof(float);

 //Выделяем память под матрицы на хосте
 float* inputMatrix = new float[matrixSize];
 float* outputMatrix = new float[matrixSize];

 //Заполняем исходную матрицу данными
 for (int i = 0; i < matrixSize; i++)
  inputMatrix[i] = i;

 // Вибираємо спосіб розрахунку транспонованою матриці
 printf("Select compute mode: 1 - Slow GPU, 2 - Fast GPU, 3 - CPU\n");
 int mode;
 scanf("%i", &mode);

 // Записуємо вихідну матрицю в файл
 printMatrixToFile("before.txt", inputMatrix, width, height);

 // Якщо используеться тільки CPU
 if (mode == CPU) {
  int start = GetTickCount();
  for (int i = 0; i < ITERATIONS; i++) {
   transposeMatrixCPU(inputMatrix, outputMatrix, width, height);
  }
  // Виводимо час виконання функції на CPU (в мілліекундах)
  printf("CPU compute time: %i\n", GetTickCount() - start);
 }
 // У разі розрахунку на GPU
 else {
  float *devInputMatrix, *devOutputMatrix;
  // Виділяємо глобальну пам'ять для зберігання даних на пристрої
  CUDA_CHECK_ERROR(cudaMalloc((void**)&devInputMatrix, byteSize));
  CUDA_CHECK_ERROR(cudaMalloc((void**)&devOutputMatrix, byteSize));
  // Копіюємо вихідну матрицю з хоста на девайс
  CUDA_CHECK_ERROR(cudaMemcpy(devInputMatrix, inputMatrix, byteSize, cudaMemcpyHostToDevice));
  // Конфігурація запуску ядра
  dim3 gridSize = dim3(width / BLOCK_DIM, height / BLOCK_DIM, 1);
  dim3 blockSize = dim3(BLOCK_DIM, BLOCK_DIM, 1);

  cudaEvent_t start, stop;
  // Створюємо події для синхронізації і виміру часу роботи GPU
  CUDA_CHECK_ERROR(cudaEventCreate(&start));
  CUDA_CHECK_ERROR(cudaEventCreate(&stop));
  //Отмечаем старт расчетов на GPU
  cudaEventRecord(start, 0);
  // Використовується функція без суспільної пам'яті
  if (mode == GPU_SLOW) {
   for (int i = 0; i < ITERATIONS; i++) {
    transposeMatrixGlobal<<<gridSize, blockSize>>>(devInputMatrix, devOutputMatrix, width, height);
   }
  }
  // Використовується функція з суспільною пам'яттю
  else if (mode == GPU_FAST) {
   for (int i = 0; i < ITERATIONS; i++) {
    transposeMatrixShared<<<gridSize, blockSize>>>(devInputMatrix, devOutputMatrix, width, height);
   }
  }
  // Відзначаємо закінчення розрахунку
  cudaEventRecord(stop, 0);
  // Синхронізуються з моментом закінчення розрахунків
  cudaEventSynchronize(stop);
  // Розраховуємо час роботи GPU
  float time = 0;
  cudaEventElapsedTime(&time, start, stop);
  
  // Виводимо час розрахунку в консоль
  printf("GPU compute time: %.0f\n", time);

  // Копіюємо результат з девайса на хост
  CUDA_CHECK_ERROR(cudaMemcpy(outputMatrix, devOutputMatrix, byteSize, cudaMemcpyDeviceToHost));

  // Чистимо ресурси на відеокарті
  CUDA_CHECK_ERROR(cudaFree(devInputMatrix));
  CUDA_CHECK_ERROR(cudaFree(devOutputMatrix));
  CUDA_CHECK_ERROR(cudaEventDestroy(start));
  CUDA_CHECK_ERROR(cudaEventDestroy(stop));
 }

 // Записуємо матрицю-результат в файл
 printMatrixToFile("after.txt", outputMatrix, height, width);

 // Чистимо пам'ять на хості
 delete[] inputMatrix, outputMatrix;

 return 0;
} 
