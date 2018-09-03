#include "common.h"
#include <cuda_runtime.h>
#include <chrono>
#include <pthread.h>

#define THREADS 8

using namespace std;

typedef struct thread_data_struct {
  int start;
  int end;
  int rows;
  int cols;
  int *matrixA;
  int *matrixB;
  long *result;
} thread_data_t;

int checkResult(long *hostRef, long *gpuRef, const int rows, const int cols){
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < rows * cols; i++){
        if (abs(hostRef[i] - gpuRef[i]) > epsilon){
            match = 0;
            printf("\nhost: %li | gpu: %li", hostRef[i], gpuRef[i]);
            break;
        }
    }
    return match;
}

void initializeMatrix(int *matrix, const int rows, const int cols){
  for (size_t i = 0; i < rows * cols; i++)
    matrix[i] = i;
}

void multiplyMatricesOnCPU(int *matrixA, int *matrixB, long *result, const int rows, const int cols){
  long sum;
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      sum = 0;
      for (size_t k = 0; k < cols; k++) {
        sum += matrixA[i * rows + k] * matrixB[k * cols + j];
      }
      result[i * rows + j] = sum;
    }
  }
}

void *multiplyMatricesWithThreads(void *args){
  thread_data_t *data = (thread_data_t *) args;
  long sum;
  for (size_t i = data->start; i < data->end; i++) {
    for (size_t j = 0; j < data->cols; j++) {
      sum = 0;
      for (size_t k = 0; k < data->cols; k++) {
        sum += data->matrixA[i * data->rows + k] * data->matrixB[k * data->cols + j];
      }
      data->result[i * data->rows + j] = sum;
    }
  }
  pthread_exit(NULL);
}

__global__ void multiplyMatricesOnGPU(int *matrixA, int *matrixB, long *result, const int rows, const int cols){
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = blockIdx.y;
  if(ix < rows && iy < cols){
    long sum = 0;
    for (size_t i = 0; i < cols; i++) {
      sum += matrixA[iy * rows + i] * matrixB[i * cols + ix];
    }
    result[iy * rows + ix] = sum;
  }
}

int main(int argc, char const *argv[]) {
  // set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev), "Error device prop");
  printf("Using Device %d: %s\n", dev, deviceProp.name);
  SAFE_CALL(cudaSetDevice(dev), "Error setting device");

  // Declare matrices
  int *matrixA;
  int *matrixB;
  long *cpu_ref;
  int *dev_matrixA;
  int *dev_matrixB;
  long *dev_result;
  long *gpu_ref;
  long *threads_result;

  // Declare threads and thread data variables
  pthread_t threads[THREADS];
  thread_data_t data[THREADS];

  // Set up size of matrix
  const int rows = 1000;
  const int cols = 1000;
  printf("Matrix size: rows %d columns %d\n", rows, cols);

  int bytes = rows * cols * sizeof(int);
  int longBytes = rows * cols * sizeof(long);

  // Allocate matrices memory
  matrixA = (int *) malloc(bytes);
  matrixB = (int *) malloc(bytes);
  cpu_ref = (long *) malloc(longBytes);
  gpu_ref = (long *) malloc(longBytes);
  threads_result = (long *) malloc(longBytes);

  // Initialize matrices
  initializeMatrix(matrixA, rows, cols);
  initializeMatrix(matrixB, rows, cols);

  // malloc device global memory
  SAFE_CALL(cudaMalloc((void **)&dev_matrixA, bytes), "Error allocating dev_matrixA");
  SAFE_CALL(cudaMalloc((void **)&dev_matrixB, bytes), "Error allocating dev_matrixB");
  SAFE_CALL(cudaMalloc((void **)&dev_result, longBytes), "Error allocating dev_result");

  // transfer data from host to device
  SAFE_CALL(cudaMemcpy(dev_matrixA, matrixA, bytes, cudaMemcpyHostToDevice), "Error copying dev_matrixA");
  SAFE_CALL(cudaMemcpy(dev_matrixB, matrixB, bytes, cudaMemcpyHostToDevice), "Error copying dev_matrixB");

  // Mutiply matrices on host
  auto start_at = std::chrono::high_resolution_clock::now();
  multiplyMatricesOnCPU(matrixA, matrixB, cpu_ref, rows, cols);
  auto end_at = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> duration_ms = end_at - start_at;
  printf("Multiply matrix on CPU elapsed: %f ms (%.2f seconds)\n", duration_ms.count(), duration_ms.count() / 1000);


  // Initialize threads data
  int step = (int)(rows / THREADS);
  for (size_t i = 0; i < THREADS; i++) {
    data[i].start = step * i;
    data[i].end = step * (i + 1);
    data[i].rows = rows;
    data[i].cols = cols;
    data[i].matrixA = matrixA;
    data[i].matrixB = matrixB;
    data[i].result = threads_result;
  }

  // Multiply matrices on host with threads
  start_at =  std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < THREADS; i++) {
    pthread_create(&threads[i], NULL, multiplyMatricesWithThreads, (void *) &data[i]);
  }
  for (size_t i = 0; i < THREADS; i++) {
    pthread_join(threads[i], NULL);
  }
  end_at =  std::chrono::high_resolution_clock::now();
  duration_ms = end_at - start_at;
  printf("Multiply matrix on host with threads (%d threads) elapsed: %f ms (%.2f seconds)\n", THREADS, duration_ms.count(), duration_ms.count() / 1000);

  // Invoke kernel at host side
  int dimx = 512;
  dim3 block(dimx, 1);
  dim3 grid((rows + block.x - 1) / block.x, cols);

  start_at = chrono::high_resolution_clock::now();
  multiplyMatricesOnGPU<<<grid, block>>>(dev_matrixA, dev_matrixB, dev_result, rows, cols);
  SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
  end_at = chrono::high_resolution_clock::now();
  duration_ms = end_at - start_at;

  printf("Multiply matrices on GPU <<<(%d,%d), (%d,%d)>>> elapsed: %f ms (%.2f seconds)\n",
        grid.x, grid.y, block.x, block.y, duration_ms.count(), duration_ms.count() / 1000);

  // SAFE_CALL kernel error
  SAFE_CALL(cudaGetLastError(), "Error with last error");

  // Copy kernel result back to host side
  SAFE_CALL(cudaMemcpy(gpu_ref, dev_result, longBytes, cudaMemcpyDeviceToHost), "Error copying dev_result");

  // Check device results
  if(checkResult(cpu_ref, gpu_ref, rows, cols) && checkResult(threads_result, gpu_ref, rows, cols)){
    printf("\nArrays match!\n\n");
  } else {
    printf("\nArrays DONT match!\n\n");
  }

  // Free device global memory
  SAFE_CALL(cudaFree(dev_matrixA), "Error freeing memory");
  SAFE_CALL(cudaFree(dev_matrixB), "Error freeing memory");
  SAFE_CALL(cudaFree(dev_result), "Error freeing memory");

  // Free matrices memory
  free(matrixA);
  free(matrixB);
  free(gpu_ref);
  free(threads_result);

  // Reset device
  SAFE_CALL(cudaDeviceReset(), "Error reseting");

  return 0;
}
