#include "common.h"
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

void initializeMatrix(int *matrix, const int rows, const int cols){
  for (size_t i = 0; i < rows * cols; i++)
    matrix[i] = i;
}

__global__ void multiplyMatricesWithCuda(int *matrixA, int *matrixB, int *result, const int rows, const int cols){
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = blockIdx.y;
  if(ix < rows && iy < cols){
    int sum = 0;
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
  int *result;
  int *dev_matrixA;
  int *dev_matrixB;
  int *dev_result;

  // Set up size of matrix
  const int rows = 1000;
  const int cols = 1000;
  printf("Matrix size: rows %d columns %d\n", rows, cols);

  int bytes = rows * cols * sizeof(int);

  // Allocate matrices memory
  matrixA = (int *) malloc(bytes);
  matrixB = (int *) malloc(bytes);
  result = (int *) malloc(bytes);

  // Initialize matrices
  initializeMatrix(matrixA, rows, cols);
  initializeMatrix(matrixB, rows, cols);

  // Allocate device global memory
  SAFE_CALL(cudaMalloc((void **)&dev_matrixA, bytes), "Error allocating dev_matrixA");
  SAFE_CALL(cudaMalloc((void **)&dev_matrixB, bytes), "Error allocating dev_matrixB");
  SAFE_CALL(cudaMalloc((void **)&dev_result, bytes), "Error allocating dev_result");

  // Transfer data from host to device
  SAFE_CALL(cudaMemcpy(dev_matrixA, matrixA, bytes, cudaMemcpyHostToDevice), "Error copying dev_matrixA");
  SAFE_CALL(cudaMemcpy(dev_matrixB, matrixB, bytes, cudaMemcpyHostToDevice), "Error copying dev_matrixB");

  // Invoke kernel at host side
  int dimx = 512;
  dim3 block(dimx, 1);
  dim3 grid((rows + block.x - 1) / block.x, cols);

  auto start_at = chrono::high_resolution_clock::now();
  multiplyMatricesWithCuda<<<grid, block>>>(dev_matrixA, dev_matrixB, dev_result, rows, cols);
  SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
  auto end_at = chrono::high_resolution_clock::now();
  chrono::duration<float, std::milli> duration_ms = end_at - start_at;

  printf("Multiply matrices on GPU <<<(%d,%d), (%d,%d)>>> elapsed: %f ms (%.2f seconds)\n",
        grid.x, grid.y, block.x, block.y, duration_ms.count(), duration_ms.count() / 1000);

  // SAFE_CALL kernel error
  SAFE_CALL(cudaGetLastError(), "Error with last error");

  // Copy kernel result back to host side
  SAFE_CALL(cudaMemcpy(result, dev_result, bytes, cudaMemcpyDeviceToHost), "Error copying dev_result");

  // Free device global memory
  SAFE_CALL(cudaFree(dev_matrixA), "Error freeing memory");
  SAFE_CALL(cudaFree(dev_matrixB), "Error freeing memory");
  SAFE_CALL(cudaFree(dev_result), "Error freeing memory");

  // Free matrices memory
  free(matrixA);
  free(matrixB);
  free(result);

  // Reset device
  SAFE_CALL(cudaDeviceReset(), "Error reseting");

  return 0;
}
