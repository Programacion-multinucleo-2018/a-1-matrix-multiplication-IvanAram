#include <iostream>
#include <chrono>

void initializeMatrix(int *matrix, const int rows, const int cols){
  for (size_t i = 0; i < rows * cols; i++)
    matrix[i] = i;
}

void multiplyMatrices(int *matrixA, int *matrixB, long *result, const int rows, const int cols){
  long sum;
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      sum = 0.f;
      for (size_t k = 0; k < cols; k++) {
        sum += matrixA[i * rows + k] * matrixB[k * cols + j];
      }
      result[i * rows + j] = sum;
    }
  }
}

int main(int argc, char const *argv[]) {
  // Declare matrices
  int *matrixA;
  int *matrixB;
  long *result;

  // Set up size of matrix
  int N;
  if (argc == 2) {
    N = atoi(argv[1]);
  } else {
    N = 1000;
  }
  const int rows = N;
  const int cols = N;
  printf("Matrix size: rows %d columns %d\n", rows, cols);

  // Allocate matrices memory
  matrixA = (int *) malloc(rows * cols * sizeof(int));
  matrixB = (int *) malloc(rows * cols * sizeof(int));
  result = (long *) malloc(rows * cols * sizeof(long));

  // Initialize matrices
  initializeMatrix(matrixA, rows, cols);
  initializeMatrix(matrixB, rows, cols);

  // Mutiply matrices on host
  auto start_at = std::chrono::high_resolution_clock::now();
  multiplyMatrices(matrixA, matrixB, result, rows, cols);
  auto end_at = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> duration_ms = end_at - start_at;
  printf("Multiply matrix on host without threads elapsed: %f ms (%.2f seconds)\n", duration_ms.count(), duration_ms.count() / 1000);

  // Free matrices memory
  free(matrixA);
  free(matrixB);
  free(result);

  return 0;
}
