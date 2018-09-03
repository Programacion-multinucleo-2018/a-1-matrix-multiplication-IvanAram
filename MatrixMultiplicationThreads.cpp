#include <iostream>
#include <pthread.h>
#include <chrono>

#define THREADS 8

typedef struct thread_data_struct {
  int start;
  int end;
  int rows;
  int cols;
  int *matrixA;
  int *matrixB;
  int *result;
} thread_data_t;

void initializeMatrix(int *matrix, const int rows, const int cols){
  for (size_t i = 0; i < rows * cols; i++)
    matrix[i] = i;
}

void *multiplyMatricesWithThreads(void *args){
  thread_data_t *data = (thread_data_t *) args;
  int sum;
  for (size_t i = data->start; i < data->end; i++) {
    for (size_t j = 0; j < data->cols; j++) {
      sum = 0.f;
      for (size_t k = 0; k < data->cols; k++) {
        sum += data->matrixA[i * data->rows + k] * data->matrixB[k * data->cols + j];
      }
      data->result[i * data->rows + j] = sum;
    }
  }
  pthread_exit(NULL);
}

int main(int argc, char const *argv[]) {
  // Declare matrices
  int *matrixA;
  int *matrixB;
  int *result;

  // Set up size of matrix
  const int rows = 1000;
  const int cols = 1000;
  printf("Matrix size: rows %d columns %d\n", rows, cols);

  // Allocate matrices memory
  matrixA = (int *) malloc(rows * cols * sizeof(int));
  matrixB = (int *) malloc(rows * cols * sizeof(int));
  result = (int *) malloc(rows * cols * sizeof(int));

  // Initialize matrices
  initializeMatrix(matrixA, rows, cols);
  initializeMatrix(matrixB, rows, cols);

  // Declare threads and thread data
  pthread_t threads[THREADS];
  thread_data_t data[THREADS];

  // Initialize threads data
  int step = (int)(rows / THREADS);
  for (size_t i = 0; i < THREADS; i++) {
    data[i].start = step * i;
    data[i].end = step * (i + 1);
    data[i].rows = rows;
    data[i].cols = cols;
    data[i].matrixA = matrixA;
    data[i].matrixB = matrixB;
    data[i].result = result;
  }

  // Multiply matrices on host with threads
  auto start_at =  std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < THREADS; i++) {
    pthread_create(&threads[i], NULL, multiplyMatricesWithThreads, (void *) &data[i]);
  }
  for (size_t i = 0; i < THREADS; i++) {
    pthread_join(threads[i], NULL);
  }
  auto end_at =  std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> duration_ms = end_at - start_at;
  printf("Multiply matrix on host with threads (%d threads) elapsed: %f ms (%.2f seconds)\n", THREADS, duration_ms.count(), duration_ms.count() / 1000);

  // Free matrices memory
  free(matrixA);
  free(matrixB);
  free(result);

  pthread_exit(NULL);
}
