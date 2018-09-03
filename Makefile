C_C = g++
CUDA_C = nvcc

CFLAGS = -std=c++11
LDFLAGS = -lpthread

EXE1 = bin/matrix_multiplication
EXE2 = bin/matrix_multiplication_threads
EXE3 = bin/matrix_multiplication_cuda
EXE4 = bin/matrix_multiplication_comparison

PROG1 = MatrixMultiplication.cpp
PROG2 = MatrixMultiplicationThreads.cpp
PROG3 = MatrixMultiplicationCuda.cu
PROG4 = MatrixMultiplicationComparisonCPU_GPU.cu

all:
	$(C_C) -o $(EXE1) $(PROG1) $(CFLAGS)
	$(C_C) -o $(EXE2) $(PROG2) $(CFLAGS) $(LDFLAGS)
	$(CUDA_C) -o $(EXE3) $(PROG3)
	$(CUDA_C) -o $(EXE4) $(PROG4) $(LDFLAGS)

rebuild: clean all

clean:
	rm -f ./bin/*
