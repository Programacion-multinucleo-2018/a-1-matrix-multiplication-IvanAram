#!/bin/bash

echo $'\nBuilding executables.\n'
make rebuild
echo $'\nRunning matrix multiplication of NxN with N = 1000, 2000 and 4000.\n'
./bin/matrix_multiplication_comparison 1000
./bin/matrix_multiplication_comparison 2000
./bin/matrix_multiplication_comparison 4000
echo $'Done.\n'
