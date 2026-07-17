CC = nvcc
FLAGS = -arch=native -O2

all: tiled_gemm naive_gemm

tiled_gemm: src/tiled_gemm.cu
	$(CC) $(FLAGS) -o tiled_gemm src/tiled_gemm.cu

naive_gemm: src/naive_gemm.cu
	$(CC) $(FLAGS) -o naive_gemm src/naive_gemm.cu

.PHONY: clean
clean:
	rm -f tiled_gemm naive_gemm
