
#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <vector>

# define TILE_WIDTH 16


__global__ void tiled_GEMM(const float *A, const float *B, float *C, int M, int K, int N){

__shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
__shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];

int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

float sum = 0.0f;

int numTiles = (K + TILE_WIDTH - 1)/TILE_WIDTH;

for (int t = 0; t < numTiles; ++t){

  int aCol = t * TILE_WIDTH + threadIdx.x;

  if (row < M && aCol < K){
  sharedA[threadIdx.y][threadIdx.x] = A[row * K + aCol];
  }
  else{
    sharedA[threadIdx.y][threadIdx.x] = 0.0f;
  }

 int bRow = t * TILE_WIDTH + threadIdx.y;

  if (bRow < K && col < N){
    sharedB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
  }
    else{
    sharedB[threadIdx.y][threadIdx.x] = 0.0f;
  }

  __syncthreads();

  for(int i = 0; i < TILE_WIDTH; ++i){

    sum+= sharedA[threadIdx.y][i] * sharedB[i][threadIdx.x];
  }
   __syncthreads();
  }

 if(row < M && col < N){

  C[row * N + col] = sum;
 }
}

int main(){

  const int M = 256;
  const int K = 256;
  const int N = 256;

  std::srand(42);

  std::vector<float> A(static_cast<std::size_t>(M) * K);
  std::vector<float> B(static_cast<std::size_t>(K) * N);
  std::vector<float> C(static_cast<std::size_t>(M) * N);

  for(int i = 0; i < M * K; ++i){
    A[i] = static_cast<float>(std::rand())/ RAND_MAX;
  }

  for(int i = 0; i < K * N; ++i){
    B[i] = static_cast<float>(std::rand())/RAND_MAX;
  }

  float *d_A;
  float *d_B;
  float *d_C;

  std::size_t size_A = static_cast<std::size_t>(M) * K * sizeof(float);
  std::size_t size_B = static_cast<std::size_t>(K) * N * sizeof(float);
  std::size_t size_C = static_cast<std::size_t>(M) * N * sizeof(float);

  cudaMalloc(&d_A, size_A);
  cudaMalloc(&d_B, size_B);
  cudaMalloc(&d_C, size_C);

  cudaMemcpy(d_A, A.data(), size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data(), size_B, cudaMemcpyHostToDevice);

  dim3 blockSize (TILE_WIDTH, TILE_WIDTH);
  dim3 gridSize((N + TILE_WIDTH - 1) / TILE_WIDTH,
              (M + TILE_WIDTH - 1) / TILE_WIDTH);


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  tiled_GEMM<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaDeviceSynchronize();

  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(C.data(), d_C, size_C, cudaMemcpyDeviceToHost);

  std::cout << "C[0] = " << C[0] << std::endl;
  std::cout << "Kernel Time: " << milliseconds << "ms" << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;

}



