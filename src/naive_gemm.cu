
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <cstdlib>

__global__ void matrixMultiply( const float *A, const float *B, float *C, int M, int K, int N){

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;



  if(row < M && col < N){
     float sum = 0.0f;
    for(int k = 0; k < K; ++k){

    sum += A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = sum;
}
}

int main(){
  const int M = 256;
  const int K = 256;
  const int N = 256;

  std::vector<float> A(static_cast<std::size_t>(M) * K);
  std::vector<float> B(static_cast<std::size_t>(K) * N);
  std::vector<float> C(static_cast<std::size_t>(M) * N);

  for(int i = 0;i < M * K; ++i){
    A[i] = static_cast<float>(std::rand())/RAND_MAX;
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

  cudaMemcpy(d_A,A.data(),size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B.data(), size_B, cudaMemcpyHostToDevice);

  dim3 blockDim(16,16);

  dim3 gridSize((N + 15)/16, (M + 15)/16);

cudaEvent_t start, stop;

cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);

matrixMultiply<<<gridSize, blockDim>>>(d_A, d_B, d_C, M, K, N);

cudaEventRecord(stop);

cudaEventSynchronize(stop);

float milliseconds = 0.0f;
cudaEventElapsedTime(&milliseconds, start, stop);

  

  cudaMemcpy(C.data(), d_C, size_C, cudaMemcpyDeviceToHost);

  std::cout << "C[0][0]" << C[0] << std::endl;
  std::cout << "Elapsed time is: " << milliseconds << "ms" << std::endl;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;


}
