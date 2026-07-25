
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>

namespace py = pybind11;

#define TILE_WIDTH 32

__global__ void tiled_GEMM(const float *A, const float *B, float *C, int M, int K, int N){
    __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE_WIDTH + threadIdx.x;
        sharedA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;

        int bRow = t * TILE_WIDTH + threadIdx.y;
        sharedB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i)
            sum += sharedA[threadIdx.y][i] * sharedB[i][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}


py::array_t<float> gemm(py::array_t<float, py::array::c_style | py::array::forcecast> A,
                         py::array_t<float, py::array::c_style | py::array::forcecast> B) {
    auto bufA = A.request();
    auto bufB = B.request();

    int M = bufA.shape[0];
    int K = bufA.shape[1];
    int N = bufB.shape[1];

    auto C = py::array_t<float>({M, N});
    auto bufC = C.request();

    float *d_A, *d_B, *d_C;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, bufA.ptr, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, bufB.ptr, sizeB, cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    tiled_GEMM<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    cudaMemcpy(bufC.ptr, d_C, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}


PYBIND11_MODULE(tiled_gemm_cuda, m) {
    m.def("gemm", &gemm, "Tiled GEMM (float32, CUDA, TILE_WIDTH=32)");
}


