#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>

void cpu_gemm(const std::vector<float>& A, const std::vector<float>& B,
              std::vector<float>& C, int M, int K, int N) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

int main() {
    const int M = 2048, K = 2048, N = 2048; 

    std::srand(42);
    std::vector<float> A(M * K), B(K * N), C(M * N);

    for (int i = 0; i < M * K; ++i) A[i] = static_cast<float>(std::rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) B[i] = static_cast<float>(std::rand()) / RAND_MAX;

    auto start = std::chrono::high_resolution_clock::now();
    cpu_gemm(A, B, C, M, K, N);
    auto stop = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(stop - start).count();

    std::cout << "C[0] = " << C[0] << std::endl;
    std::cout << "CPU Time: " << ms << "ms" << std::endl;

    return 0;
}
