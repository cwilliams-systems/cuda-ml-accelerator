import torch
import time
import subprocess
import os

# ── Configuration ──────────────────────────────────────────────────────
SIZES = [256, 512, 1024, 2048]
WARMUP_RUNS = 3
TIMED_RUNS = 5

# ── PyTorch CPU baseline ───────────────────────────────────────────────
def benchmark_pytorch_cpu(N):
    A = torch.randn(N, N, dtype=torch.float32)
    B = torch.randn(N, N, dtype=torch.float32)

    # warmup
    for _ in range(WARMUP_RUNS):
        C = torch.matmul(A, B)

    # timed runs
    times = []
    for _ in range(TIMED_RUNS):
        start = time.perf_counter()
        C = torch.matmul(A, B)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return sum(times) / len(times)

# ── PyTorch CUDA baseline ──────────────────────────────────────────────
def benchmark_pytorch_cuda(N):
    A = torch.randn(N, N, dtype=torch.float32, device='cuda')
    B = torch.randn(N, N, dtype=torch.float32, device='cuda')

    # warmup
    for _ in range(WARMUP_RUNS):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()

    # timed runs
    times = []
    for _ in range(TIMED_RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        C = torch.matmul(A, B)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return sum(times) / len(times)

# ── Print results table ────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"GEMM Benchmark — Tesla T4 — float32")
print(f"{'='*65}")
print(f"{'Size':<12} {'PyTorch CPU (ms)':<20} {'PyTorch CUDA (ms)':<20}")
print(f"{'-'*65}")

results = {}
for N in SIZES:
    cpu_ms  = benchmark_pytorch_cpu(N)
    cuda_ms = benchmark_pytorch_cuda(N)
    results[N] = {'cpu': cpu_ms, 'cuda': cuda_ms}
    print(f"{N}x{N:<8} {cpu_ms:<20.3f} {cuda_ms:<20.3f}")

print(f"{'='*65}")
print(f"\nNote: custom CUDA kernel timings added after pybind11 (July 25)")
print(f"Current naive GEMM benchmark: 0.248ms at 256x256 (CUDA events)")
