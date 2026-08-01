# cuda-ml-accelerator
**⚡ 932.9 GFLOPS at 4096×4096 (float32, T4)** These feature a tiled shared-memory GEMM with Nsight-validated tile size selection, pybind11 Python bindings, and correctness checks against CPU, naive, and PyTorch baselines across five matrix sizes. It achieves 26% of the measured throughput of PyTorch/cuBLAS at this size, showing 932.9 GFLOPS compared to cuBLAS's approximately 3,613 GFLOPS equivalent. The remaining gaps are the results from register blocking, double buffering, and hand-tuned instruction scheduling used in cuBLAS but are not included in this porjet

![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![GPU](https://img.shields.io/badge/GPU-Tesla%20T4-76B900)
![pybind11](https://img.shields.io/badge/pybind11-enabled-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

CUDA GEMM kernel with shared memory tiling, Nsight profiling, and Python bindings for Summer 2026.
Built on a Tesla T4 (Google Colab). Benchmarked against PyTorch CPU and CUDA baselines.

## Performance
| Matrix Size | CPU ms | Naive CUDA ms | Tiled CUDA ms | PyTorch CUDA ms | Tiled GFLOPS |
|------------|--------|---------------|---------------|------------------|--------------|
| 256×256 | 20.647 | 0.218 | 0.207 | 0.057 | 162.1 |
| 512×512 | 203.569 | 1.005 | 0.599 | 0.116 | 448.1 |
| 1024×1024 | 3509 | 5.574 | 5.355 | 0.608 | 401.0 |
| 2048×2048 | 65316.9 | 69.062 | 28.7* | 4.203 | 598.6 |
| 4096×4096 | 813709 | 330.641 | 147.31* | 38.039 | 932.9 |

The warm-run times have been averaged (the individual cold-start runs on Colab's shared T4 yielded higher and more inconsistent results: see the Tile Size Optimization section). The PyTorch CUDA figures make use of warmup and averaging as defined in the script benchmarks/pytorch_baseline.py.

Correct results are obtained at all sizes since the naive, tiled, and CPU versions all produce the same C[0] output. The PyTorch/cuBLAS version beats the hand-written tiled kernel at every size; this performance advantage is due to the use of register blocking, double buffering, and hand-tuned instruction scheduling in cuBLAS, factors which are beyond the scope of this project.

## Architecture
Tiled shared-memory GEMM with TILE_WIDTH=32 (updated July 24 from an initial TILE_WIDTH=16 see below). Each thread block loads a 32×32 tile of A and B into shared memory, computes the partial dot product, then slides to the next tile. This eliminates redundant global memory reads — each element is loaded once per tile instead of once per output element.
### Tile size decision (July 24, 2026)
Benchmarked TILE_WIDTH=16 vs TILE_WIDTH=32 at 2048×2048 with Nsight Compute. TILE=32 wins: ~9.4% faster Duration (41.89ms vs 46.26ms), higher occupancy (99.97% vs 99.63%), and DRAM read bandwidth drops by more than half (92.49→39.43 GB/s) reflecting greater shared-memory data reuse per global load, not reduced throughput. Zero shared-memory bank conflicts measured at either tile size, so no padding was needed. Result cross-validated with repeated warm cudaEvent timing after discovering cold-start variance on Colab's shared T4.
## Nsight Compute Profile (1024×1024, T4, double precision)
Finding: kernel is compute bound on the FP64 pipeline at 97.7% utilization.
T4 FP32/FP64 ratio is 32:1. ML Accelerator uses float32 to exploit full T4 throughput.
See the float32 profile below for the shipped TILE_WIDTH=32 kernel.
## Nsight Compute Profile — float32, TILE_WIDTH=32 (July 24, 2026)
Matrix: 2048×2048, float32. Data collected during the TILE=16 vs TILE=32 comparison.

| Metric | Value |
|---|---|
| Duration (Nsight) | 41.89ms |
| Achieved occupancy | 99.97% |
| DRAM read bandwidth | 39.43 GB/s |
| DRAM write bandwidth | 0.506 GB/s |
| Bank conflicts (load/store) | 0 / 0 |
| Block size | 1024 threads |
| Registers per thread | 42 |

DRAM read bandwidth is lower than TILE_WIDTH=16's 92.49 GB/s at the same size — this reflects greater shared-memory data reuse per global load (less total data movement needed to do the same work), not reduced throughput. Zero bank conflicts measured, so no padding was applied.
## Project Structure
## Project Structure
```
cuda-ml-accelerator/
├── README.md
├── LICENSE
├── Makefile
│
├── src/
│   ├── naive_gemm.cu       # Naive GEMM (double precision baseline)
│   ├── tiled_GEMM.cu       # Tiled shared-memory GEMM (TILE_WIDTH=32, final)
│   ├── cpu_gemm.cpp        # CPU reference implementation
│   └── bindings.cu         # pybind11 bindings — exposes gemm() to Python
│
├── benchmarks/
│   └── benchmark.py        # PyTorch CPU/CUDA baseline benchmarks
│
├── docs/
│   ├── architecture.md     # GEMM explanation, tiling, tile size decision
│   ├── shared-memory.md    # Shared memory mechanics and occupancy
│   └── memory-coalescing.md # Bank conflict analysis
│
└── images/
    ├── tile-diagram.png       # Tile loading pattern
    └── memory-hierarchy.png   # Global → shared → register hierarchy
```
## Requirements
- CUDA Toolkit 13.0+
- Python 3.x
- pybind11
- PyTorch (for baseline comparison)
- Tesla T4 or equivalent Nvidia GPU
## Build
```bash
make tiled_gemm
./tiled_gemm
```
## Usage (Python)
Once built, the CUDA kernel is callable directly from Python via the pybind11 bindings:

```python
import numpy as np
import tiled_gemm_cuda

A = np.random.rand(1024, 1024).astype(np.float32)
B = np.random.rand(1024, 1024).astype(np.float32)

C = tiled_gemm_cuda.gemm(A, B)
```

Build the module first:
```bash
nvcc -O3 -arch=sm_75 -shared -Xcompiler -fPIC \
  $(python3 -m pybind11 --includes) \
  -I$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])") \
  src/bindings.cu -o tiled_gemm_cuda.so
```
## Status
- [x] Naive GEMM kernel
- [x] Tiled GEMM with shared memory (TILE_WIDTH=32)
- [x] CUDA event timing
- [x] Nsight Compute profile
- [x] Float32 optimization
- [x] Full benchmark table across all matrix sizes
- [x] pybind11 Python bindings
- [x] PyTorch baseline comparison (CPU + CUDA, 256–4096, July 27)
