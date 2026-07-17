# cuda-ml-accelerator

CUDA GEMM kernel with shared memory tiling, Nsight profiling, and Python bindings — Summer 2026.

Built on a Tesla T4 (Google Colab). Benchmarked against PyTorch CPU and CUDA baselines.

## Performance

*Full benchmark table coming July 27 — currently profiling across matrix sizes 256×256 to 4096×4096.*

| Matrix Size | CPU ms | Naive CUDA ms | Tiled CUDA ms | PyTorch CUDA ms | GB/s |
|------------|--------|---------------|---------------|-----------------|------|
| 1024×1024 | TBD | 23.956 | 18.897 | TBD | 34.93 |

## Architecture

Tiled shared-memory GEMM with TILE_WIDTH=16. Each thread block loads a 16×16 tile of A and B into shared memory, computes the partial dot product, then slides to the next tile. This eliminates redundant global memory reads — each element is loaded once per tile instead of once per output element.

## Nsight Compute Profile (1024×1024, T4)


Findings: Kernel is compute bound on the FP64 pipeline. Switching to float32 is the primary optimization opportunity — T4 has a 32:1 FP32/FP64 performance ratio.

## Project Structure


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

## Status

- [x] Naive GEMM kernel
- [x] Tiled GEMM with shared memory (TILE_WIDTH=16)
- [x] CUDA event timing
- [x] Nsight Compute profile
- [ ] Float32 optimization (in progress)
- [ ] Full benchmark table across all matrix sizes
- [ ] pybind11 Python bindings
- [ ] PyTorch baseline comparison
