# cuda-ml-accelerator
CUDA GEMM kernel with shared memory tiling, Nsight profiling, and Python bindings — Summer 2026.
Built on a Tesla T4 (Google Colab). Benchmarked against PyTorch CPU and CUDA baselines.
## Performance
*Full benchmark table coming July 27 — currently profiling across matrix sizes 256×256 to 4096×4096.*
| Matrix Size | CPU ms | Naive CUDA ms | Tiled CUDA ms | PyTorch CPU ms | PyTorch CUDA ms |
|------------|--------|---------------|---------------|----------------|-----------------|
| 256×256 | TBD | 0.218 | 0.170 | 0.434 | 0.065 |
| 512×512 | TBD | TBD | 0.703 | 2.350 | 0.134 |
| 1024×1024 | TBD | TBD | 3.401 | 21.977 | 0.807 |
| 2048×2048 | TBD | TBD | 34.700 | 133.788 | 6.070 |
## Architecture
Tiled shared-memory GEMM with TILE_WIDTH=16. Each thread block loads a 16×16 tile of A and B into shared memory, computes the partial dot product, then slides to the next tile. This eliminates redundant global memory reads — each element is loaded once per tile instead of once per output element.
## Nsight Compute Profile (1024×1024, T4, double precision)
Finding: kernel is compute bound on the FP64 pipeline at 97.7% utilization.
T4 FP32/FP64 ratio is 32:1 — ML Accelerator uses float32 to exploit full T4 throughput.
New Nsight profile on float32 tiled kernel coming July 24.
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
- [x] Float32 optimization
- [ ] Full benchmark table across all matrix sizes
- [ ] pybind11 Python bindings
- [x] PyTorch baseline comparison (CPU + CUDA, 256–2048, July 18)
