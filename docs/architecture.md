
## What is GEMM?
GEMM, also known as General Matrix Multiplication, computes C = A × B, where A is an M × K matrix
and B is a K × N matrix, with the result C being an M × N matrix. This is one of the most common
operations in deep learning — it underlies every linear layer, attention projection, and fully
connected layer. This is also why frameworks like PyTorch dispatch matrix multiplication straight
to libraries like cuBLAS.

## Why is naive GEMM slow on a GPU?
A typical triple-nested-loop GEMM assigns one output element per thread. Each thread reads one
entire row from A and one entire column from B directly from global memory — the GPU's largest
but slowest memory tier. For a K dimension of size K, that's 2×K reads from global memory per
output element, with heavy redundancy: neighboring threads independently read overlapping data
from global memory, with no sharing between them.

## Why does tiling matter?
Tiling breaks matrices A and B into small square blocks called tiles. A thread block loads one
tile of each into shared memory once, then reuses it repeatedly before moving to the next tile.
This trades a large number of slow global memory reads for a small number of slow reads plus many
fast shared-memory reads — the core idea behind every optimized GEMM kernel, including cuBLAS itself.

![Tile loading pattern — how tiles from Matrix A and B load into shared memory](../images/tile-diagram.png)

## Tile size decision: TILE_WIDTH=32
Benchmarked TILE_WIDTH=16 vs TILE_WIDTH=32 at 2048×2048 using Nsight Compute (July 24, 2026):

| Metric | TILE=16 | TILE=32 |
|---|---|---|
| Duration (Nsight) | 46.26ms | 41.89ms |
| Achieved occupancy | 99.63% | 99.97% |
| DRAM read bandwidth | 92.49 GB/s | 39.43 GB/s |

TILE_WIDTH=32 is about 9.4% faster and achieves higher occupancy. The lower DRAM bandwidth at
TILE=32 reflects *less total data movement* — each element is reused more times per global load —
not reduced throughput capability. Results were cross-validated with repeated cudaEvent timing
after discovering cold-start variance on Colab's shared T4 (single-shot runs ranged 25–42ms at
this size).
