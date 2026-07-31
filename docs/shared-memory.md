
#What is Shared Memory?

Shared Memory is a small(tens of KB), fast on chip memory that is scoped to a singular thread block
meaning every thread in the block can read and write it and it is roughly 10 to 20x faster than global memory.

[Diagram: memory hierarchy: see images/memory-hierarchy.png]

##How the tiled kernel uses it

Each thread block declares two shared meory arrays sized TILE_WIDTH X TILE_WIDTH (in my case 32x32 in the final kernel).
Threads cooperatively load one tile of A and one tile of B in these arrays using __syncthreads to synchronize then every thread in
the block computes a partial dot product by reading the shared memory TILDE_WIDTH amount of times(in my case 32 fast reads for every 1 slow global reads)

## Occupancy tradeoff
At TILE_WIDTH=32, each block uses 1024 threads (the T4's per-block
maximum) and 8KB of shared memory. This limits each SM to one resident
block at a time, but that single block already fills the SM's full
32-warp capacity, so achieved occupancy stayed at 99.97%, actually
slightly higher than TILE_WIDTH=16's 99.63% (which allowed more blocks
per SM but with 256 threads each).



