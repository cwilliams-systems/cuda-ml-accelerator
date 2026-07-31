
# Memory Coalescing & Bank Conflicts

## Bank conflicts explained
Shared memory is physically divided into banks (typically 32 on modern
GPUs). When multiple threads in a warp access different addresses in the
same bank simultaneously, those accesses serialize instead of happening
in parallel which is a "bank conflict" that silently kills performance.

## Measured result: zero conflicts
Profiled with Nsight Compute (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld/st`)
on both TILE_WIDTH=16 and TILE_WIDTH=32 at 2048×2048:

| Tile Size | Load conflicts | Store conflicts |
|---|---|---|
| 16 | 0 | 0 |
| 32 | 0 | 0 |

Zero bank conflicts at either tile size, so the standard
`sharedB[TILE_WIDTH][TILE_WIDTH+1]` padding fix was not needed — the
kernel's existing access pattern (`sharedA[threadIdx.y][i]`,
`sharedB[i][threadIdx.x]`) doesn't create bank conflicts on this hardware.
