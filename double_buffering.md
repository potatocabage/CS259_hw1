# Double-buffering ablation — `conv.cu` vs `conv_no_db.cu`

A side-by-side comparison of the WMMA convolution kernel with and without the
synapse double-buffering scheme. All numbers are from `ncu` per-kernel
counters on the eda-cluster TITAN V (compute capability 7.0).

## What "double-buffering" means here

In [conv.cu](conv.cu), the synapse load for each `(k_row, k_col)` goes
`global → per-thread regs → shared memory`, with two shared buffers
(`synapse_smem_buf1`, `synapse_smem_buf2`) and a per-thread `synapse_reg_buf`.
The structure is:

```
pre-load (0, 0): global → regs → active_smem
for each (k_row, k_col):
    if has_next: load NEXT (k_row, k_col): global → regs   ← prefetch
    for j in NI_CHUNK / 16: WMMA mma using ACTIVE smem buf
    __syncthreads()
    if has_next: store regs → IDLE smem buf                ← stage prefetch
    __syncthreads()
    swap active / idle
```

The point is to overlap the next iteration's global load with the current
iteration's `mma_sync`.

## What `conv_no_db.cu` does instead

[conv_no_db.cu](conv_no_db.cu) drops the prefetch: one shared buffer, no
register staging. Single helper `load_synapse_xy_to_smem` writes straight
from global to smem.

```
for each (k_row, k_col):
    load this (k_row, k_col): global → smem
    __syncthreads()
    for j in NI_CHUNK / 16: WMMA mma using smem buf
    __syncthreads()
```

Everything else — input-tile loader, WMMA fragment shapes, epilogue/ReLU,
launch config (256 threads, grid `(B, NYSCL/8, NXSCL/16)`),
`NI_CHUNK = 64`, host-side padding — is identical.

## Correctness

`CONV_VERIFY=1 ./conv_cuda_no_db` prints `OK` for all four configs. Numerics
match the original to within fp16 tolerance — the math sequence is the same,
only issue order differs.

## Per-kernel ncu comparison

Captured with the same command from [ncu_guide.md](ncu_guide.md) §8(a),
output in [ncu_metrics_no_db.csv](ncu_metrics_no_db.csv) and
[ncu_sol_no_db.csv](ncu_sol_no_db.csv); compare against [ncu_metrics.csv](ncu_metrics.csv)
and [ncu_sol.csv](ncu_sol.csv).

| Config     | kernel µs orig → no-db (Δ) | GFLOPS orig → no-db | HMMA %peak orig → no-db | L1 bytes (no-db ÷ orig) | DRAM bytes (no-db ÷ orig) |
|------------|----------------------------|---------------------|-------------------------|--------------------------|---------------------------|
| Conv1 B=1  | 847 → 811 (**−4.3%**)      | 4366 → 4563         | 4.43 → 4.61             | **0.57×**                | 0.86×                     |
| Conv1 B=16 | 11555 → 11192 (**−3.1%**)  | 5123 → 5288         | 5.19 → 5.35             | **0.57×**                | 0.90×                     |
| Conv2 B=1  | 10980 → 10434 (**−5.0%**)  | 84 → 89             | 0.11 → 0.12             | **0.54×**                | 0.91×                     |
| Conv2 B=16 | 12581 → 12004 (**−4.6%**)  | 1176 → 1233         | 1.56 → 1.63             | **0.54×**                | 1.00×                     |

GFLOPS uses the algorithmic FLOP count from [conv_flop.md](conv_flop.md):
`2 · B · NYSCL · NXSCL · KY · KX · Ni · Nn`.

### SoL throughputs (% of peak)

| Config     | metric                  | orig    | no-db   |
|------------|-------------------------|---------|---------|
| Conv1 B=1  | Compute (SM)            | 13.02 % | 11.90 % |
| Conv1 B=1  | Memory                  | 58.23 % | 59.52 % |
| Conv1 B=1  | L1/TEX Cache            | 69.85 % | 72.04 % |
| Conv1 B=1  | L2 Cache                |  6.30 % |  4.42 % |
| Conv1 B=1  | DRAM                    |  5.64 % |  5.15 % |
| Conv1 B=16 | Compute (SM)            | 15.12 % | 13.77 % |
| Conv1 B=16 | Memory                  | 67.64 % | 68.91 % |
| Conv1 B=16 | L1/TEX Cache            | 68.61 % | 70.06 % |
| Conv1 B=16 | L2 Cache                |  7.34 % |  5.11 % |
| Conv1 B=16 | DRAM                    |  8.37 % |  7.83 % |
| Conv2 B=1  | Compute (SM)            |  0.31 % |  0.29 % |
| Conv2 B=1  | Memory                  |  1.43 % |  1.49 % |
| Conv2 B=1  | L1/TEX Cache            | 57.20 % | 59.45 % |
| Conv2 B=1  | L2 Cache                |  0.16 % |  0.12 % |
| Conv2 B=1  | DRAM                    |  0.07 % |  0.08 % |
| Conv2 B=16 | Compute (SM)            |  4.39 % |  4.04 % |
| Conv2 B=16 | Memory                  | 19.94 % | 20.67 % |
| Conv2 B=16 | L1/TEX Cache            | 50.06 % | 51.80 % |
| Conv2 B=16 | L2 Cache                |  2.26 % |  1.65 % |
| Conv2 B=16 | DRAM                    |  2.22 % |  2.33 % |

## Interpretation

Removing the double-buffering made the kernel **3–5 % faster** across all four
configs and cut **L1/TEX traffic by ~43–46 %**. DRAM is essentially unchanged
(0.86×–1.00×) — the same data is being fetched from HBM in both versions, so
the prefetch wasn't reducing global reads.

Why the prefetch hurt rather than helped:

- This kernel is **L1-throughput-bound, not latency-bound** (see
  [roofline.md](roofline.md) and [ncu_guide.md](ncu_guide.md) §9). The SoL
  diagnosis already pointed at L1 as the binding pipe — Memory %peak ≈ 60–70 %
  while Compute %peak < 16 %. The double-buffering scheme was buying latency
  hiding the kernel didn't need.
- The `global → regs → smem` staging emits more L1/TEX-pipe activity per
  element than `global → smem` (which on Volta still goes
  LDG-into-temp + STS, but with tighter scheduling and no live-across-iteration
  register file). Whatever the exact mechanism — extra L1 line transactions
  from the explicit reg buffer, or scheduling artifacts from the persistent
  `synapse_reg_buf[4]` carrying state across kk iterations — `l1tex__t_bytes.sum`
  drops by ~0.55× when we remove it.
- HMMA pipe utilization rises slightly (0.18–0.22 pp) because the L1 pipe
  spends less time on bookkeeping traffic and the tensor pipe gets a marginally
  larger share of issue slots.

Bottom line: double-buffering is a textbook trick for latency-bound kernels;
applying it to an L1-bandwidth-bound kernel is at best neutral and here is
mildly counterproductive. Useful evidence for question 5 ("optimizations that
did not help") in the assignment.

## How to reproduce

```bash
cd /usr/eda/CS251A/nwei/cs259-miniproj-ref
make conv_cuda_no_db

# correctness
CONV_VERIFY=1 ./conv_cuda_no_db

# per-kernel metrics (writes ncu_metrics_no_db.csv)
TMPDIR=$HOME/.ncu_tmp ncu -k regex:convolution_kernel \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__inst_executed_pipe_tensor_op_hmma.sum,l1tex__t_bytes.sum,lts__t_bytes.sum \
    --csv --log-file ncu_metrics_no_db.csv ./conv_cuda_no_db

# Speed-of-Light section (writes ncu_sol_no_db.csv)
TMPDIR=$HOME/.ncu_tmp ncu --section SpeedOfLight -k regex:convolution_kernel \
    --csv --log-file ncu_sol_no_db.csv ./conv_cuda_no_db
```

Wall-clock `./conv_cuda_no_db` numbers include cudaMalloc + H→D copy +
fp32→fp16 conversion and shouldn't be used for a clean kernel-vs-kernel
comparison — use `gpu__time_duration.sum` from the CSVs.
