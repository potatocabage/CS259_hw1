# Optimization log — `conv_opt.cu` on TITAN V (sm_70)

This file tracks iterative optimizations to the WMMA convolution kernel
originally implemented in [conv.cu](conv.cu). The starting point and the
analysis driving the iterations are documented in [roofline.md](roofline.md).

Each iteration follows the same pattern:
1. **Hypothesis** — what we expect to fix and why, grounded in the previous ncu run.
2. **Change** — concrete code edit (function names, line numbers in `conv_opt.cu`).
3. **Build & correctness** — `make conv_cuda_opt`, `CONV_VERIFY=1 ./conv_cuda_opt`.
4. **ncu measurement** — same metrics schema as `roofline.md` (DRAM bytes, L1/L2 traffic, kernel duration, GFLOPS, SoL %, tensor pipe util).
5. **Conclusion** — what worked, what surprised, and what the next iteration targets.

Hardware roof for reference (from [roofline.md](roofline.md#hardware-nvidia-titan-v-sm_70)):
**110 TFLOPS** tensor peak, **652.8 GB/s** HBM2, ridge ≈ **168 FLOP/byte**.

---

## Headline numbers

| Iteration | Conv1 B=16 GFLOPS | Conv2 B=16 GFLOPS | Conv1 ÷ baseline | Conv2 ÷ baseline | Notes |
|---|---:|---:|---:|---:|---|
| Iter 0 (baseline)  | 5,123 | 1,178 | 1.00× | 1.00× | conv_opt.cu = clone of conv.cu, reproduces roofline.md |
| Iter 1 (Nn fusion) | 7,034 | 1,907 | 1.37× | 1.62× | NN_GROUP=4, hoist input load out of `i` loop, drop double buffer |
| Iter 2 (SMEM pad)  | 10,353 | 2,581 | 2.02× | 2.19× | NI_PAD=8 halfs → 16-way bank conflict → 2-way (18× reduction) |
| Iter 3             | TBD | TBD | TBD   | TBD   | Conv2 (b,y,x) M-dim packing |

(Numbers are filled in as each iteration completes.)

---

## Iteration 0 — Baseline reproduction

**Goal:** confirm `conv_opt.cu` (verbatim clone) reproduces the [roofline.md](roofline.md) numbers within ~3% on this machine. Every "speedup" claimed below is relative to this run, not the historical numbers.

**Result:** within ~1% on every metric. All 4 layer/batch combinations pass `CONV_VERIFY=1` correctness.

| Metric | Conv1 B=16 (Iter 0) | (roofline.md) | Conv2 B=16 (Iter 0) | (roofline.md) |
|---|---:|---:|---:|---:|
| Kernel duration | 11.553 ms | 11.55 ms | 12.564 ms | 12.58 ms |
| Achieved GFLOPS | **5,123** | 5,125 | **1,178** | 1,176 |
| % of 110 TFLOPS peak | 4.66% | 4.66% | 1.07% | 1.07% |
| Tensor pipe util (ncu) | 5.19% | 5.19% | 1.56% | 1.56% |
| DRAM read | 401.58 MB | 403.98 MB | 174.57 MB | 174.04 MB |
| DRAM write | 231.72 MB | 233.14 MB | 9.57 MB | 10.22 MB |
| DRAM total | 633.3 MB | 637.1 MB | 184.1 MB | 184.3 MB |
| Achieved AI (DRAM) | 93.5 FLOP/B | 92.9 | 80.4 FLOP/B | 80.3 |
| L1 traffic | 2170.8 MB | 2171 MB | 650.1 MB | 650 MB |
| L2 traffic | 1489.1 MB | 1490 MB | 499.3 MB | 499 MB |
| DRAM SoL | 8.39% | 8.37% | 2.23% | 2.22% |
| L1/TEX SoL | **68.42%** | 68.61% | **50.05%** | 50.06% |
| L2 SoL | 7.32% | 7.34% | 2.25% | 2.26% |
| Compute SM SoL | 15.05% | 15.12% | 4.39% | 4.39% |
| HMMA insts | 115.6 M | 115.6 M | 37.7 M | 37.7 M |
| ncu rule | "Memory more heavily utilized than Compute: L1 bottleneck" | (same) | (grid too small to fill SMs; L1 still 50%) | (same) |

The reproducibility is exact enough to attribute later changes to code edits, not to environment.

---

## Iteration 1 — Nn-tile fusion (hoist `load_neuron_i_tile`, share toeplitz across `NN_GROUP` outputs)

**Hypothesis.** The dominant L1 cost is the input-tile reload pattern flagged in [roofline.md:166](roofline.md#L166): `load_neuron_i_tile` sits inside the `i` (output-channel chunk) loop, so the same input tile is staged into SMEM `Nn/16` times per `ic_base`. The toeplitz fragment is loaded from SMEM into registers `Nn/16` times for the same reason. Fusing the output-channel loop and reusing one toeplitz across `NN_GROUP=4` accumulators should:
- cut input-tile global loads `NN_GROUP×` (4× for Conv1, 4× for Conv2 vs 32×-old-equivalent because ic_base loops 8× in Conv2 — net 8× over the old design),
- cut toeplitz SMEM loads `NN_GROUP×` for both layers,
- drop the double-buffer machinery (single-buffer the filter; we now do enough work per stage to amortize a single sync).

**Change** ([conv_opt.cu](conv_opt.cu)):
- Added `load_synapse_group_smem<NI_CHUNK, NN_GROUP>` that stages NN_GROUP filter slices `[NN_GROUP][16][NI_CHUNK]` into one SMEM buffer in a single pass.
- Rewrote the kernel as `template<int NI_CHUNK, int NN_GROUP>`. Outer loop is now `for i_group in [0, (Nn/16)/NN_GROUP)`; inside, `c_frags[NN_GROUP]` accumulates. `load_neuron_i_tile` runs once per `(i_group, ic_base)`. Inner WMMA loop loads `toeplitz_frag` once per `(k_row, k_col, j)` and reuses it across all `NN_GROUP` `mma_sync` calls.
- Dropped the double-buffer scheme (`synapse_smem_buf1`/`buf2`, `synapse_reg_buf`, `load_synapse_xy_from_local`). Single buffer + `__syncthreads()` before and after each `(k_row, k_col)` filter stage.
- Launcher `convolution_kernel<64, 4>`. Conv1 (Nn=64) runs with `i_groups=1`; Conv2 (Nn=512) runs with `i_groups=8`.
- ptxas: 71 regs/thread, 39424 bytes SMEM, **0 spill stores/loads**.

**Correctness.** All four `CONV_VERIFY=1` cases pass (`max_abs ≈ 2e-6` Conv1, ≈ 6e-6 Conv2). Bit-identical pattern to the baseline — the WMMA itself is unchanged, only the loop nest around it.

**ncu deltas (B=16):**

| Metric | Conv1 baseline | Conv1 Iter 1 | Δ | Conv2 baseline | Conv2 Iter 1 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Kernel duration | 11.55 ms | **8.41 ms** | **−27%** | 12.56 ms | **7.76 ms** | **−38%** |
| Achieved GFLOPS | 5,123 | **7,034** | +37% | 1,178 | **1,907** | +62% |
| Tensor pipe util | 5.19% | 7.12% | +37% rel. | 1.56% | 2.52% | +62% rel. |
| L1 traffic | 2170.8 MB | **1429.0 MB** | **−34%** | 650.1 MB | **231.7 MB** | **−64%** |
| L1/TEX SoL | 68.42% | **47.59%** | **−20.8 pp** | 50.05% | **40.22%** | **−9.8 pp** |
| L2 traffic | 1489.1 MB | 1194.3 MB | −20% | 499.3 MB | 231.5 MB | −54% |
| DRAM read | 401.6 MB | 274.3 MB | −32% | 174.6 MB | 47.2 MB | **−73%** |
| DRAM write | 231.7 MB | **595.7 MB** | **+157%** ⚠ | 9.6 MB | 19.6 MB | +104% |
| DRAM SoL | 8.39% | 15.73% | +7.3 pp | 2.23% | 1.29% | −0.9 pp |
| Compute SM SoL | 15.05% | 12.81% | −2.2 pp | 4.39% | 4.29% | flat |
| HMMA insts | 115.6 M | 115.6 M | 0% | 37.7 M | 37.7 M | 0% |

**Conclusion.**
- The L1 thrashing fix landed exactly as the roofline doc predicted. L1 SoL on Conv1 dropped from 68% → 48%; Conv1 input bytes from DRAM dropped 32%, Conv2 input bytes dropped 73%.
- HMMA instructions are unchanged — so the speedup is *purely* from delivering the same math operations sooner, which is the textbook "fix the data path, not the math" win.
- Conv2 sees a bigger relative speedup because it had the worst input-reload amplification (32×). Its grid is still only 32 blocks (the [16 batches × 2 y-tiles × 1 x-tile]) so SM occupancy is the next ceiling — that's what Iter 3 targets.
- **Surprise:** Conv1 DRAM **write** spiked 2.6×. Output is 205 MB but DRAM write is now 595 MB. Hypothesis: the new epilogue writes one 16-channel slice at a time × NN_GROUP=4 slices per output pixel. Each pixel's 64 output channels span 2× 128-byte cache lines (32 floats = 1 line). NN_GROUP=4 slices means each cache line is touched twice per block, separated by `__syncthreads()` and a `wmma::store_matrix_sync` (which writes to SMEM, so it's not the issue per se, but it sequences the writes). With L2 = 4.5 MB and ~16K cache lines in flight across concurrent blocks, lines evict between the two partial writes, causing read-modify-write. Compute throughput already dropped slightly (15% → 13%) suggesting writeback contention is real. **However**, the kernel is still faster overall, so we accept this trade for now and consider epilogue restructuring as a candidate for a later iteration.

**Bottleneck after Iter 1:**
- Conv1: L1 still the dominant SoL (48%) but tensor pipe at only 7%. Most cycles are spent waiting for SMEM data. Likely sub-bottlenecks: SMEM bank conflicts, sync stalls, instruction count from filter staging.
- Conv2: L1 at 40%, but the real ceiling is launch parallelism — 32 blocks ≪ 80 SMs.

**Next iteration plan.** Probe with `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` and `smsp__average_warp_latency_per_inst_issued` to decide between (a) SMEM padding to remove bank conflicts, (b) staging multiple `(k_row, k_col)` filter slices at once to cut sync count.

---

## Iteration 2 — SMEM padding for bank-conflict reduction

**Probe.** Before writing Iter 2, ran `ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,smsp__warps_active.avg.pct_of_peak_sustained_active` on the Iter 1 binary:

| | Conv1 B=16 | Conv2 B=16 |
|---|---:|---:|
| SMEM load bank conflicts | **275 M** | **89.7 M** |
| Warps active | 24.85% | 12.50% |

Conflicts of that magnitude are the dominant L1 cost — at ~1 cycle each, 275M conflicts on 80 SMs × 1.2 GHz ≈ 2.9 ms of an 8.4 ms run. Warps active is also low (only 1 of 4 warp slots firing on average).

**Hypothesis.** The SMEM tiles use row stride `NI_CHUNK = 64` halfs = 128 bytes = exactly 32 4-byte banks. So consecutive WMMA-tile rows (16-row stride) all hit the same bank set — a 16-way bank conflict per `wmma::load_matrix_sync`. WMMA requires `ld * sizeof(half)` to be a multiple of 16 bytes (`ld` multiple of 8 halfs), so the smallest legal pad that breaks the alignment is 8 halfs: stride 72 halfs = 36 banks → mod 32 = 4. This converts 16-way conflicts into 2-way conflicts (8× reduction, the best achievable under the WMMA alignment constraint).

**Change** ([conv_opt.cu](conv_opt.cu)):
- Added `#define NI_PAD 8` and `constexpr NI_STRIDE = NI_CHUNK + NI_PAD = 72`.
- SMEM declarations switched from `[... * NI_CHUNK]` to `[... * NI_STRIDE]` (input tile, filter tile).
- `load_neuron_i_tile` and `load_synapse_group_smem` write to `(... * NI_STRIDE + ni)` so the padding halfs are skipped per row.
- `wmma::load_matrix_sync` `ld` arg switched from `NI_CHUNK` to `NI_STRIDE` (both toeplitz and filter loads).
- Toeplitz row base shifts: `... * 18 * NI_STRIDE + ... * NI_STRIDE` (was `* NI_CHUNK`).
- Filter offset uses `n * (16 * NI_STRIDE)` (was `* NI_CHUNK`).
- ptxas: 71 regs/thread (unchanged), **43,328 bytes SMEM** (+3,904 over Iter 1), 0 spills.

**Correctness.** All four `CONV_VERIFY=1` cases pass with bit-identical `max_abs` to Iter 1 — the padding halfs are written but never read, so output is unchanged.

**ncu deltas (B=16):**

| Metric | Conv1 Iter 1 | Conv1 Iter 2 | Δ | Conv2 Iter 1 | Conv2 Iter 2 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Kernel duration | 8.41 ms | **5.72 ms** | **−32%** | 7.76 ms | **5.74 ms** | **−26%** |
| Achieved GFLOPS | 7,034 | **10,353** | +47% | 1,907 | **2,581** | +35% |
| **vs baseline** | 1.37× | **2.02×** | | 1.62× | **2.19×** | |
| Tensor pipe util | 7.12% | **10.54%** | +48% rel. | 2.52% | **3.41%** | +35% rel. |
| **Bank conflicts (load)** | 275.2 M | **15.2 M** | **−94.5%** (18×) | 89.7 M | **4.78 M** | **−94.7%** (18×) |
| L1 traffic | 1429 MB | 1429 MB | flat (logical loads same) | 232 MB | 232 MB | flat |
| L1/TEX SoL | 47.59% | **22.35%** | **−25.2 pp** | 40.22% | **16.04%** | **−24.2 pp** |
| DRAM read | 274.3 MB | 265.9 MB | −3% | 47.2 MB | 47.2 MB | flat |
| DRAM write | 595.7 MB | 608.7 MB | +2% (still inflated) | 19.6 MB | 19.6 MB | flat |
| Compute SM SoL | 12.81% | **20.50%** | +7.7 pp | 4.29% | 6.31% | +2.0 pp |
| Warps active | 24.85% | 24.83% | flat | 12.50% | 12.50% | flat |

The reduction came in better than expected — bank conflicts dropped 18× (vs the ~8× I projected from a 16-way → 2-way analysis). I think the over-shoot is because some accesses were *worse* than 16-way (sub-warp non-uniformity), and the padding fixes those too. **L1 traffic is unchanged because the logical load count didn't change** — what changed is that each load now completes in 1 transaction instead of being serialized into many.

**Conclusion.** Bank conflicts were the dominant L1 cost. Removing them gave the largest single-iteration speedup so far (1.47× / 1.35× over Iter 1). Tensor pipe util doubled on Conv1.

**Bottleneck after Iter 2:**
- **Conv1**: Now relatively balanced. L1 22%, DRAM 23%, Compute 21%. Tensor pipe at 10.5% — still 89% of cycles waiting. The remaining stall is most likely **barrier waits** (`__syncthreads()` ≈ 19 per ic_base) since L1 and DRAM are no longer saturated. ncu's automatic rule shifted from "Memory more heavily utilized than Compute" (L1 bottleneck) to "**low compute throughput and memory bandwidth utilization … typically indicate latency issues**" — i.e., neither memory nor compute is full; warps are stalled.
- **Conv2**: ncu's rule is now "**kernel grid is too small to fill the available resources**" — only **0.2 full waves** across 80 SMs. With 32 blocks and ≤2 blocks/SM, only ~20% of SMs are doing useful work. The bank-conflict fix reduced the wait time per warp, but the same 32-block grid is still bottlenecked by parallelism.

The two bottlenecks are now different per layer, so Iter 3 will target Conv2's grid problem (the larger relative win) while leaving Conv1 alone.

