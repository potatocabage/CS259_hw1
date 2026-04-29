# Roofline analysis — `conv.cu` B=16 on TITAN V

> **Beginner's Introduction to Roofline Analysis**
> If you are new to GPU profiling, the **Roofline Model** is a visual tool used to understand the performance bottlenecks of a piece of code (a "kernel"). It compares how fast your kernel is doing math (Compute, measured in GFLOPS) against how much data it needs to move from memory (Memory Bandwidth, measured in GB/s). 
> 
> The core metric is **Arithmetic Intensity (AI)**, measured in FLOPs per byte. It tells you how much math you do for every byte of data you read/write to memory. 
> - **Memory-Bound (Sloped Roof):** If your AI is low, your kernel is starving for data. Performance is limited by how fast memory can deliver it.
> - **Compute-Bound (Flat Roof):** If your AI is high, you have plenty of data, and performance is limited by how fast the GPU can crunch the numbers.
> - **Ridge Point:** The AI where the hardware transitions from being memory-bound to compute-bound.

## Hardware (NVIDIA TITAN V, sm_70)

| Spec | Value |
|---|---|
| Tensor Core peak (FP16 mul, FP32 accum)¹ | **110 TFLOPS** |
| FP32 peak (vector, no tensor) | 14.9 TFLOPS |
| HBM2 bandwidth | **652.8 GB/s** |
| L2 cache | 4.5 MB |
| SMs | 80 |
| Tensor ridge point (Tensor peak ÷ HBM2 BW) | **≈ 168 FLOP/byte** |
| FP32 ridge point | ≈ 22.8 FLOP/byte |

**Hardware Terminology:**
* **Tensor Cores:** Specialized processing units on NVIDIA GPUs designed specifically to do extremely fast matrix multiplications (like those in AI/deep learning). They operate much faster than standard FP32 (32-bit floating point) units.
* **HBM2 Bandwidth:** High Bandwidth Memory. This is the main "global memory" or VRAM of the GPU. The bandwidth tells us the maximum speed data can be transferred to the processing cores.
* **L2 Cache:** A small, ultra-fast memory on the GPU that stores frequently accessed data to prevent slow trips to the main HBM2 memory.
* **SMs (Streaming Multiprocessors):** The independent processing blocks on the GPU. The TITAN V has 80 of them.
* **Ridge Point (168 FLOP/byte):** For every byte of data we read from main memory, we need to do at least 168 math operations (FLOPs) to keep the Tensor Cores fully busy. If we do fewer, we are limited by memory speed.

¹ NVIDIA TITAN V product page; the 110 TFLOPS figure is the marketing peak at
boost clock. ncu profiles at base clock by default (~1.2 GHz vs. 1.455 GHz
boost), so the *sustainable* peak under the profiler is closer to ≈ 90 TFLOPS.
The 110 TFLOPS roofline is still the right ceiling for the design analysis;
the gap just shows up as headroom even before clock effects.

The kernel uses WMMA tensor cores ([conv.cu:193-244](conv.cu#L193-L244)), so the
relevant compute roof is the **tensor** roof, not the FP32 roof.

## Per-kernel measurements (ncu)

Source: [ncu_metrics.csv](ncu_metrics.csv) (kernel ID 1 = Conv1 B=16, ID 3 = Conv2
B=16). FLOPs from [conv_flop.md](conv_flop.md).

| Quantity | Conv1 B=16 | Conv2 B=16 |
|---|---:|---:|
| Algorithmic FLOPs | 59.19 G | 14.80 G |
| Theoretical min input bytes (FP16, half) | 51.38 MB | 3.21 MB |
| Theoretical min synapse bytes (FP16) | 73.7 KB | 4.72 MB |
| Theoretical min output bytes (FP32) | 102.76 MB | 6.42 MB |
| **Theoretical min total bytes** | **154.21 MB** | **14.35 MB** |
| **Theoretical AI** | **383.8 FLOP/B** | **1031 FLOP/B** |
| `dram__bytes_read.sum` | 403.98 MB | 174.04 MB |
| `dram__bytes_write.sum` | 233.14 MB | 10.22 MB |
| **Measured DRAM bytes** | **637.12 MB** | **184.27 MB** |
| **Achieved AI (DRAM)** | **92.9 FLOP/B** | **80.3 FLOP/B** |
| L2 traffic (`lts__t_bytes.sum`) | 1490 MB | 499 MB |
| L1 traffic (`l1tex__t_bytes.sum`) | 2171 MB | 650 MB |
| Kernel duration (`gpu__time_duration.sum`) | 11.55 ms | 12.58 ms |
| Native HMMA inst. issued | 115.6 M | 37.7 M |
| FLOPs from HMMA × 512 (sanity) | 59.19 G ✓ | 19.33 G (1.31× alg.) |
| **Achieved GFLOPS (algorithmic)** | **5,125** | **1,176** |
| % of 110 TFLOPS tensor peak | **4.66 %** | **1.07 %** |
| Tensor pipe utilization (ncu) | 5.19 % | 1.56 % |

**Measurement Terminology:**
* **Algorithmic FLOPs:** The true number of useful math operations needed for the convolution, irrespective of hardware implementation.
* **Theoretical min total bytes:** If we had a perfect cache and only read/wrote every required piece of data exactly once from main memory, this is how many bytes we would transfer.
* **Theoretical AI:** `Algorithmic FLOPs / Theoretical min total bytes`. Because these numbers (383.8 and 1031) are higher than the Ridge Point (168), the algorithm *should* be compute-bound.
* **Measured DRAM bytes:** How much data our kernel *actually* moved to/from main memory during execution, measured by Nsight Compute (`ncu`). Notice this is much higher than the theoretical minimum!
* **Achieved AI (DRAM):** `Algorithmic FLOPs / Measured DRAM bytes`. Because we moved more data than necessary, our actual AI dropped to 92.9 and 80.3. Since these are *below* the 168 ridge point, our implementation is operating in the memory-bound region relative to DRAM.
* **Achieved GFLOPS:** How many billion operations per second we actually achieved. 5,125 GFLOPS is roughly 5.1 TFLOPS, which is less than 5% of the GPU's peak capacity.

`SpeedOfLight` percentages from [ncu_sol.csv](ncu_sol.csv):

> **Speed of Light (SoL):** This tells you what percentage of the hardware's maximum possible performance you are achieving in different areas (Memory, Compute, Caches). 100% means you have perfectly saturated that part of the hardware.

| SoL throughput | Conv1 B=16 | Conv2 B=16 |
|---|---:|---:|
| DRAM throughput | 8.37 % | 2.22 % |
| L1/TEX cache throughput | **68.61 %** | **50.06 %** |
| L2 cache throughput | 7.34 % | 2.26 % |
| Compute (SM) throughput | 15.12 % | 4.39 % |
| Memory throughput (max of L1/L2/DRAM) | 67.64 % | 19.94 % |

ncu's automatic bottleneck rule for both kernels:
> *"Memory is more heavily utilized than Compute: Look at the Memory Workload
> Analysis section to identify the L1 bottleneck."*

## Roofline placement

See [roofline.svg](roofline.svg) (regenerated from `roofline_plot.py`).

The 110 TFLOPS / 652.8 GB/s ridge point sits at **≈ 168 FLOP/byte**. Both kernels
are theoretically far above the ridge → "compute-bound" by design. But measured
DRAM AI puts them *below* the ridge:

| Kernel | Theoretical AI | Achieved DRAM AI | Achieved GFLOPS | Roof at achieved AI | Frac. of roof |
|---|---:|---:|---:|---:|---:|
| Conv1 B=16 | 383.8 | 92.9  | 5,125 | min(110 T, 92.9 × 652.8 GB/s) = **60.7 TFLOPS** | 8.4 % |
| Conv2 B=16 | 1031.0 | 80.3 | 1,176 | min(110 T, 80.3 × 652.8 GB/s) = **52.4 TFLOPS** | 2.2 % |

**Implications of the Placement:**
Both achieved points sit well **below** even the bandwidth-bound sloped roof at their
own AI. This means the kernels are leaving an order of magnitude of performance on the table, even after
accounting for the heavy DRAM traffic. 
Why? Because the standard one-level (DRAM) roofline does not
fully explain the bottleneck. The actual ceiling limiting performance is **L1 cache traffic**. The L1 cache is working incredibly hard (at 68.6% and 50.1% of its peak throughput), starving the math cores (Compute SM throughput is only 15% and 4%).

## Theoretical vs. achieved AI — what the gap means

Why did our AI drop from ~400+ to ~90? We suffer from **DRAM Amplification** — we are reading the same data from slow main memory multiple times instead of keeping it in fast caches.

| Kernel | Theoretical | Achieved (DRAM) | Ratio |
|---|---:|---:|---:|
| Conv1 B=16 | 384 FLOP/B | 93 FLOP/B | DRAM transfers 4.1 × the minimum |
| Conv2 B=16 | 1031 FLOP/B | 80 FLOP/B | DRAM transfers 12.8 × the minimum |

For **Conv1**, the 4.1× DRAM amplification has two contributors:
- **Reuse Failure:** Each input element is conceptually read once and reused KY·KX = 9 times across
  the kernel window and Nn = 64 times across output channels. The kernel splits
  Ni = 64 into a single 64-wide chunk (`NI_CHUNK=64` at [conv.cu:363](conv.cu#L363)),
  so synapse and input are loaded **once per output tile**, not once total. With
  28×14 = 392 output blocks per batch × 16 batches = 6272 blocks, each fetching
  its own 10×18×64 input tile, a lot of input is re-read from DRAM that L2 (4.5
  MB) can't hold.
- **No Padding Overhead:** The `(NXSCL_pad=224, NYSCL_pad=224)` tiling exactly matches the unpadded
  spatial dims for Conv1, so there is no padding overhead in this case (the
  HMMA count matches algorithmic FLOPs exactly).

For **Conv2**, the 12.8× DRAM amplification is much worse, but the working set
fits in L2: input + synapse + output ≈ 14.4 MB total, and L2 is 4.5 MB. ncu shows
DRAM throughput at only 2.2 %, while L1 throughput is at 50 %. So DRAM bytes
here are dominated by cold-cache fills and replay overhead more than reuse
failure. Two structural inefficiencies still apply:
- **WMMA Padding Overhead:** Tensor cores operate on fixed tile sizes (like 16x16). The output is padded from 14×14 to 16×16 for the WMMA tile shape
  ([conv.cu:307](conv.cu#L307)), so the kernel computes 16²/14² = **1.31×** as many
  HMMAs as algorithmically required (confirmed by `sm__inst_executed_pipe_tensor_op_hmma.sum
  × 512 = 19.33 GFLOPs > 14.80 GFLOPs algorithmic`).
- **L1 Cache Thrashing:** With Ni = 512 split into NI_CHUNK = 64 sub-chunks, the synapse is reloaded
  Nn/16 = 32 times *per* (k_row, k_col) per ic_base, and the input tile is
  reloaded 8 times per ic_base. The L1 traffic / DRAM traffic ratio is 3.5 ×,
  meaning L1 absorbs most reuse — but L1 itself is the saturated stage.

## Compute-bound or bandwidth-bound?

By **theoretical AI**, both kernels *should* be compute-bound (well above the tensor
ridge of 168 FLOP/B).

By **achieved AI**, both are nominally below the ridge. However, looking at the Speed of Light (SoL) section
makes the actual bottleneck clearer: it is **L1 cache throughput**, not DRAM
bandwidth. DRAM throughput is only 8 % (Conv1) and 2 % (Conv2) of peak, while
L1/TEX throughput is at 68 % and 50 %. 

**Conclusion:**
- **Conv1 B=16**: **Cache-bound** (specifically L1 throughput-limited). It also has non-trivial DRAM traffic because our tiling strategy doesn't fit well in the L2 cache.
- **Conv2 B=16**: **Cache-bound** with very low overall utilization. The working set fits in L2, but the kernel code issues so many redundant shared-memory and L1 data load instructions per math operation that the Tensor Cores spend most of their cycles just waiting for data to arrive from L1.

## Optimization headroom

There is massive room for improvement. Conv1 can theoretically run **roughly 12×** faster just hitting the memory bandwidth limit at its achieved AI (5.1 → 60 TFLOPS), and **roughly 21×** faster to the tensor peak (5.1 → 110 TFLOPS). Conv2 has even more headroom (45x to 90x).

**How to optimize this kernel (in simpler terms):**

1. **Improve achieved AI by reducing DRAM re-reads (Fix the Tiling).** The current kernel loads a large chunk of input data into shared memory for every output block. If we change how we divide the work (larger output tiles, or using cooperative thread groups), threads could share data better, meaning we fetch from slow DRAM less often.
2. **Reduce L1/SMEM pressure between iterations.** On Volta, the L1 cache and Shared Memory are physically the exact same hardware block. High `l1tex` traffic means you are issuing too many Shared Memory reads (`ld.shared`) and L1 cache hits. 
    * **The "Abuse":** Look at your loops. `load_neuron_i_tile` (Global $\rightarrow$ SMEM) is *inside* the `i` loop (output channels). For Conv2, `i` loops 32 times. This means you are reloading the exact same input tile from Global Memory into SMEM 32 times per block. The L1 cache gracefully catches these 32 global reads (so you don't hit DRAM), but it generates massive L1 traffic. Furthermore, for every single output channel chunk, you re-read the exact same `toeplitz_frag` from SMEM into registers.
    * **The Fix:** You need your warps to compute more than one output channel chunk at a time. If a warp computes a 2x2 grid of output tiles simultaneously, you can load the input from SMEM into registers *once* and reuse it for multiple output chunks, slashing `ld.shared` instructions.
3. **Pad the *input*, not the output, for Conv2.** Because we padded the 14x14 output to 16x16 to satisfy the Tensor Core shape requirements, we inflated FLOPs by 31%. Finding a way to handle the 14x14 boundaries without padding the compute (e.g., a fused 14x14 tail kernel) would instantly reduce wasted work.
4. **Reduce synchronization stalls.** The compute cores are mostly idle (4-15% active). The current "double-buffer" code on Volta uses `__syncthreads()` multiple times per inner loop to safely swap buffers. This forces all warps to wait. To hide this latency on Volta, you typically need to unroll the loop, increase the buffer depth, or reorder instructions so math happens while the next load is actively being serviced.

In short: the math tells us this algorithm *should* run blazingly fast. However, the current code moves data back and forth between caches (especially L1/SMEM) so inefficiently that the math cores are starving. Closing the AI gap (theoretical → achieved) by fixing how data is reused in L1 and Shared Memory is the key to unlocking the GPU's power.

## Reproducing

```bash
make conv_cuda
TMPDIR=$HOME/.ncu_tmp ncu -k regex:convolution_kernel \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__inst_executed_pipe_tensor_op_hmma.sum,l1tex__t_bytes.sum,lts__t_bytes.sum \
    --csv --log-file ncu_metrics.csv ./conv_cuda

TMPDIR=$HOME/.ncu_tmp ncu --section SpeedOfLight -k regex:convolution_kernel \
    --csv --log-file ncu_sol.csv ./conv_cuda

python3 roofline_plot.py     # writes roofline.svg
```

See [ncu_guide.md](ncu_guide.md) for the step-by-step explanation of these
commands.
