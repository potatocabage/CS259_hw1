# cs259-miniproj-ref

CPU reference implementations for the CS 259 mini-project.

## Contents

| File | Description |
|------|-------------|
| `attention.cpp` | Standard and flash attention (prefill + decode) |
| `conv.cpp` | Direct convolution (VGG Conv1 and Conv2) |
| `timing.h` | Timing helper used by both |

## Build and run

```bash
make        # builds ./attention and ./conv
./attention
./conv
```

Requires GCC with OpenMP (`-fopenmp`) and C++17.

## Configurations

**Attention** — head dimension D=64:

| Case | S or C |
|------|--------|
| Prefill | 4096, 65536 |
| Decode  | 4096, 65536 |

`attention.cpp` provides three implementations:
- `standard_prefill` — two-pass softmax, allocates an O(S) scores buffer per query
- `flash_prefill` — online softmax, single pass, no scores buffer
- `standard_decode` — two-pass softmax for a single query vs KV cache

For S=65536, `standard_prefill` is skipped (O(S²) — too slow).
Flash attention is provided as a reference; you may adapt it for your GPU kernel.

**Convolution** — 3×3 kernel, stride 1:

| Layer | Ny × Nx | Ni | Nn | B |
|-------|---------|----|----|---|
| Conv1 | 224×224 | 64 | 64 | 1, 16 |
| Conv2 | 14×14 | 512 | 512 | 1, 16 |


## Assignment questions

1. Parallelization Strategy
What is your parallelization strategy? What problem dimensions did you map to blocks and threads? Are there limitations to this approach — for example, does it scale well with batch size, context length, or channel count?

2. Algorithmic FLOPs
Compute the algorithmic FLOP count for each configuration. Show your derivation.

For convolution, count multiply-add operations over the kernel and channel dimensions. For attention, account for the QK dot products and the weighted sum over values — note that prefill involves a triangular (causal) access pattern.

Report your results in GFLOPs.

3. Execution Time
What is the measured execution time for each configuration? What is the achieved GFLOPS (using your algorithmic FLOP count from Q2)?

4. Roofline Analysis
Plot your results on the roofline model for your GPU. For each kernel configuration:

Compute the theoretical arithmetic intensity (algorithmic FLOPs / minimum bytes required to read inputs and write outputs once)
Measure the actual DRAM traffic using ncu (dram__bytes_read.sum, dram__bytes_write.sum) and compute the achieved arithmetic intensity from that
Place both on the roofline and identify whether the kernel is compute-bound or memory-bandwidth-bound
How do the theoretical and measured arithmetic intensities compare? What does the difference tell you, and what does your roofline placement say about the potential for further optimization?

5. Optimizations
What optimizations did you try? Which had the most impact, and which had little or no effect? If an optimization did not help, explain why.

What to Turn In
A PDF report answering the questions above, including roofline plots
Your CUDA source file(s) as a separate attachment (do not zip with the PDF)
Turn in via Canvas using your group submission.