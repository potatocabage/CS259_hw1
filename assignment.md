# CS 259 Mini-Project 1: GPU Kernel Optimization

**Due:** April 28th, EOD  
**Group size:** 1–4 (recommended: 2)

---

## Summary

The goal of this assignment is to gain hands-on experience with GPU programming and
to understand the compute and memory characteristics of two important deep learning
kernels: convolution and attention.

You will implement CUDA kernels for both, profile their performance, and analyze
where they fall on the roofline model.

---

## Getting Started

You should have received an email with account access to the course GPU machine.
Make sure `/usr/local/cuda/bin` is in your `PATH`.

A CPU reference implementation for both kernels is provided in this repository.
Build and run it to see expected outputs and get familiar with the data layouts:

```bash
make
./conv
./attention
```

For an introduction to CUDA, this is a good starting point:  
https://developer.nvidia.com/blog/even-easier-introduction-cuda/

The official programming guide covers GPU features in depth:  
https://docs.nvidia.com/cuda/cuda-c-programming-guide/

To query your GPU's parameters (peak FLOPS, memory bandwidth, SM count, etc.):

```bash
./Samples/1_Utilities/deviceQuery/deviceQuery
```

---

## Task

Implement CUDA kernels for the following two workloads.

### Part 1 — Convolution

Implement a CUDA kernel for direct 2D convolution with the following VGG configurations:

| Layer | Ny × Nx | Ky × Kx | Ni  | Nn  | Stride | Batch |
|-------|---------|---------|-----|-----|--------|-------|
| Conv1 | 224×224 | 3×3     | 64  | 64  | 1      | 1, 16 |
| Conv2 | 14×14   | 3×3     | 512 | 512 | 1      | 1, 16 |

The data layout used in the reference implementation is:
- Weights: `[Ky][Kx][Nn][Ni]`
- Input:   `[B][NYPAD][NXPAD][Ni]` (zero-padded)
- Output:  `[B][NYSCL][NXSCL][Nn]`

You are free to change the data layout, use any CUDA features (shared memory, tensor
cores, warp-level intrinsics), or restructure the computation.

### Part 2 — Attention

Implement CUDA kernels for single-head attention with head dimension D=64, for both
the **prefill** case (S queries attending to S keys causally) and the **decode** case
(1 query attending to a KV cache of size C):

| Case    | Sizes         |
|---------|---------------|
| Prefill | S = 4096, 65536 |
| Decode  | C = 4096, 65536 |

The reference implementation provides `standard_prefill`, `flash_prefill`, and
`standard_decode`. You may implement either standard or flash attention for your GPU
kernel. Note that for the larger context size (S=65536), standard attention may not
fit in GPU memory — the provided flash attention implementation may serve as a useful
reference in that case.

---

## Questions

Answer the following questions **for each kernel** (Conv1, Conv2, prefill, decode).
Where configurations differ (e.g., batch size or context length), report results for
all and discuss any differences.

### 1. Parallelization Strategy

What is your parallelization strategy? What problem dimensions did you map to
blocks and threads? Are there limitations to this approach — for example, does it
scale well with batch size, context length, or channel count?

### 2. Algorithmic FLOPs

Compute the algorithmic FLOP count for each configuration. Show your derivation.

For convolution, count multiply-add operations over the kernel and channel dimensions.
For attention, account for the QK dot products and the weighted sum over values —
note that prefill involves a triangular (causal) access pattern.

Report your results in GFLOPs.

### 3. Execution Time and Throughput

What is the measured execution time for each configuration? What is the achieved
GFLOPS? How does this compare to the GPU's peak throughput?

### 4. Roofline Analysis

Plot your results on the roofline model for your GPU. For each kernel configuration:

- Estimate the arithmetic intensity (FLOPs / bytes transferred to/from DRAM)
- Identify whether the kernel is compute-bound or memory-bandwidth-bound
- Mark where it falls on the roofline

What does this tell you about the potential for further optimization?

### 5. Optimizations

What optimizations did you try? Which had the most impact, and which had little or
no effect? If an optimization did not help, explain why.

---

## What to Turn In

- A PDF report answering the questions above, including roofline plots
- Your CUDA source file(s) as a separate attachment (do not zip with the PDF)

Turn in via Canvas using your group submission.
