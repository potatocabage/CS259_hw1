Members: Nathan Wei (105762362), Pritam Mukhopadhyaya (205698479)  
COM SCI 259 Spring 2026

# Mini-Project 1 Report Questions

1. #  Parallelization Strategy 

What is your parallelization strategy? What problem dimensions did you map to blocks and threads? Are there limitations to this approach — for example, does it scale well with batch size, context length, or channel count?

### Conv

For the conv kernel, we decided to challenge ourselves by writing an implicit GEMM kernel that utilized the tensor core. Each warp handled all channels for 16 pixels in the output spatial row and each threadblock consisted of 8 warps (so 8x16 per threadblock) in order to saturate SMEM and the tensor cores. Each 16x16x16 tensor core operation calculates a 16x16 partial sum of a N x Cout x (Ky x Kx x Cin) im2col matrix multiplication. We structure our loops around this.

The loops go as follows:  
For chunks of 16 output channels:  
	For chunks of 64 (template toggleable) input channels:  
		Load input chunk  
		For each spatial pixel in the kernel  
			Load kernel slice  
			Calculate im2col with wmma  
	Write to global mem

We also double buffer the loading of the kernel slice.  
This design unfortunately suffered from L1 thrashing, which we will address later.  
Algorithmically, the design suffers from not saving old multiplications of kernel element x input pixel, which leads to heavy recalculation. It also scales poorly with output channels since it requires multiple loadings of redundant input chunks for each output channel.  
For conv2, we pad the output to be 16x16 spatially. We acknowledge that it is possible to keep it 14x14 because the halo extends it, but due to time constraints, we pad naively.  
In order to use the half precision tensor cores, we have a float to half conversion kernel that we will omit in our calculations.

### Standard Prefill

Our primary parallelization strategy is row parallel where each block (grid dimension \= S) outputs a single output row and each thread (block dimension \= D) outputs one feature of an output row. We also incorporate tiling such that each block considers D keys for a query in parallel as the block sequentially progresses through its assigned row in tiles of size D to compute the scores. We then do a reduction step in which we determine the max score and sum per tile before performing an online softmax trick where we can track the running max to rescale the weights and track the weighted sum to rescale afterwards.  
While we can fuse the different states of self attention prefill into one kernel and avoid materializing an S x S matrix, our strategy has its own limitations. Since we consider causal masking and skip dot product computations when attending the ith query to the jth key where j \> i, the sequential workload between different blocks is not balanced. For example, the 1st block would sequentially process one tile while the (S-1)th block would sequentially process S/D tiles.

### Standard Decode

We do a single block (grid dimension \= 1\) where each thread (block dimension \= D) computes one output feature. These threads then sequentially process along the context length. Specifically, we process along the context dimension in tiles of size D for a total of C/D iterations. This means we first take the dot product between the current query against D keys. Next, we perform a reduction to determine the max score and sum for each of these tiles. Then, we perform an online softmax calculation using the same trick described in standard prefill.  
The key limitation of this approach is that it requires each thread to do sequential work that scales linearly with the context length C. This is because the online softmax trick, as implemented, adds sequential dependencies. As a consequence, as C \>\> D we perform significantly more work sequentially than in parallel. It is worth noting, however, that the memory scaling is not as affected by the context length and is bounded by the tile size.

2. #  Algorithmic FLOPS

Compute the algorithmic FLOP count for each configuration. Show your derivation. For convolution, count multiply-add operations over the kernel and channel dimensions. For attention, account for the QK dot products and the weighted sum over values — note that prefill involves a triangular (causal) access pattern. Report your results in GFLOPs.

### Conv

We will compute the algorithmic/theoretical FLOP count rather than the true FLOP count (with FLOP for index calculation, etc.). We will also treat multiply-add as 2 separate FLOPs. Since our kernel does not reuse any multiply-add results, our FLOP count is just the standard 2 \* B \* NYSCL \* NXSCL \* Ky \* Kx \* Ni \* Nn. This is 2\*16\*226\*226\*3\*3\*64\*64=60251701248 for conv1 and 2\*16\*18\*18\*3\*3\*256\*256=6115295232 for conv2 (extra padding).

### Standard Prefill

Let **Q** \= (S, D), **K** \= (S, D), **V** \= (S, D), **O** \= (S, D).   
Let score(i, j) be the score of row i of **Q** against row j of **K**. To compute score(i, j), we first take the dot product of qi and kj then normalize with the factor 1/sqrt(D) (can be precomputed once). To enforce causal relationships, score(i, j) is assigned \-infinity when j \> i, reducing the number of dot products needed. So, this leads to 1 \+ 2 \+ … \+ S \= S(S+1)/2 dot products with each dot product involving D pairs of multiply and add operations giving roughly a total of DS2 FLOPs for this step.  
We then perform a weighted sum of these scores over **V**. Specifically, we have to multiply and accumulate i values between the jth column of **V** for all i \<= j. This similarly leads to 1 \+ 2 \+ … \+ S \= S(S+1)/2 dot products where each dot product involves D pairs of multiply and add operations, giving roughly a total of DS2 FLOPs for this step.  
In total, when just accounting for the QK dot product and sum over values, we approximate 2DS2 FLOPs. With D \= 64, we approximate 2.15 GFLOPs and 5.50\*102 GFLOPs for S \= 4096 and S \= 65536 respectively.

### Standard Decode

Let **q** \= (D), **K** \= (C, D), **V** \= (C, D), **O** \= (C, D).   
	We must compute the score of query **q** against all rows of **K**. This means we must take C dot products which equates to 2\*D\*C floating point operations. Unlike prefill, we cannot exploit causality to reduce this number. For each of the D output features, we must accumulate C weighted values based on the scores. This gives us another 2\*D\*C floating point operations. This gives us a total of 4\*D\*C FLOPs. With D fixed at 64, our algorithmic FLOP counts are 1.05\*10\-3 GFLOPs and 1.68\*10\-1 GFLOPs for C \= 4096 and C \= 655336 respectively.

3. #  Execution Time

What is the measured execution time for each configuration? What is the achieved GFLOPS (using your algorithmic FLOP count from Q2)?

### Conv

We measure kernel-only time using ncu's `gpu__time_duration.sum` (the wall-clock `./conv_cuda` timing also folds in `cudaMalloc`, host→device copy, and the fp32→fp16 conversion kernel, which we omit per the assignment's "kernel launch + device sync" methodology). Achieved GFLOPS uses our algorithmic FLOP count from Q2 (59.19 GFLOPs for Conv1 B=16 and 14.80 GFLOPs for Conv2 B=16; values for B=1 and other configs scale linearly with B and use the unpadded 14×14 spatial extent for Conv2). Source CSVs: [ncu_metrics.csv](ncu_metrics.csv), [ncu_metrics_no_db.csv](ncu_metrics_no_db.csv).

| Config       | conv.cu (double-buffered) | conv\_no\_db.cu (single buffer) | Δ time |
|--------------|--------------------------:|--------------------------------:|-------:|
| Conv1 B=1    | 0.847 ms / 4366 GFLOPS    | 0.811 ms / 4563 GFLOPS          | −4.3%  |
| Conv1 B=16   | 11.555 ms / 5123 GFLOPS   | 11.192 ms / 5288 GFLOPS         | −3.1%  |
| Conv2 B=1    | 10.980 ms / 84 GFLOPS     | 10.434 ms / 89 GFLOPS           | −5.0%  |
| Conv2 B=16   | 12.581 ms / 1176 GFLOPS   | 12.004 ms / 1233 GFLOPS         | −4.6%  |

Removing the synapse double-buffering is a uniform 3–5% speedup across all four configs. As we discuss in Q4 and Q5, this is because the kernel is bound by L1/TEX throughput rather than load latency, so the prefetch's `global → regs → smem` staging path emitted L1 traffic that the simpler `global → smem` path avoids — DRAM traffic is essentially unchanged but L1/TEX bytes drop by ≈45%.

### Standard Prefill

We measure the execution time of the kernel from kernel launch and device synchronization, omitting the time to copy data to and from the device. We measure execution times of 11.45 ms and 2489.06 ms for S \= 4096 and S \= 65536 respectively. We then estimate the throughput as 187.77 GFLOPS and 220.97 GFLOPS for S \= 4096 and S \=  65536 by dividing the algorithmic FLOP count by the measured execution time.

### Standard Decode

We measure the execution time of the kernel from kernel launch and device synchronization, omitting the time to copy data to and from the device. We measure execution times of 1.23 ms and 9.49 ms for C \= 4096 and C \= 65536 respectively. We then estimate the throughput as 0.85 GFLOPS and 17.70 GFLOPS for C \= 4096 and C \=  65536 by dividing the algorithmic FLOP count by the measured execution time.

4. #  Roofline Analysis

Plot your results on the roofline model for your GPU. For each kernel configuration:

- Compute the theoretical arithmetic intensity (algorithmic FLOPs / minimum bytes required to read inputs and write outputs once)  
- Measure the actual DRAM traffic using ncu (dram\_\_bytes\_read.sum, dram\_\_bytes\_write.sum) and compute the achieved arithmetic intensity from that  
- Place both on the roofline and identify whether the kernel is compute-bound or memory-bandwidth-bound

How do the theoretical and measured arithmetic intensities compare? What does the difference tell you, and what does your roofline placement say about the potential for further optimization?

### Conv

Theoretical AI is the algorithm's lower bound — algorithmic FLOPs divided by the minimum total bytes of one-shot reads of the input (fp16) and synapse (fp16) plus one-shot writes of the output (fp32). Achieved AI uses ncu's `dram__bytes_read.sum + dram__bytes_write.sum`. Achieved GFLOPS uses ncu's `gpu__time_duration.sum`. All B=16 numbers below come straight from [ncu_metrics.csv](ncu_metrics.csv) and [ncu_metrics_no_db.csv](ncu_metrics_no_db.csv); the theoretical-byte breakdown is in [roofline.md](roofline.md).

| Kernel | Theoretical AI | DRAM bytes | Achieved AI | Achieved GFLOPS | Bound |
|---|---:|---:|---:|---:|---|
| Conv1 B=16, conv.cu (db)      | 383.8 FLOP/B | 637.12 MB | 92.9 FLOP/B  | 5123 | L1-throughput |
| Conv1 B=16, conv\_no\_db.cu   | 383.8 FLOP/B | 574.96 MB | 102.9 FLOP/B | 5288 | L1-throughput |
| Conv2 B=16, conv.cu (db)      | 1031 FLOP/B  | 184.27 MB | 80.3 FLOP/B  | 1176 | L1-throughput |
| Conv2 B=16, conv\_no\_db.cu   | 1031 FLOP/B  | 184.05 MB | 80.4 FLOP/B  | 1233 | L1-throughput |

By **theoretical AI** both kernels sit far above the TITAN V tensor ridge (≈ 168 FLOP/B), so the algorithm is compute-bound by design. By **achieved DRAM AI** they fall below the ridge (Conv1 by ≈4×, Conv2 by ≈13×) — DRAM traffic is much higher than the lower bound. The gap is dominated by reuse failure: with `NI_CHUNK=64` the synapse and input tiles are reloaded once per output-channel block (and Conv2's 14×14 output is padded to 16×16 for WMMA, inflating HMMAs by 1.31×). Both points fall **below even the bandwidth-bound sloped roof** at their own AI, which is the giveaway: the actual ceiling is **L1/TEX throughput**, not DRAM. ncu's SoL section confirms it — L1/TEX hits 50–69% of peak while DRAM is ≤8% and tensor-pipe utilization is ≤5% (see [ncu_sol.csv](ncu_sol.csv) / [ncu_sol_no_db.csv](ncu_sol_no_db.csv)). ncu's automatic bottleneck rule prints "Memory is more heavily utilized than Compute … look at the L1 bottleneck" for every config.

Removing the synapse double-buffering (`conv_no_db.cu`) cuts L1/TEX bytes by ~45% (the `global → regs → smem` path emitted measurable extra L1 activity per element), but **leaves DRAM traffic almost untouched** — confirming the prefetch was buying latency hiding the kernel didn't need. The achieved AI nudges up for Conv1 (92.9 → 102.9) because Conv1's working set spills out of the 4.5 MB L2 and DRAM reads are sensitive to the issue order; for Conv2 the working set already fits in L2, so DRAM is the same to within noise (80.3 → 80.4). In both cases the speedup (3–5%) is modest because L1 is still the binding stage.

The roofline plot ([roofline.svg](roofline.svg), regenerated via [roofline_plot.py](roofline_plot.py)) shows: theoretical points sit on the tensor ceiling, achieved points sit roughly an order of magnitude below the bandwidth-bound roof at their AI, and the no\_db variants land slightly to the right of and above the db variants. Conv1 has roughly 12× headroom to the bandwidth roof at its measured AI (5.1 → 60 TFLOPS) and ≈21× to tensor peak; Conv2 has even more (≈45–90×). Practical paths to closing the gap are larger output tiles (so a warp computes multiple output-channel chunks per loaded input tile, slashing `ld.shared`), padding the *input* (not output) for Conv2 to stop inflating HMMAs, and reordering the staging to keep the tensor pipe issuing rather than waiting on L1 — see Q5 for the ablations we tried.

### Standard Prefill

The minimum bytes required to read inputs and write outputs once is the total number of bytes across Q, K, V, and O. This total is (S \* D \+ S \* D \+ S \* D \+ S \* D) \* 4 bytes \= 16\*S\*D bytes. We define theoretical arithmetic intensity \= algorithmic FLOPs / minimum byte total \=  2DS2 / 16SD \= S/8. We obtain theoretical arithmetic intensities of 512 FLOPs/byte and 8192 FLOPs/byte for S \= 4096 and S \= 65536 respectively.  
	The achieved DRAM traffic reported by ncu (dram\_\_byes\_read.sum \+ dram\_\_bytes\_write.sum) is 4.39MB for S \= 4096 and 1.46TB for S \= 65536\. The achieved FLOP count reported by ncu (2\*smsp\_\_sass\_thread\_inst\_executed\_op\_ffma\_pred\_on.sum) is 2.39 GFLOPs for S \= 4096 and 609.97 GFLOPs for S \= 65536\. This results in achieved arithmetic intensities of 544 FLOP/byte and 0.418 FLOP/byte respectively for S \= 4096 and S \= 65536\.  
	The theoretical and achieved arithmetic intensities are fairly comparable for S \= 4096 but the achieved arithmetic intensity is roughly 10 times lower than its arithmetic counterpart for S \= 65536\. While the roofline plot shows our prefill implementation is still compute bound (right of the ridge point), our implementation suffers from rereading the lower triangular of the scores matrix which is a key limitation for larger S. This gap in the roofline plot can likely be reduced with further optimization.

### Standard Decode

The minimum bytes required to read inputs and write outputs once is the total number of bytes across Q, K, V and O. This total is (D \+ C\*D \+ C\*D \+ D) \* 4 bytes \= 8D(C \+ 1\) bytes or approximately 8DC bytes. We define theoretical arithmetic intensity \= algorithmic FLOPs / minimum byte total \= 4DC / 8DC \= 0.5 for both C \= 4096 and C \= 65536\.  
	The achieved DRAM traffic reported by ncu (dram\_\_byes\_read.sum \+ dram\_\_bytes\_write.sum) is 2.11MB for C \= 4096 and 35.01MB for C \= 65536\. The achieved FLOP count reported by ncu (2\*smsp\_\_sass\_thread\_inst\_executed\_op\_ffma\_pred\_on.sum) is 1.16 MFLOPs for C \= 4096 and 18.61 MFLOPs for C \= 65536\. This results in achieved arithmetic intensities of 0.55 FLOPs/byte and 0.53 respectively for C \= 4096 and C \= 65536\.  
	The theoretical and achieved arithmetic intensities are comparable for both C \= 4096 and C \= 65536\. This tells us that there is no real room for improvement for these configurations. The roofline plot shows that both theoretical and achieved arithmetic intensities are memory bound (left of the ridge point). This suggests that decode is inherently memory bound even with optimal implementations for specific configurations.

### 

### 

### 

### 

### 

### Standard Attention Roofline Plot

We were unable to generate the roofline plots directly with ncu and had to create it manually.

![][image1]

5. #  Optimizations

What optimizations did you try? Which had the most impact, and which had little or no effect? If an optimization did not help, explain why.

### Conv

As an ablation, we tried removing the double buffering. This halved the amount of reads/writes to L1 and actually increased the throughput by about 3-5%.

### Standard Prefill

We first started with a naive implementation where we planned to buffer the scores per row. We then had each row make three passes to compute the softmax scores: dot product, finding max/sum, and accumulating weighted values. While this exploited row parallelism like our current solution, it was bottlenecked by the scores buffer which would be proportional to S2. This buffer size would not work for S \= 65536, which motivated us to try a tiled approach that uses the online softmax trick. This approach lets us prefill with S \= 65536 which is already a huge improvement over naive.

### Standard Decode

We started with a naive implementation that involved three kernels to make three passes for computing scores, finding max/sum, and accumulating weighted values. It also used a buffer whose size was proportional to C. We then wanted to use something similar to the FlashAttention reference with online softmax. The online softmax trick lets us fuse these three passes into one kernel. This improved our measured execution time somewhat as the tiling implementation of online softmax removes the need for a global buffer to be accessed by separate kernels. After viewing the ridge plots, we decided it would not be worthwhile to pursue more optimizations.