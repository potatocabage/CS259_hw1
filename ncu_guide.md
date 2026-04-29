# Using `ncu` (Nsight Compute) for the convolution kernel

A practical walk-through of profiling [conv.cu](conv.cu) with `ncu`, written for
someone who hasn't used it before. It assumes you're on the eda cluster's
TITAN V machine with CUDA 12.4 already in `PATH`.

## 1. Build the binary

```bash
cd /usr/eda/CS251A/nwei/cs259-miniproj-ref
make conv_cuda
```

The [Makefile](Makefile) compiles with `nvcc -O3 -arch=sm_70 -std=c++17`.
`-arch=sm_70` targets Volta (TITAN V); the kernel uses `wmma::*` intrinsics, so
the binary is **not portable** to a GPU with a different compute capability —
re-`make` if you ever move to a different machine. Verify what you have:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
# NVIDIA TITAN V, 7.0
```

## 2. Run once *without* ncu

Always run the binary plain first, both to confirm it works and to record the
host wall-clock timings printed by [`bench()`](timing.h):

```bash
./conv_cuda
# === Convolution CUDA (WMMA) ===
#   Layer                                 ms      GFLOPS
#   Conv1-VGG  B=1                    2702.67     1.37
#   Conv1-VGG  B=16                   1213.23    48.79
#   ...
```

These numbers are what you should report for the README's Q3 ("execution
time"). They include host-side `cudaMalloc`/`Memcpy` and the FP32→FP16
conversion kernels — i.e. **everything `launch_convolution_layer` does**, not
just the WMMA kernel. ncu will give you the per-kernel time separately.

## 3. What `ncu` actually does

`ncu` is a **kernel-level profiler**. It does not measure your program's wall
clock. For each kernel launch it matches against your filter, it:

1. Pauses the launch.
2. Re-runs the kernel many times (one replay per group of metrics it can collect
   in a single pass), reading hardware perf counters between replays.
3. Reports a single per-launch number per metric (typically the median or sum
   across replays).

Consequences:
- The wall-clock printout from `./conv_cuda` is **meaningless under ncu**
  because each kernel has been re-run several times — that's why you saw 2.4 s
  for Conv1 B=16 under ncu vs. ~1.2 s plain.
- Per-kernel measurements like `gpu__time_duration.sum` and
  `dram__bytes_read.sum` *are* still per-launch and reliable.
- For kernel timing, **use ncu's `gpu__time_duration.sum`** (the kernel's own
  duration), not the host `bench()` time.

`nsys` (Nsight Systems) is a different tool — a tracing profiler for end-to-end
timelines, no replays, no counters. Use `nsys` if you want host+device timeline
visualization; use `ncu` if you want per-kernel hardware counters.

## 4. Permissions & environment quirks

`ncu` needs perf-counter access at the driver level (CAP_SYS_ADMIN-equivalent).
Common failure modes on this machine:

**`ERR_NVGPUCTRPERM`** — non-root users are blocked from reading GPU
performance counters. You'll see something like:

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 0. For instructions
on enabling permissions and to get more information see
https://developer.nvidia.com/ERR_NVGPUCTRPERM
```

*Background.* The NVIDIA driver gates perf-counter access behind a kernel-module
parameter, `NVreg_RestrictProfilingToAdminUsers`. It defaults to **1** since
driver 418.43 (April 2019, CVE-2018-6260 mitigation) — perf counters can leak
information across processes and users via timing side-channels, so non-admins
are blocked by default. Setting it to **0** opens counter access to all users.

*Quick diagnosis* — read the live driver setting (no root needed):

```bash
cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly
# RmProfilingAdminOnly: 1   ← restricted (you'll hit ERR_NVGPUCTRPERM)
# RmProfilingAdminOnly: 0   ← open (ncu works for non-root)
```

*Fix (requires root).* Persist the setting in modprobe config and reload the
driver:

```bash
echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" \
    | sudo tee /etc/modprobe.d/nvidia-profiling.conf

# Reload the module — only works if no process is currently using the GPU.
# Stop any CUDA workloads first; nvidia-smi will show what's running.
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia
# Or simpler: reboot.
```

After reload, re-check `RmProfilingAdminOnly: 0` and re-run ncu.

*Workarounds without root.*
- **Run ncu as root** (`sudo ncu ...`) — works on machines where you have sudo
  but the admin won't change the modprobe setting. Note this profiles the
  process you launch under sudo, so the binary runs as root too; fine for a
  benchmark binary, not fine if the program writes files to your home dir or
  expects your environment.
- **Containers / NGC images** — if you're inside a container that already has
  the perf-counter capability granted (e.g. `--cap-add SYS_ADMIN` or NVIDIA's
  NGC profiling images), this error doesn't apply.
- There is **no `LD_PRELOAD` or per-process workaround** — the check is in the
  kernel module, not user-space. If neither sudo nor an admin reload is
  available, you're stuck and have to ask the system admin.

On the eda cluster's TITAN V machine the setting is already 0 (otherwise none
of the commands in this guide would have worked), so this is informational —
useful if you ever try to reproduce on another box.

**`Failed to open/create lock file /tmp/nsight-compute-lock`** — another user
on the box created `/tmp/nsight-compute-lock` and the file ended up owned by
them. Workaround: redirect ncu's tmp dir.

```bash
mkdir -p ~/.ncu_tmp
TMPDIR=$HOME/.ncu_tmp ncu ...   # the rest of the command
```

This is what the rest of this guide does.

**`Profiling failed because a driver resource was unavailable.`** — another
profiler / DCGM is running on the GPU. Profiling counters are exclusive on
Volta; only one tool can collect at a time. Check:

```bash
ps -ef | grep -E "(ncu|nsight|dcgm)" | grep -v grep
nvidia-smi
```

Wait for the other tool to finish, or coordinate with whoever is running it.

## 5. The two flags that matter most

### `--set` — preset metric bundles

Lists with `ncu --list-sets`. The useful ones for this assignment:

| Set | What you get | When to use |
|---|---|---|
| `basic` | LaunchStats, Occupancy, SoL, WorkloadDistribution (default) | Sanity / quick look |
| `roofline` | All `SpeedOfLight_Hierarchical*RooflineChart` sections | Roofline analysis (Q4) |
| `full` | Everything ncu can collect | Deep dive; **slow** (many replays) |

Sets are pre-bundled groups of *sections*; you can also pull individual sections
with `--section`, e.g. `--section SpeedOfLight` for just the throughput-percent
table.

### `--metrics` — granular counter list

When you know exactly which metrics you want (e.g. for AI computation), use
`--metrics name1,name2,...`. This is much faster than `--set full` because ncu
needs fewer replays. Browse available metrics with `ncu --query-metrics`.

The metrics this assignment needs:

| Metric | Meaning |
|---|---|
| `dram__bytes_read.sum` | Bytes read from HBM2 by this kernel |
| `dram__bytes_write.sum` | Bytes written to HBM2 |
| `gpu__time_duration.sum` | Per-launch kernel time (ns) |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed` | Tensor pipe utilization (% of peak sustained) |
| `sm__inst_executed_pipe_tensor_op_hmma.sum` | Number of native HMMA instructions issued |
| `l1tex__t_bytes.sum` | L1/TEX traffic (request bytes) |
| `lts__t_bytes.sum` | L2 traffic |

Convention: `*.sum` aggregates across SMs/sub-units; `*.avg` averages.

## 6. Filtering kernels — `-k`

Without filtering, `ncu` profiles **every** kernel the program launches. For
this binary that includes `float2half_kernel` and `float2half_pad_kernel` per
layer (3 launches × 4 configs = 12 profiled kernels), roughly tripling the
runtime. Almost always you want:

```bash
ncu -k regex:convolution_kernel ...
```

Other filters (`--launch-skip N --launch-count M`, `-c <count>`, `--kernel-id ::N`)
are useful when you want a specific instance. For this assignment I capture all
four `convolution_kernel` invocations in one run and pick the rows I need from
the CSV (Conv1 B=1, Conv1 B=16, Conv2 B=1, Conv2 B=16 in launch order).

## 7. Output formats

| Flag | Result |
|---|---|
| (default) | Human-readable table on stdout |
| `--csv --log-file foo.csv` | CSV (one row per metric per kernel) |
| `--export foo.ncu-rep` | Binary report; open with `ncu-ui foo.ncu-rep` for the GUI |

The CSV is what you want for any scripted post-processing.

## 8. The exact commands used for this assignment

### (a) Targeted metric query — for the AI / GFLOPS computation

```bash
TMPDIR=$HOME/.ncu_tmp ncu -k regex:convolution_kernel \
    --metrics \
dram__bytes_read.sum,\
dram__bytes_write.sum,\
gpu__time_duration.sum,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__inst_executed_pipe_tensor_op_hmma.sum,\
l1tex__t_bytes.sum,\
lts__t_bytes.sum \
    --csv --log-file ncu_metrics.csv \
    ./conv_cuda
```

What to look for in `ncu_metrics.csv`:
- Four rows per metric, one per `convolution_kernel` invocation. The Grid Size
  column distinguishes them: `(1, 28, 14)` is Conv1 B=1, `(16, 28, 14)` is Conv1
  B=16, `(1, 2, 1)` is Conv2 B=1, `(16, 2, 1)` is Conv2 B=16.
- DRAM = `dram__bytes_read.sum + dram__bytes_write.sum`. **Achieved AI =
  algorithmic FLOPs ÷ DRAM bytes.**
- `gpu__time_duration.sum` is in nanoseconds. **Achieved GFLOPS = FLOPs ÷
  duration.**
- `sm__inst_executed_pipe_tensor_op_hmma.sum × 512` should equal your
  algorithmic FLOP count. On Volta each native HMMA does an 8×8×4 mul-add =
  256 MACs = 512 FLOPs. If this is *higher* than your algorithmic count, the
  kernel is computing extra work (e.g. our Conv2 pads 14×14 → 16×16, doing 1.31×
  more HMMAs).

### (b) Speed-of-Light section — for the bottleneck story

```bash
TMPDIR=$HOME/.ncu_tmp ncu --section SpeedOfLight -k regex:convolution_kernel \
    --csv --log-file ncu_sol.csv \
    ./conv_cuda
```

What to look for in `ncu_sol.csv`:
- `Memory Throughput` (max of L1/L2/DRAM) and `Compute (SM) Throughput`. If
  either is high (>60 %) that's your bottleneck. If both are low, the kernel
  is latency-bound (warps stalled waiting on memory or sync).
- The breakdown: `DRAM Throughput`, `L1/TEX Cache Throughput`, `L2 Cache
  Throughput`. For conv.cu the numbers tell a clear story: DRAM is at 8 % /
  2 % but L1 is at 69 % / 50 % — meaning it's an *L1 throughput-bound* kernel,
  not a DRAM-bound one.
- The `SOLBottleneck` rule row is ncu's automatic diagnosis. For this kernel:
  *"Memory is more heavily utilized than Compute: Look at the Memory Workload
  Analysis section to identify the L1 bottleneck."*

### (c) Optional — full roofline charts

```bash
TMPDIR=$HOME/.ncu_tmp ncu --set roofline -k regex:convolution_kernel \
    --launch-skip 1 --launch-count 1 \
    --export ncu_conv1_b16.ncu-rep \
    ./conv_cuda
ncu-ui ncu_conv1_b16.ncu-rep      # open the GUI roofline chart
```

The GUI (`ncu-ui`) plots the achieved point on a hierarchical roofline (DRAM,
L2, L1 ceilings + tensor / FP32 / FP64 ceilings). The CSV equivalent is less
useful — most of the chart data is in binary form inside the `.ncu-rep`.

## 9. Reading the SoL section

For a quick sanity check on any kernel:

| If you see... | It usually means... |
|---|---|
| Compute throughput high (>60 %), memory low | Compute-bound — try cheaper instructions or fewer ops |
| Memory throughput high (>60 %), compute low | Memory-bound — work on AI: reuse, blocking, fusion |
| Both high | Well-tuned; near the ceiling |
| Both low (<30 %) | Latency-bound; check Warp State Stats for the stall reason (long-scoreboard, barrier, ...) |
| L1/TEX > 60 %, DRAM low | L1 cache pipe is the bottleneck; reduce L1 requests per FLOP (more reuse in registers) |

For this kernel the diagnosis is the last row: high L1 traffic per HMMA is
holding back the tensor pipe, which sits at ~5 % utilization while the SMs
chew through shared-memory loads.

## 10. Common gotchas

- **Wall-clock numbers under ncu are wrong.** Use `gpu__time_duration.sum`.
- **Cache state matters.** ncu replays multiple times; by replay 2 the caches
  are warm. If you want cold-cache numbers, add `--cache-control none`. For
  this assignment we want the steady-state numbers, so leave it default.
- **GPU clocks are locked to base by default** (`--clock-control base`). That's
  ~1.2 GHz on TITAN V; boost is 1.455 GHz. For roofline analysis this is fine
  (the comparison is to peak); if you want production numbers, use
  `--clock-control none`.
- **Multi-GPU systems**: ncu profiles GPU 0 by default. Use `CUDA_VISIBLE_DEVICES`
  before the command to pick a specific GPU.
- **Forking processes**: add `--target-processes all` if your binary forks.
  conv_cuda doesn't, so we don't need it.
- **CSV log file vs. stdout**: when you pass `--log-file`, **all** output
  (including the program's own stdout) goes to the file. Pass without
  `--log-file` for a quick interactive run.

## 11. Putting it together

For the assignment you really only need command (a) above. With the four output
rows it produces, you can compute every number in [roofline.md](roofline.md):

| Quantity | Formula | Source |
|---|---|---|
| Algorithmic FLOPs | `2 · B · NYSCL · NXSCL · KY · KX · Ni · Nn` | [conv_flop.md](conv_flop.md) |
| Theoretical min bytes | `B·NYPAD·NXPAD·Ni·2 + KY·KX·Nn·Ni·2 + B·NYSCL·NXSCL·Nn·4` | derived |
| Theoretical AI | FLOPs ÷ theoretical bytes | derived |
| DRAM bytes | `dram__bytes_read.sum + dram__bytes_write.sum` | ncu_metrics.csv |
| Achieved AI | FLOPs ÷ DRAM bytes | derived |
| Kernel time | `gpu__time_duration.sum` (ns → s) | ncu_metrics.csv |
| Achieved GFLOPS | FLOPs ÷ time | derived |
| % of tensor peak | Achieved GFLOPS ÷ 110 000 | derived |

Command (b) gets you the per-cache-level throughput percentages that make the
"L1-bound, not DRAM-bound" diagnosis clear.
