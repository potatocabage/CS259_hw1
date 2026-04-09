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
