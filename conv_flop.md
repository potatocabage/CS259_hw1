# Convolution FLOP counts (batch 16)

The kernel computes a 2D convolution with a 3×3 filter, stride 1, and ReLU.
The FLOP formula in [conv.cu:412](conv.cu#L412) is:

```
flops = 2 * B * NYSCL * NXSCL * KY * KX * Ni * Nn
```

The `2 *` accounts for one multiply + one add per MAC. Stride is 1, so
`NYSCL = Ny` and `NXSCL = Nx`. The ReLU adds `B * NYSCL * NXSCL * Nn`
comparisons but is conventionally excluded from convolution FLOP counts
(and is excluded by the formula above).

Constants: `KY = KX = 3`.

## Conv1-VGG, B = 16

Parameters: `B=16, Ny=Nx=224, Ni=Nn=64`.

```
flops = 2 * 16 * 224 * 224 * 3 * 3 * 64 * 64
```

Step by step:

| step                          | value           |
|-------------------------------|-----------------|
| `B = 16`                      | 16              |
| `NYSCL * NXSCL = 224 * 224`   | 50,176          |
| `KY * KX = 3 * 3`             | 9               |
| `Ni * Nn = 64 * 64`           | 4,096           |
| `2 * 16 * 50,176`             | 1,605,632       |
| `* 9`                         | 14,450,688      |
| `* 4,096`                     | **59,190,018,048** |

**Conv1, B=16: 59,190,018,048 FLOPs ≈ 59.19 GFLOPs (≈ 5.919 × 10¹⁰)**

## Conv2-VGG, B = 16

Parameters: `B=16, Ny=Nx=14, Ni=Nn=512`.

```
flops = 2 * 16 * 14 * 14 * 3 * 3 * 512 * 512
```

Step by step:

| step                          | value           |
|-------------------------------|-----------------|
| `B = 16`                      | 16              |
| `NYSCL * NXSCL = 14 * 14`     | 196             |
| `KY * KX = 3 * 3`             | 9               |
| `Ni * Nn = 512 * 512`         | 262,144         |
| `2 * 16 * 196`                | 6,272           |
| `* 9`                         | 56,448          |
| `* 262,144`                   | **14,797,504,512** |

**Conv2, B=16: 14,797,504,512 FLOPs ≈ 14.80 GFLOPs (≈ 1.480 × 10¹⁰)**

## Summary

| layer       | B  | spatial   | Ni  | Nn  | FLOPs              | GFLOPs  |
|-------------|----|-----------|-----|-----|--------------------|---------|
| Conv1-VGG   | 16 | 224 × 224 | 64  | 64  | 59,190,018,048     | 59.19   |
| Conv2-VGG   | 16 | 14 × 14   | 512 | 512 | 14,797,504,512     | 14.80   |
| **total**   |    |           |     |     | **73,987,522,560** | **73.99** |
