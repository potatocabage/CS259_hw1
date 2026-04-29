#include <cuda_runtime.h>
#include <mma.h>
#include "timing.h"
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using namespace nvcuda;

// Convolution parameters
#define KY  3   /* kernel height */
#define KX  3   /* kernel width  */
#define SY  1   /* stride y      */
#define SX  1   /* stride x      */

typedef float VTYPE;

static void fill(VTYPE *m, long long n, float scale, int seed)
{
    for (long long i = 0; i < n; i++)
        m[i] = scale * sinf((float)(i * 3 + seed * 7));
}

/* CPU reference convolution (with ReLU) — matches the GPU kernel's semantics. */
static void convolution_reference(
    const VTYPE *synapse, const VTYPE *neuron_i, VTYPE *neuron_n_ref,
    int B, int Ny, int Nx, int Ni, int Nn)
{
    int NXPAD = Nx + KX - 1,  NYPAD = Ny + KY - 1;
    int NXSCL = (Nx + SX - 1) / SX,  NYSCL = (Ny + SY - 1) / SY;
    for (int b = 0; b < B; b++)
    for (int y = 0; y < Ny; y++)
    for (int x = 0; x < Nx; x++)
    for (int n = 0; n < Nn; n++) {
        float sum = 0.f;
        for (int ky = 0; ky < KY; ky++)
        for (int kx = 0; kx < KX; kx++)
        for (int i  = 0; i  < Ni; i++)
            sum += synapse[((ky*KX + kx)*Nn + n)*Ni + i]
                 * neuron_i[((long long)(b*NYPAD + y+ky)*NXPAD + x+kx)*Ni + i];
        neuron_n_ref[((long long)(b*NYSCL + y)*NXSCL + x)*Nn + n] = sum > 0.f ? sum : 0.f;
    }
}

static void compare_outputs(const VTYPE *got, const VTYPE *ref, long long n)
{
    double max_abs = 0, max_rel = 0;
    long long bad = 0;
    for (long long k = 0; k < n; k++) {
        double d = fabs((double)got[k] - (double)ref[k]);
        double r = fabs((double)ref[k]);
        if (d > max_abs) max_abs = d;
        if (r > 1e-6 && d/r > max_rel) max_rel = d/r;
        if (d > 1e-2) bad++;
    }
    printf("    max_abs=%.4e  max_rel=%.4e  mismatches(>1e-2)=%lld/%lld %s\n",
        max_abs, max_rel, bad, n, bad == 0 ? "OK" : "FAIL");
}

__global__ void float2half_kernel(const float * in, half * out, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < size; idx += blockDim.x * gridDim.x){
        out[idx] = __float2half(in[idx]);
    }
}

// Read compact [B][NYPAD][NXPAD][Ni] float, write into spatially padded
// [B][NYPAD_pad][NXPAD_pad][Ni] half. Caller must pre-zero the destination.
__global__ void float2half_pad_kernel(
    const float *src, half *dst,
    int B, int NYPAD, int NXPAD, int Ni,
    int NYPAD_pad, int NXPAD_pad)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * NYPAD * NXPAD * Ni;
    for (; idx < total; idx += blockDim.x * gridDim.x){
        int ni  = idx % Ni;
        int rem = idx / Ni;
        int x   = rem % NXPAD;
        int rem2 = rem / NXPAD;
        int y   = rem2 % NYPAD;
        int b   = rem2 / NYPAD;
        int dst_idx = ((b * NYPAD_pad + y) * NXPAD_pad + x) * Ni + ni;
        dst[dst_idx] = __float2half(src[idx]);
    }
}

/* ------------------------------------------------------------------ */
/* Device helper functions (moved above kernel for forward decl)      */
/* ------------------------------------------------------------------ */
//
// NI_PAD pads each row of the SMEM tiles by 8 halfs (16 bytes) so that the
// row stride NI_STRIDE = NI_CHUNK + 8 = 72 halfs = 36 banks (mod 32 = 4).
// This converts the 16-way bank conflict from a NI_CHUNK=64 (= 32 banks)
// row stride into a 2-way conflict, reclaiming most of the SMEM read time.
// 8 halfs is the smallest pad that satisfies WMMA's "ld must be a multiple
// of 8 halfs" alignment requirement.
//
#define NI_PAD 8

template <int NI_CHUNK, int BLOCK_Y>
__device__ void load_neuron_i_tile(
    half* neuron_i_smem,
    half* neuron_i_global,
    int t_idx,
    int batch_idx, int y_out, int x_out, int Ni, int ic_base,
    int blockDim_x, int NYPAD, int NXPAD
)
{
    constexpr int NI_STRIDE = NI_CHUNK + NI_PAD;
    const int tile_h = KY - 1 + BLOCK_Y;    // 10 for KY=3, BLOCK_Y=8; 4 for BLOCK_Y=2
    const int tile_w = KX - 1 + 16;         // 18 for KX=3
    const int total  = tile_h * tile_w * NI_CHUNK;
    for (int idx = t_idx; idx < total; idx += blockDim_x) {
        int ni  = idx % NI_CHUNK;
        int rem = idx / NI_CHUNK;
        int tx  = rem % tile_w;
        int ty  = rem / tile_w;
        int gidx = (batch_idx * NYPAD * NXPAD * Ni)
                 + ((y_out + ty) * NXPAD * Ni)
                 + ((x_out + tx) * Ni)
                 + ic_base + ni;
        int smem_idx = (ty * tile_w + tx) * NI_STRIDE + ni;
        neuron_i_smem[smem_idx] = neuron_i_global[gidx];
    }
    __syncthreads();
}

// Stage NN_GROUP contiguous 16-output-channel filter slices into smem in one pass.
// Smem layout: [NN_GROUP][16][NI_STRIDE] row-major (last NI_PAD halfs unused).
// First slice corresponds to output-channel block (i_group * NN_GROUP + 0).
template <int NI_CHUNK, int NN_GROUP>
__device__ void load_synapse_group_smem(
    half* synapse_smem,
    const half* synapse_global,
    int k_row, int k_col, int i_group, int ic_base, int t_idx, int blockDim_x,
    int Ni, int Nn, int kx_dim)
{
    constexpr int NI_STRIDE = NI_CHUNK + NI_PAD;
    const int total = NN_GROUP * 16 * NI_CHUNK;
    for (int idx = t_idx; idx < total; idx += blockDim_x) {
        int ic_local = idx % NI_CHUNK;
        int rem      = idx / NI_CHUNK;
        int n_local  = rem % 16;
        int slice    = rem / 16;
        int n_global = (i_group * NN_GROUP + slice) * 16 + n_local;
        int gidx = ((k_row * kx_dim + k_col) * Nn + n_global) * Ni
                 + ic_base + ic_local;
        int smem_idx = (slice * 16 + n_local) * NI_STRIDE + ic_local;
        synapse_smem[smem_idx] = synapse_global[gidx];
    }
}


/* ------------------------------------------------------------------ */
/* CUDA kernel for convolution — Iter 1+2+3                           */
/* ------------------------------------------------------------------ */
//
// Iter 1: Nn-tile fusion. load_neuron_i_tile hoisted out of the output-channel
//   loop. Each warp accumulates NN_GROUP output-channel chunks at once with one
//   shared toeplitz fragment.
// Iter 2: SMEM padded by NI_PAD=8 halfs to break 16-way row-stride bank conflict
//   into 2-way (~18× fewer conflicts).
// Iter 3: BLOCK_Y is now a template parameter so we can scale the spatial-y
//   block tile down for layers with small NYSCL. Conv1 still uses BLOCK_Y=8;
//   Conv2 uses BLOCK_Y=2 to grow the grid from 32 → 112 blocks (3.5× more
//   blocks → better SM occupancy when ncu rule was "grid too small").
//
template <int NI_CHUNK, int NN_GROUP, int BLOCK_Y>
__global__ void convolution_kernel(
    half *synapse,    /* [KY * KX * Nn * Ni]         */
    half *neuron_i,   /* [B * NYPAD * NXPAD * Ni]    */
    float *neuron_n,   /* [B * NYSCL * NXSCL * Nn]    */
    int B, int NYPAD, int NXPAD, int NYSCL, int NXSCL, int Ni, int Nn, int ky_dim, int kx_dim)
{
    // Grid Dim is (B, NYSCL_pad / BLOCK_Y, NXSCL_pad / 16). Spatial dims pre-padded by launcher.
    // Block Dim is (BLOCK_Y * 32, 1, 1). Each block computes a BLOCK_Y×16 spatial tile × Nn output channels.
    // Each warp computes one row (1×16) of spatial outputs across all NN_GROUP*16 = 64 output channels at a time.
    int t_idx    = threadIdx.x;             // 0..(BLOCK_Y*32 - 1)
    int warp_idx = t_idx / 32;              // 0..(BLOCK_Y - 1)

    int batch_idx = blockIdx.x;
    int y_out     = blockIdx.y * BLOCK_Y;
    int x_out     = blockIdx.z * 16;

    int pad_y = (ky_dim - 1) / 2;
    int pad_x = (kx_dim - 1) / 2;

    constexpr int NI_STRIDE = NI_CHUNK + NI_PAD;                       // padded row stride breaks bank conflicts
    constexpr int TILE_H    = BLOCK_Y + KY - 1;                        // input tile height (incl. halo)
    constexpr int TILE_W    = 16 + KX - 1;                             // input tile width  (incl. halo)
    __shared__ half  neuron_i_smem[TILE_H * TILE_W * NI_STRIDE];
    __shared__ half  synapse_smem[NN_GROUP * 16 * NI_STRIDE];          // NN_GROUP slices of [16][NI_STRIDE]
    __shared__ float epilogue_smem[BLOCK_Y * 16 * 16];                 // BLOCK_Y warps * 16 cols * 16 channels

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> toeplitz_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> filter_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frags[NN_GROUP];

    const int i_groups = (Nn / 16) / NN_GROUP;  // assumes Nn % (16*NN_GROUP) == 0

    for (int i_group = 0; i_group < i_groups; i_group++) {
        #pragma unroll
        for (int n = 0; n < NN_GROUP; n++) wmma::fill_fragment(c_frags[n], 0.0f);

        for (int ic_base = 0; ic_base < Ni; ic_base += NI_CHUNK) {
            // Load TILE_H x TILE_W x NI_CHUNK input tile into SMEM. Once per (i_group, ic_base).
            load_neuron_i_tile<NI_CHUNK, BLOCK_Y>(neuron_i_smem, neuron_i, t_idx,
                                                  batch_idx, y_out, x_out, Ni, ic_base,
                                                  blockDim.x, NYPAD, NXPAD);

            for (int k_row = 0; k_row < ky_dim; k_row++) {
                for (int k_col = 0; k_col < kx_dim; k_col++) {
                    // Stage NN_GROUP filter slices for this (k_row, k_col).
                    load_synapse_group_smem<NI_CHUNK, NN_GROUP>(
                        synapse_smem, synapse,
                        k_row, k_col, i_group, ic_base,
                        t_idx, blockDim.x, Ni, Nn, kx_dim);
                    __syncthreads();

                    int k_off_y = k_row - ky_dim / 2;
                    int k_off_x = k_col - kx_dim / 2;
                    int toep_row_base = (warp_idx + pad_y + k_off_y) * TILE_W * NI_STRIDE
                                      + (pad_x + k_off_x) * NI_STRIDE;

                    #pragma unroll 1
                    for (int j = 0; j < NI_CHUNK / 16; j++) {
                        // Load toeplitz once per (k_row, k_col, j); reuse across NN_GROUP MMAs.
                        wmma::load_matrix_sync(toeplitz_frag,
                                               neuron_i_smem + toep_row_base + j * 16,
                                               NI_STRIDE);

                        #pragma unroll
                        for (int n = 0; n < NN_GROUP; n++) {
                            int filter_idx = n * (16 * NI_STRIDE) + j * 16;
                            wmma::load_matrix_sync(filter_frag,
                                                   synapse_smem + filter_idx,
                                                   NI_STRIDE);
                            wmma::mma_sync(c_frags[n], toeplitz_frag, filter_frag, c_frags[n]);
                        }
                    }

                    __syncthreads();   // protect synapse_smem before next stage overwrites it
                }
            }
        }

        // Epilogue: store the NN_GROUP accumulators one slice at a time. Each
        // slice goes through the same BLOCK_Y x 16 x 16 SMEM block + float4 flush
        // as the baseline. Keeps the smem footprint independent of NN_GROUP.
        const int floats_per_block = BLOCK_Y * 16 * 16;
        const int steps = floats_per_block / (4 * blockDim.x);

        #pragma unroll 1
        for (int n = 0; n < NN_GROUP; n++) {
            int eslot = warp_idx * 16 * 16;
            wmma::store_matrix_sync(&epilogue_smem[eslot], c_frags[n], 16, wmma::mem_row_major);
            __syncthreads();

            for (int step = 0; step < steps; step++) {
                int f4_idx     = step * blockDim.x + t_idx;
                int smem_idx   = f4_idx * 4;
                int smem_pixel = smem_idx / 16;
                int local_y    = smem_pixel / 16;
                int local_x    = smem_pixel % 16;
                int col_in_blk = smem_idx % 16;

                int n_chunk    = i_group * NN_GROUP + n;
                int global_idx = (batch_idx * NYSCL * NXSCL * Nn)
                               + ((y_out + local_y) * NXSCL * Nn)
                               + ((x_out + local_x) * Nn)
                               + (n_chunk * 16 + col_in_blk);
                float4 v = *(float4*)(&epilogue_smem[smem_pixel * 16 + col_in_blk]);
                v.x = fmaxf(v.x, 0.f);   // ReLU — matches CPU reference
                v.y = fmaxf(v.y, 0.f);
                v.z = fmaxf(v.z, 0.f);
                v.w = fmaxf(v.w, 0.f);
                *(float4*)(&neuron_n[global_idx]) = v;
            }
            __syncthreads();
        }
    }
}

/* ------------------------------------------------------------------ */
/* Host wrapper function                                              */
/* ------------------------------------------------------------------ */
void launch_convolution_layer(
    float *synapse,
    float *neuron_i,
    float *neuron_n,
    int B, int Ny, int Nx, int Ni, int Nn)
{
    
    int NXPAD = Nx + KX - 1,  NYPAD = Ny + KY - 1;
    int NXSCL = (Nx + SX - 1) / SX,  NYSCL = (Ny + SY - 1) / SY;

    // Iter 3: BLOCK_Y is now a template parameter (kept for code clarity), but
    // experiments below show smaller blocks regress Conv2 — too few warps per
    // SM scheduler outweighs the extra grid parallelism. So BLOCK_Y stays at 8.
    // Conv2's actual fix is NN_GROUP=8 (vs NN_GROUP=4 for Conv1): doubles the
    // output channels processed per warp, halves the i_group/sync loop count
    // for layers where Nn is large enough (Nn ≥ 16*8 = 128).
    const int BLOCK_Y = 8;
    const bool USE_BIG_NN_GROUP = (Nn >= 128);   // Conv2 (Nn=512) ✓; Conv1 (Nn=64) ✗
    int NYSCL_pad = ((NYSCL + BLOCK_Y - 1) / BLOCK_Y) * BLOCK_Y;
    int NXSCL_pad = ((NXSCL + 15) / 16) * 16;
    int NYPAD_pad = NYSCL_pad + KY - 1;
    int NXPAD_pad = NXSCL_pad + KX - 1;

    long long syn_elems     = (long long)KY * KX * Nn * Ni;
    long long inp_elems     = (long long)B * NYPAD * NXPAD * Ni;
    long long inp_elems_pad = (long long)B * NYPAD_pad * NXPAD_pad * Ni;
    long long out_elems_pad = (long long)B * NYSCL_pad * NXSCL_pad * Nn;

    size_t synapse_half_bytes      = syn_elems * sizeof(half);
    size_t neuron_i_half_pad_bytes = inp_elems_pad * sizeof(half);
    size_t synapse_float_bytes     = syn_elems * sizeof(float);
    size_t neuron_i_float_bytes    = inp_elems * sizeof(float);
    size_t neuron_n_pad_bytes      = out_elems_pad * sizeof(float);

    half *synapse_device_half, *neuron_i_device_half;
    float *synapse_device_float, *neuron_i_device_float;
    float *neuron_n_device;

    // allocate mem
    cudaMalloc((void**)&synapse_device_half, synapse_half_bytes);
    cudaMalloc((void**)&neuron_i_device_half, neuron_i_half_pad_bytes);
    cudaMalloc((void**)&synapse_device_float, synapse_float_bytes);
    cudaMalloc((void**)&neuron_i_device_float, neuron_i_float_bytes);
    cudaMalloc((void**)&neuron_n_device, neuron_n_pad_bytes);

    // zero the padded half input — values outside the unpadded region stay 0.
    cudaMemset(neuron_i_device_half, 0, neuron_i_half_pad_bytes);

    // copy mem (float-sized, compact)
    cudaMemcpy(synapse_device_float, synapse, synapse_float_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(neuron_i_device_float, neuron_i, neuron_i_float_bytes, cudaMemcpyHostToDevice);

    // convert float -> half
    int conversion_threads = 256;
    {
        int n = (int)syn_elems;
        int blocks = min((n + conversion_threads - 1) / conversion_threads, 2048);
        float2half_kernel<<<blocks, conversion_threads>>>(
            synapse_device_float, synapse_device_half, n);
    }
    {
        // Pad-aware conversion: read compact float, write into padded half.
        int n = (int)inp_elems;
        int blocks = min((n + conversion_threads - 1) / conversion_threads, 2048);
        float2half_pad_kernel<<<blocks, conversion_threads>>>(
            neuron_i_device_float, neuron_i_device_half,
            B, NYPAD, NXPAD, Ni, NYPAD_pad, NXPAD_pad);
    }
    cudaDeviceSynchronize();
    cudaFree(synapse_device_float);
    cudaFree(neuron_i_device_float);

    dim3 blockDim(BLOCK_Y * 32);
    dim3 gridDim(B, NYSCL_pad / BLOCK_Y, NXSCL_pad / 16);

    if (USE_BIG_NN_GROUP) {
        convolution_kernel<64, 8, 8><<<gridDim, blockDim>>>(
            synapse_device_half, neuron_i_device_half, neuron_n_device,
            B, NYPAD_pad, NXPAD_pad, NYSCL_pad, NXSCL_pad, Ni, Nn, KY, KX);
    } else {
        convolution_kernel<64, 4, 8><<<gridDim, blockDim>>>(
            synapse_device_half, neuron_i_device_half, neuron_n_device,
            B, NYPAD_pad, NXPAD_pad, NYSCL_pad, NXSCL_pad, Ni, Nn, KY, KX);
    }
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "convolution_kernel error: %s\n", cudaGetErrorString(err));

    // Copy back the valid [NYSCL][NXSCL] region of each batch from the padded output.
    for (int b = 0; b < B; b++) {
        cudaMemcpy2D(
            &neuron_n[(long long)b * NYSCL * NXSCL * Nn],
            (size_t)NXSCL * Nn * sizeof(float),
            &neuron_n_device[(long long)b * NYSCL_pad * NXSCL_pad * Nn],
            (size_t)NXSCL_pad * Nn * sizeof(float),
            (size_t)NXSCL * Nn * sizeof(float),
            NYSCL,
            cudaMemcpyDeviceToHost);
    }

    // free mem
    cudaFree(synapse_device_half);
    cudaFree(neuron_i_device_half);
    cudaFree(neuron_n_device);

}


static void run(const char *name, int B, int Ny, int Nx, int Ni, int Nn)
{
    int NYPAD = Ny + KY - 1,  NXPAD = Nx + KX - 1;
    int NYSCL = (Ny + SY - 1) / SY,  NXSCL = (Nx + SX - 1) / SX;

    long long syn_n = (long long)KY * KX * Nn * Ni;
    long long inp_n = (long long)B * NYPAD * NXPAD * Ni;
    long long out_n = (long long)B * NYSCL * NXSCL * Nn;

    VTYPE *synapse  = (VTYPE *)malloc(syn_n * sizeof(VTYPE));
    VTYPE *neuron_i = (VTYPE *)calloc(inp_n, sizeof(VTYPE));   /* zero-init = zero padding */
    VTYPE *neuron_n = (VTYPE *)malloc(out_n * sizeof(VTYPE));

    fill(synapse, syn_n, 0.01f, 1);
    for (int b = 0; b < B; b++)
    for (int y = 0; y < Ny; y++)
    for (int x = 0; x < Nx; x++)
    for (int i = 0; i < Ni; i++)
        neuron_i[((long long)(b*NYPAD + y)*NXPAD + x)*Ni + i] =
            0.01f * sinf((float)(b*Ny*Nx*Ni + y*Nx*Ni + x*Ni + i));

    long long flops = 2LL * B * NYSCL * NXSCL * KY * KX * Ni * Nn;

    char label[64];
    snprintf(label, sizeof(label), "%s  B=%-2d", name, B);
    bench(label, [&]{ launch_convolution_layer(synapse, neuron_i, neuron_n, B, Ny, Nx, Ni, Nn); }, flops);

    if (getenv("CONV_VERIFY")) {
        VTYPE *neuron_n_ref = (VTYPE *)malloc(out_n * sizeof(VTYPE));
        convolution_reference(synapse, neuron_i, neuron_n_ref, B, Ny, Nx, Ni, Nn);
        compare_outputs(neuron_n, neuron_n_ref, out_n);
        free(neuron_n_ref);
    }

    free(synapse); free(neuron_i); free(neuron_n);
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */
int main(void)
{
    printf("=== Convolution CUDA (WMMA) ===\n\n");

    printf("  %-42s  %10s  %10s\n", "Layer", "ms", "GFLOPS");
    printf("  %s\n", "─────────────────────────────────────────────────────────────");

    /*        name         B    Ny   Nx   Ni   Nn  */
    run("Conv1-VGG",  1,  224, 224,  64,  64);
    run("Conv1-VGG", 16,  224, 224,  64,  64);
    run("Conv2-VGG",  1,   14,  14, 512, 512);
    run("Conv2-VGG", 16,   14,  14, 512, 512);

    printf("\n");
    return 0;
}