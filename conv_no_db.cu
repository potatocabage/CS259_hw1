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

template <int NI_CHUNK>
__device__ void load_neuron_i_tile(
    half* neuron_i_smem,
    half* neuron_i_global,
    int t_idx,
    int batch_idx, int y_out, int x_out, int Ni, int ic_base,
    int blockDim_x, int NYPAD, int NXPAD
)
{
    const int tile_h = KY - 1 + 8;          // 10 for KY=3
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
        neuron_i_smem[idx] = neuron_i_global[gidx];
    }
    __syncthreads();
}

template <int NI_CHUNK>
__device__ void load_synapse_xy_to_smem(
    half* synapse_smem,
    half* synapse_global,
    int k_row, int k_col, int Nn_idx, int ic_base, int t_idx, int blockDim_x,
    int Ni, int Nn, int kx_dim, int ky_dim
)
{
    // Stage a [16][NI_CHUNK] slice of synapse[KY][KX][Nn][Ni] starting at
    // (k_row, k_col, Nn_idx*16, ic_base) directly into shared memory.
    int elements_per_thread = (16 * NI_CHUNK) / blockDim_x;
    for (int k = 0; k < elements_per_thread; k++){
        int linear   = k * blockDim_x + t_idx;
        int n_local  = linear / NI_CHUNK;
        int ic_local = linear % NI_CHUNK;
        int global_flat = ((k_row * kx_dim + k_col) * Nn + Nn_idx * 16 + n_local) * Ni
                        + ic_base + ic_local;
        synapse_smem[linear] = synapse_global[global_flat];
    }
}


/* ------------------------------------------------------------------ */
/* CUDA kernel for convolution (no double-buffering variant)          */
/* ------------------------------------------------------------------ */
template <int NI_CHUNK>
__global__ void convolution_kernel(
    half *synapse,    /* [KY * KX * Nn * Ni]         */
    half *neuron_i,   /* [B * NYPAD * NXPAD * Ni]    */
    float *neuron_n,   /* [B * NYSCL * NXSCL * Nn]    */
    int B, int NYPAD, int NXPAD, int NYSCL, int NXSCL, int Ni, int Nn, int ky_dim, int kx_dim)
{
    // Grid Dim is (B, NYSCL / 8, NXSCL / 16); spatial dims are pre-padded by the launcher.
    // Block Dim is (256, 1, 1)
    // each block does a 1x8x16xNn tile in output
    // each warp does a 1x1x16xNn tile in output
    int t_idx = threadIdx.x; //0 to 255
    int warp_idx = t_idx / 32; // 0 to 7

    int batch_idx = blockIdx.x;
    int y_out = blockIdx.y * 8;
    int x_out = blockIdx.z * 16;

    int pad_y = (ky_dim - 1) / 2;
    int pad_x = (kx_dim - 1) / 2;

    __shared__ half  neuron_i_smem[10 * 18 * NI_CHUNK];  // (8 + KY-1) * (16 + KX-1) * NI_CHUNK
    __shared__ half  synapse_smem[16 * NI_CHUNK];        // single 16-channel synapse block
    __shared__ float epilogue_smem[8 * 16 * 16];         // 8 warps * 16 output cols * 16-channel block

    // make wmma fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> toeplitz_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> filter_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    for (int i = 0; i < Nn / 16; i ++){
        // c_frag accumulates partial sums across input-channel chunks.
        wmma::fill_fragment(c_frag, 0.0f);

        for (int ic_base = 0; ic_base < Ni; ic_base += NI_CHUNK){
            // load neuron_i tile (1x10x18xNI_CHUNK) into smem for this chunk
            load_neuron_i_tile<NI_CHUNK>(neuron_i_smem, neuron_i, t_idx, batch_idx, y_out, x_out, Ni, ic_base, blockDim.x, NYPAD, NXPAD);

            for (int k_row = 0; k_row < ky_dim; k_row ++){
                for (int k_col = 0; k_col < kx_dim ; k_col ++){
                    // Load this (k_row, k_col)'s synapse block from global into smem.
                    load_synapse_xy_to_smem<NI_CHUNK>(synapse_smem, synapse,
                        k_row, k_col, i, ic_base, t_idx, blockDim.x, Ni, Nn, kx_dim, ky_dim);
                    __syncthreads();

                    for(int j = 0; j < NI_CHUNK / 16; j++){

                        // load toeplitz from smem into registers (smem tile stride = 18 * NI_CHUNK)
                        int k_off_y = k_row - ky_dim / 2;
                        int k_off_x = k_col - kx_dim / 2;
                        int toeplitz_idx = ((warp_idx + pad_y + k_off_y) * 18 * NI_CHUNK) + ((pad_x + k_off_x) * NI_CHUNK) + (j * 16);
                        wmma::load_matrix_sync(toeplitz_frag, neuron_i_smem + toeplitz_idx, NI_CHUNK);

                        // load filter from smem into registers
                        int filter_idx = j * 16;
                        wmma::load_matrix_sync(filter_frag, synapse_smem + filter_idx, NI_CHUNK);

                        // dispatch wmma
                        wmma::mma_sync(c_frag, toeplitz_frag, filter_frag, c_frag);
                    }

                    __syncthreads();
                }
            }
        }

        // Per-i epilogue: stage one 16-channel block into smem, float4-flush to global with ReLU.
        int eslot = warp_idx * 16 * 16;            // warp * (16 outputs * 16 channels)
        wmma::store_matrix_sync(&epilogue_smem[eslot], c_frag, 16, wmma::mem_row_major);
        __syncthreads();

        // 8 warps * 16 cols * 16 chans = 2048 floats = 512 float4. With 256 threads → 2 steps.
        const int floats_per_block = 8 * 16 * 16;
        const int steps = floats_per_block / (4 * blockDim.x);
        for (int step = 0; step < steps; step++) {
            int f4_idx     = step * blockDim.x + t_idx;
            int smem_idx   = f4_idx * 4;
            int smem_pixel = smem_idx / 16;        // 0..127 = warp_idx*16 + local_x
            int local_y    = smem_pixel / 16;      // = warp_idx
            int local_x    = smem_pixel % 16;
            int col_in_blk = smem_idx % 16;        // 0..15 within the 16-channel block

            int global_idx = (batch_idx * NYSCL * NXSCL * Nn)
                           + ((y_out + local_y) * NXSCL * Nn)
                           + ((x_out + local_x) * Nn)
                           + (i * 16 + col_in_blk);
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

    // Pad output spatial to block tile (8, 16); input grows by (KY-1, KX-1).
    // For Conv1 (224×224) these are no-ops; for Conv2 (14×14) we round up to 16×16.
    int NYSCL_pad = ((NYSCL + 7) / 8) * 8;
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

    dim3 blockDim(256);
    dim3 gridDim(B, NYSCL_pad / 8, NXSCL_pad / 16);

    convolution_kernel<64><<<gridDim, blockDim>>>(
        synapse_device_half, neuron_i_device_half, neuron_n_device,
        B, NYPAD_pad, NXPAD_pad, NYSCL_pad, NXSCL_pad, Ni, Nn, KY, KX);
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
    printf("=== Convolution CUDA (WMMA, no double-buffering) ===\n\n");

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
