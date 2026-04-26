#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timing.h"

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

/* ------------------------------------------------------------------ */
/* Float-to-half conversion kernel                                    */
/* ------------------------------------------------------------------ */
__global__ void float2half_kernel(const float *in, half *out, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < size; idx += blockDim.x * gridDim.x){
        out[idx] = __float2half(in[idx]);
    }
}

/* ------------------------------------------------------------------ */
/* Device helper functions (defined before kernel that calls them)    */
/* ------------------------------------------------------------------ */

// Load a (10 x 18 x Ni) input tile into shared memory.
// The tile covers y_out..(y_out+9), x_out..(x_out+17) for Ni channels.
// With blockDim_x=128 and Ni=64, each thread loads 2 elements per (i,j) pair
// but we iterate so that blockDim_x threads cover Ni elements per spatial pos.
// 10 rows × 18 cols × 64 channels = 11520 elements.
// 128 threads, each iteration loads 128 elements → 11520/128 = 90 iterations.
__device__ void load_neuron_i_tile(
    half* neuron_i_smem,
    half* neuron_i_global,
    int t_idx,
    int batch_idx, int y_out, int x_out, int Ni,
    int blockDim_x, int NYPAD, int NXPAD
)
{
    // Tile dimensions in smem: 10 rows, 18 cols, Ni channels
    int tile_h = KY - 1 + 8;   // 10
    int tile_w = KX - 1 + 16;  // 18
    int tile_size = tile_h * tile_w * Ni;

    for (int idx = t_idx; idx < tile_size; idx += blockDim_x) {
        int ni = idx % Ni;
        int rem = idx / Ni;
        int tx = rem % tile_w;
        int ty = rem / tile_w;

        int gy = y_out + ty;
        int gx = x_out + tx;

        int global_idx = (batch_idx * NYPAD * NXPAD * Ni)
                       + (gy * NXPAD * Ni)
                       + (gx * Ni)
                       + ni;
        neuron_i_smem[idx] = neuron_i_global[global_idx];
    }
    __syncthreads();
}

// Load one [k_row][k_col] synapse slice (Nn x Ni = 64x64 = 4096 elements)
// from global memory directly into a per-thread register buffer.
// Each thread loads (Nn*Ni / blockDim_x) elements contiguously in its reg buf.
__device__ void load_synapse_xy_from_global(
    half* synapse_reg_buf,
    half* synapse_global,
    int k_row, int k_col, int t_idx, int blockDim_x,
    int Ni, int Nn, int kx_dim, int ky_dim
)
{
    int elements_per_thread = (Nn * Ni) / blockDim_x;
    int base = (k_row * kx_dim * Nn * Ni) + (k_col * Nn * Ni);
    for (int k = 0; k < elements_per_thread; k++){
        int global_flat = base + k * blockDim_x + t_idx;
        synapse_reg_buf[k] = synapse_global[global_flat];
    }
}

// Copy from per-thread register buffer into shared memory.
// Reverses the interleaving: reg_buf[k] → smem[k * blockDim_x + t_idx]
__device__ void load_synapse_xy_from_local(
    half* synapse_smem,
    half* synapse_reg_buf,
    int t_idx, int blockDim_x,
    int Ni, int Nn
)
{
    int elements_per_thread = (Nn * Ni) / blockDim_x;
    for (int k = 0; k < elements_per_thread; k++){
        int smem_flat = k * blockDim_x + t_idx;
        synapse_smem[smem_flat] = synapse_reg_buf[k];
    }
}

// Load one [k_row][k_col] synapse slice from global into shared memory directly.
// Used for the initial load (no double-buffering needed for the first tile).
__device__ void load_synapse_xy_to_smem(
    half* synapse_smem,
    half* synapse_global,
    int k_row, int k_col, int t_idx, int blockDim_x,
    int Ni, int Nn, int kx_dim, int ky_dim
)
{
    int elements_per_thread = (Nn * Ni) / blockDim_x;
    int base = (k_row * kx_dim * Nn * Ni) + (k_col * Nn * Ni);
    for (int k = 0; k < elements_per_thread; k++){
        int global_flat = base + k * blockDim_x + t_idx;
        int smem_flat = k * blockDim_x + t_idx;
        synapse_smem[smem_flat] = synapse_global[global_flat];
    }
}

/* ------------------------------------------------------------------ */
/* CUDA kernel for convolution                                        */
/*                                                                    */
/* Grid Dim:  (B, NYSCL / 8, NXSCL / 16)                            */
/* Block Dim: (128, 1, 1) = 4 warps                                  */
/*                                                                    */
/* Each block computes a 1 x 8 x 16 x 64 output tile                */
/* Each warp handles nn_tile = warp_id (16 output channels),         */
/*   looping over 8 y-rows.                                          */
/* ------------------------------------------------------------------ */
__global__ void convolution_kernel(
    half *synapse,     /* [KY * KX * Nn * Ni]         */
    half *neuron_i,    /* [B * NYPAD * NXPAD * Ni]    */
    float *neuron_n,   /* [B * NYSCL * NXSCL * Nn]    */
    int B, int NYPAD, int NXPAD, int NYSCL, int NXSCL,
    int Ni, int Nn, int ky_dim, int kx_dim)
{
    int t_idx = threadIdx.x;       // 0 to 127
    int warp_id = t_idx / 32;     // 0 to 3
    int batch_idx = blockIdx.x;
    int y_out = blockIdx.y * 8;
    int x_out = blockIdx.z * 16;

    // Shared memory for input tile and double-buffered synapse
    __shared__ half neuron_i_smem[10 * 18 * 64];       // (KY-1+8) x (KX-1+16) x Ni
    __shared__ half synapse_smem_buf1[64 * 64];         // Nn x Ni
    __shared__ half synapse_smem_buf2[64 * 64];         // Nn x Ni

    // Per-thread register buffer for prefetching next synapse tile
    // Each thread loads (Nn * Ni / blockDim.x) = 4096/128 = 32 elements
    half synapse_reg_buf[32];

    // Local pointers to track active/idle synapse buffers
    half *active_synapse_smem = synapse_smem_buf1;
    int active_buf_idx = 0;

    // Load neuron_i tile (10 x 18 x 64) into shared memory
    load_neuron_i_tile(neuron_i_smem, neuron_i,
        t_idx, batch_idx, y_out, x_out, Ni, blockDim.x, NYPAD, NXPAD);

    // Load synapse[0,0] (64x64) into shared memory
    load_synapse_xy_to_smem(active_synapse_smem, synapse,
        0, 0, t_idx, blockDim.x, Ni, Nn, kx_dim, ky_dim);
    __syncthreads();

    // Each warp handles one nn_tile (16 output channels)
    int nn_base = warp_id * 16;

    // WMMA fragments — per-warp
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> toeplitz_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> filter_frag;

    // One accumulator per y-row (8 y-rows per block)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[8];
    for (int y = 0; y < 8; y++)
        wmma::fill_fragment(c_frag[y], 0.0f);

    // Main loop over filter positions
    for (int k_row = 0; k_row < ky_dim; k_row++){
        for (int k_col = 0; k_col < kx_dim; k_col++){
            // Determine next filter position for prefetching
            int next_k_row = k_row;
            int next_k_col = k_col + 1;
            if (next_k_col >= kx_dim){
                next_k_row = k_row + 1;
                next_k_col = 0;
            }
            bool has_next = (next_k_row < ky_dim);

            // Prefetch next synapse tile from global → registers
            if (has_next){
                load_synapse_xy_from_global(synapse_reg_buf, synapse,
                    next_k_row, next_k_col, t_idx, blockDim.x, Ni, Nn, kx_dim, ky_dim);
            }

            // Compute: for each y-row, accumulate over input channels
            for (int y_local = 0; y_local < 8; y_local++){
                for (int j = 0; j < Ni / 16; j++){
                    // Load toeplitz (input) fragment from shared memory
                    // Tile-local coordinates: row = (y_local + k_row), col = (k_col)
                    // smem layout: [ty][tx][ni], dims 10 x 18 x Ni
                    int tile_y = y_local + k_row;
                    int tile_x = k_col;
                    // The 16x16 WMMA tile: M=16 x-positions, K=16 input channels
                    // Starting at (tile_y, tile_x, j*16) in the smem tile
                    // Stride in smem between consecutive x-positions = Ni
                    int toeplitz_smem_offset = (tile_y * 18 * Ni) + (tile_x * Ni) + (j * 16);
                    wmma::load_matrix_sync(toeplitz_frag,
                        neuron_i_smem + toeplitz_smem_offset, Ni);

                    // Load filter fragment from active synapse buffer
                    // synapse_smem layout: [Nn][Ni] = [64][64], row-major
                    // We want rows nn_base..(nn_base+15), cols j*16..(j*16+15)
                    int filter_smem_offset = (nn_base * Ni) + (j * 16);
                    wmma::load_matrix_sync(filter_frag,
                        active_synapse_smem + filter_smem_offset, Ni);

                    // Accumulate
                    wmma::mma_sync(c_frag[y_local], toeplitz_frag, filter_frag, c_frag[y_local]);
                }
            }

            __syncthreads();

            // Copy prefetched registers → idle smem buffer
            if (has_next){
                half *idle_synapse_smem = (active_buf_idx == 0)
                    ? synapse_smem_buf2 : synapse_smem_buf1;
                load_synapse_xy_from_local(idle_synapse_smem, synapse_reg_buf,
                    t_idx, blockDim.x, Ni, Nn);
            }
            __syncthreads();

            // Swap active buffer
            active_buf_idx ^= 1;
            active_synapse_smem = (active_buf_idx == 0)
                ? synapse_smem_buf1 : synapse_smem_buf2;
        }
    }

    // Store accumulators to global memory
    for (int y_local = 0; y_local < 8; y_local++){
        int out_idx = (batch_idx * NYSCL * NXSCL * Nn)
                    + ((y_out + y_local) * NXSCL * Nn)
                    + (x_out * Nn)
                    + nn_base;
        wmma::store_matrix_sync(neuron_n + out_idx, c_frag[y_local],
            Nn, wmma::mem_row_major);
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

    // Element counts
    long long syn_elems  = (long long)KY * KX * Nn * Ni;
    long long inp_elems  = (long long)B * NYPAD * NXPAD * Ni;
    long long out_elems  = (long long)B * NYSCL * NXSCL * Nn;

    // Byte sizes
    size_t syn_float_bytes = syn_elems * sizeof(float);
    size_t inp_float_bytes = inp_elems * sizeof(float);
    size_t syn_half_bytes  = syn_elems * sizeof(half);
    size_t inp_half_bytes  = inp_elems * sizeof(half);
    size_t out_float_bytes = out_elems * sizeof(float);

    // Device allocations
    float *synapse_device_float, *neuron_i_device_float;
    half  *synapse_device_half,  *neuron_i_device_half;
    float *neuron_n_device;

    cudaMalloc((void**)&synapse_device_float,  syn_float_bytes);
    cudaMalloc((void**)&neuron_i_device_float, inp_float_bytes);
    cudaMalloc((void**)&synapse_device_half,   syn_half_bytes);
    cudaMalloc((void**)&neuron_i_device_half,  inp_half_bytes);
    cudaMalloc((void**)&neuron_n_device,       out_float_bytes);

    // Copy float data host → device
    cudaMemcpy(synapse_device_float,  synapse,  syn_float_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(neuron_i_device_float, neuron_i, inp_float_bytes, cudaMemcpyHostToDevice);

    // Convert float → half on device
    int conversion_threads = 256;
    {
        int n = (int)syn_elems;
        int blocks = min((n + conversion_threads - 1) / conversion_threads, 2048);
        float2half_kernel<<<blocks, conversion_threads>>>(
            synapse_device_float, synapse_device_half, n);
    }
    {
        int n = (int)inp_elems;
        int blocks = min((n + conversion_threads - 1) / conversion_threads, 2048);
        float2half_kernel<<<blocks, conversion_threads>>>(
            neuron_i_device_float, neuron_i_device_half, n);
    }
    cudaDeviceSynchronize();
    cudaFree(synapse_device_float);
    cudaFree(neuron_i_device_float);

    // Launch convolution kernel
    dim3 blockDim(128);
    dim3 gridDim(B, NYSCL / 8, NXSCL / 16);

    convolution_kernel<<<gridDim, blockDim>>>(
        synapse_device_half, neuron_i_device_half, neuron_n_device,
        B, NYPAD, NXPAD, NYSCL, NXSCL, Ni, Nn, KY, KX);
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(neuron_n, neuron_n_device, out_float_bytes, cudaMemcpyDeviceToHost);

    // Cleanup
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

    printf("\n");
    return 0;
}