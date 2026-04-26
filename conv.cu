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

__global__ void float2half_kernel(const float * in, half * out, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < size; idx += blockDim.x * gridDim.x){
        out[idx] = __float2half(in[idx]);
    }
}

/* ------------------------------------------------------------------ */
/* Device helper functions (moved above kernel for forward decl)      */
/* ------------------------------------------------------------------ */

__device__ void load_neuron_i_tile(
    half* neuron_i_smem,       // FIX: was float*
    half* neuron_i_global,     // FIX: was float*
    int t_idx, 
    int batch_idx, int y_out, int x_out, int Ni, 
    int blockDim_x, int NYPAD, int NXPAD
)
{
    for (int i = 0; i < 10; i++){
        for (int j = 0; j < 9; j++){
            int local_idx = (i * 9 * blockDim_x) + (j * blockDim_x) + t_idx;
            int global_idx = (batch_idx * NYPAD * NXPAD * Ni) + ((y_out + i) * NXPAD * Ni) + ((x_out + j*2) * Ni) + t_idx; 
            neuron_i_smem[local_idx] = neuron_i_global[global_idx];
        }
    }
    __syncthreads();
}

__device__ void load_synapse_xy_from_global(
    half* synapse_reg_buf,     // FIX: was float*
    half* synapse_global,      // FIX: was float*
    int k_row, int k_col, int t_idx, int blockDim_x, 
    int Ni, int Nn, int kx_dim, int ky_dim
)
{
    int elements_per_thread = (Nn * Ni) / blockDim_x;  // FIX: was Nn_over_2 = (Nn+1)/2
    int base = (k_row * kx_dim * Nn * Ni) + (k_col * Nn * Ni);
    for (int k = 0; k < elements_per_thread; k++){
        int global_flat = base + k * blockDim_x + t_idx;
        synapse_reg_buf[k] = synapse_global[global_flat];  // FIX: index k, not k*blockDim_x+t_idx
    }
}

__device__ void load_synapse_xy_from_local(
    half* synapse_smem,        // FIX: was float*
    half* synapse_reg_buf,     // FIX: was float*
    int k_row, int k_col, int t_idx, int blockDim_x, 
    int Ni, int Nn, int kx_dim, int ky_dim
)
{
    int elements_per_thread = (Nn * Ni) / blockDim_x;  // FIX: was Nn_over_2
    for (int k = 0; k < elements_per_thread; k++){
        int smem_flat = k * blockDim_x + t_idx;
        synapse_smem[smem_flat] = synapse_reg_buf[k];  // FIX: read [k], write interleaved
    }
}


__device__ void load_synapse_row_tile(
    half* synapse_row_smem,    // FIX: was float*
    half* synapse_global,      // FIX: was float*
    int k_row, int t_idx, int blockDim_x, 
    int Ni, int Nn, int kx_dim
)
{
    int elements_per_thread = (Nn * Ni) / blockDim_x;  // FIX: was Nn_over_2
    for (int j = 0; j < kx_dim; j++){
        int base = (k_row * kx_dim * Nn * Ni) + (j * Nn * Ni);
        for (int k = 0; k < elements_per_thread; k++){
            int smem_flat = (j * Nn * Ni) + (k * blockDim_x) + t_idx;
            int global_flat = base + (k * blockDim_x) + t_idx;
            synapse_row_smem[smem_flat] = synapse_global[global_flat];
        }
    }
    __syncthreads();
}

/* ------------------------------------------------------------------ */
/* CUDA kernel for convolution                                        */
/* ------------------------------------------------------------------ */
__global__ void convolution_kernel(
    half *synapse,    /* [KY * KX * Nn * Ni]         */
    half *neuron_i,   /* [B * NYPAD * NXPAD * Ni]    */
    float *neuron_n,   /* [B * NYSCL * NXSCL * Nn]    */
    int B, int NYPAD, int NXPAD, int NYSCL, int NXSCL, int Ni, int Nn, int ky_dim, int kx_dim)
{
    // TODO: Calculate thread indices and compute convolution
    // Grid Dim is (B, NYSCL / 8, NXSCL / 16)
    // Block Dim is (128, 1, 1)
    // each block does a 1x8x16x64 tile in output
    // each warp does a 1x1x16x64 tile in output
    int t_idx = threadIdx.x; //0 to 127
    int batch_idx = blockIdx.x;
    int y_out = blockIdx.y * 8;
    int x_out = blockIdx.z * 16;
    
    int pad_y = (ky_dim - 1) / 2;
    int pad_x = (kx_dim - 1) / 2;
    
    __shared__ half neuron_i_smem[1 * 10 * 18 * 64];
    __shared__ half synapse_smem_buf1[1 * 64 * 64];
    __shared__ half synapse_smem_buf2[1 * 64 * 64];
    half synapse_reg_buf[32];  // FIX: was 64*64; each thread only loads Nn*Ni/blockDim.x = 32
    
    // FIX: was __shared__ with & — __shared__ vars can't have initializers, & gives wrong type
    half *active_synapse_smem_buf = synapse_smem_buf1;
    half *idle_synapse_smem_buf = synapse_smem_buf2;
    int active_buf_idx = 0;
    
    // load neuron_i tile (1x10x18x64) into smem
    load_neuron_i_tile(neuron_i_smem, neuron_i, t_idx, batch_idx, y_out, x_out, Ni, blockDim.x, NYPAD, NXPAD);
    
    // load synapse[0,0] (1x1x64x64) into regs, then copy to smem with correct layout
    load_synapse_xy_from_global(synapse_reg_buf, synapse, 0, 0, t_idx, blockDim.x, Ni, Nn, kx_dim, ky_dim);
    load_synapse_xy_from_local(active_synapse_smem_buf, synapse_reg_buf, 0, 0, t_idx, blockDim.x, Ni, Nn, kx_dim, ky_dim);
    __syncthreads();

    // make wmma fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> toeplitz_frag;   // FIX: was wmmma
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> filter_frag;     // FIX: was wmmma
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;                       // FIX: was wmmma
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k_row = 0; k_row < ky_dim; k_row ++){  // FIX: was hardcoded 3
        for (int k_col = 0; k_col < kx_dim ; k_col ++){
            int next_k_row = k_row;
            int next_k_col = k_col + 1;
            if (next_k_col >= kx_dim){
                next_k_row = k_row + 1;
                next_k_col = 0;
            }
            bool has_next = (next_k_row < ky_dim);

            if (has_next){
                load_synapse_xy_from_global(synapse_reg_buf, synapse, next_k_row, next_k_col, t_idx, blockDim.x, Ni, Nn, kx_dim, ky_dim);  // FIX: was load_synapse_xy (undefined)
            }
            active_synapse_smem_buf = (active_buf_idx == 0) ? synapse_smem_buf1 : synapse_smem_buf2;
            
            for (int i = 0; i < Nn / 16; i ++){
                for(int j = 0; j < Ni / 16; j++){

                    // load toeplitz from smem into registers
                    int k_off_y = k_row - ky_dim / 2;
                    int k_off_x = k_col - kx_dim / 2;
                    
                    // DESIGN TODO: toeplitz_idx uses global coords (NXPAD) but indexes into smem tile.
                    // You need tile-local coordinates here.
                    int toeplitz_idx = ((y_out + pad_y + k_off_y) * NXPAD * Ni) + ((x_out + pad_x + k_off_x) * Ni); //TODO: Padding
                    wmma::load_matrix_sync(toeplitz_frag, neuron_i_smem + toeplitz_idx, Ni);  // FIX: was &neuron_i_smem

                    // load filter from smem into registers
                    // FIX: removed k_col * Nn * Ni — each smem buf is one [k_row][k_col] slice, no k_col dimension
                    int filter_idx = (i * 16 * Ni) + (j * 16);
                    wmma::load_matrix_sync(filter_frag, active_synapse_smem_buf + filter_idx, Ni);  // FIX: was active_synapse_row_smem_buf

                    // dispatch wmma
                    wmma::mma_sync(c_frag, toeplitz_frag, filter_frag, c_frag);
                }
            }

            __syncthreads();

            if (has_next){
                idle_synapse_smem_buf = (active_buf_idx == 0) ? synapse_smem_buf2 : synapse_smem_buf1;
                load_synapse_xy_from_local(idle_synapse_smem_buf, synapse_reg_buf, next_k_row, next_k_col, t_idx, blockDim.x, Ni, Nn, kx_dim, ky_dim);
            }
            __syncthreads();
            
            active_buf_idx ^= 1;
            
        }

        // gather wmma
        int out_idx = (batch_idx * NYSCL * NXSCL * Nn) + (y_out * NXSCL * Nn) + (x_out * Nn);
        wmma::store_matrix_sync(neuron_n + out_idx, c_frag, Nn, wmma::mem_row_major);  // FIX: was neuron_n (no offset)

        // DESIGN TODO: store is inside k_row loop — this stores partial results.
        // The accumulator should finish ALL k_row * k_col iterations before storing.

        // write output tile to global memory
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

    // FIX: compute element counts first, derive byte sizes from them
    long long syn_elems = (long long)KY * KX * Nn * Ni;
    long long inp_elems = (long long)B * NYPAD * NXPAD * Ni;    // FIX: was missing B
    long long out_elems = (long long)B * NYSCL * NXSCL * Nn;    // FIX: was Ni, missing B, was sizeof(half)

    size_t synapse_half_bytes  = syn_elems * sizeof(half);
    size_t neuron_i_half_bytes = inp_elems * sizeof(half);
    size_t synapse_float_bytes  = syn_elems * sizeof(float);
    size_t neuron_i_float_bytes = inp_elems * sizeof(float);
    size_t neuron_n_bytes = out_elems * sizeof(float);           // FIX: output is float, not half

    half *synapse_device_half, *neuron_i_device_half;
    float *synapse_device_float, *neuron_i_device_float;
    float *neuron_n_device;

    // allocate mem
    cudaMalloc((void**)&synapse_device_half, synapse_half_bytes);
    cudaMalloc((void**)&neuron_i_device_half, neuron_i_half_bytes);
    cudaMalloc((void**)&synapse_device_float, synapse_float_bytes);   // FIX: was half-sized
    cudaMalloc((void**)&neuron_i_device_float, neuron_i_float_bytes); // FIX: was half-sized
    cudaMalloc((void**)&neuron_n_device, neuron_n_bytes);

    // copy mem (float-sized)
    cudaMemcpy(synapse_device_float, synapse, synapse_float_bytes, cudaMemcpyHostToDevice);     // FIX: was half-sized
    cudaMemcpy(neuron_i_device_float, neuron_i, neuron_i_float_bytes, cudaMemcpyHostToDevice);  // FIX: was half-sized

    // convert float -> half
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
    
    // TODO: Calculate block and grid dimensions
    dim3 blockDim(128);
    dim3 gridDim(B, NYSCL / 8, NXSCL / 16);

    // TODO: Launch the kernel
    convolution_kernel<<<gridDim, blockDim>>>(
        synapse_device_half, neuron_i_device_half, neuron_n_device, 
        B, NYPAD, NXPAD, NYSCL, NXSCL, Ni, Nn, KY, KX);  // FIX: was missing NYPAD, NXPAD, NYSCL, NXSCL, KY, KX
    // TODO: Synchronize and check for errors
    cudaDeviceSynchronize();

    // read answer
    cudaMemcpy(neuron_n, neuron_n_device, neuron_n_bytes, cudaMemcpyDeviceToHost);
    
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