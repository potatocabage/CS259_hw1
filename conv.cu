#include <cuda_runtime.h>
#include <mma.h>
#include "timing.h"
#include <cuda_fp16.h>

// Convolution parameters
#define KY  3   /* kernel height */
#define KX  3   /* kernel width  */
#define SY  1   /* stride y      */
#define SX  1   /* stride x      */


__global__ void float2half_kernel(const float * in, half * out, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (; idx < size; idx += blockDim.x * gridDim.x){
        out[idx] = __float2half(in[idx]);
    }
}

/* ------------------------------------------------------------------ */
/* CUDA kernel for convolution                                        */
/* ------------------------------------------------------------------ */
__global__ void convolution_kernel(
    half *synapse,    /* [KY * KX * Nn * Ni]         */
    half *neuron_i,   /* [B * NYPAD * NXPAD * Ni]    */
    float *neuron_n,   /* [B * NYSCL * NXSCL * Nn]    */
    int B, int NYPAD, int NXPAD, int NYSCL, int NXSCL, int Ni, int Nn, int KY, int KX)
{
    // TODO: Calculate thread indices and compute convolution
    // Grid Dim is (B, NYSCL / 8, NXSCL / 16)
    // Block Dim is (8, 1, 1)
    // each block does a 1x8x16x64 tile in output
    // each warp does a 1x1x16x64 tile in output
    int t_idx = threadIdx.x; //0 to 127
    int batch_idx = blockIdx.x;
    int y_out = blockIdx.y * 8;
    int x_out = blockIdx.z * 16;
    
    int pad_y = (KY - 1) / 2;
    int pad_x = (KX - 1) / 2;
    
    __shared__ half neuron_i_smem[1 * 10 * 18 * 64];
    __shared__ half synapse_smem_buf1[1 * 64 * 64];
    __shared__ half synapse_smem_buf2[1 * 64 * 64];
    half synapse_reg_buf[1 * 64 * 64];
    
    __shared__ half *active_synapse_smem_buf = &synapse_smem_buf1;
    __shared__ half *idle_synapse_smem_buf = &synapse_smem_buf2;
    int active_buf_idx = 0;
    
    // load neuron_i tile (1x10x18x64) into smem
    load_neuron_i_tile(neuron_i_smem, neuron_i, t_idx, batch_idx, y_out, x_out, Ni, blockDim.x, NYPAD, NXPAD);
    
    // load synapse[0,0] (1x1x64x64) into smem
    load_synapse_xy_from_global(active_synapse_smem_buf, synapse, 0, 0, t_idx, blockDim.x, Ni, Nn, KX, KY);
    __syncthreads();

    // make wmma fragments
    wmmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> toeplitz_frag;
    wmmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> filter_frag;
    wmmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k_row = 0; k_row < 3; k_row ++){
        for (int k_col = 0; k_col < KX ; k_col ++){
            int next_k_row = k_row;
            int next_k_col = k_col + 1;
            if (next_k_col >= KX){
                next_k_row = k_row + 1;
                next_k_col = 0;
            }
            bool has_next = (next_k_row < KY);

            if (has_next){
                load_synapse_xy(synapse_reg_buf, synapse, next_k_row, next_k_col, t_idx, blockDim.x, Ni, Nn, KX, KY);
            }
            active_synapse_smem_buf = (active_buf_idx == 0) ? synapse_smem_buf1 : synapse_smem_buf2;
            
            for (int i = 0; i < Nn / 16; i ++){
                for(int j = 0; j < Ni / 16; j++){

                    // load toeplitz from smem into registers
                    int k_off_y = k_row - KY / 2;
                    int k_off_x = k_col - KX / 2;
                    

                    int toeplitz_idx = ((y_out + pad_y + k_off_y) * NXPAD * Ni) + ((x_out + pad_x + k_off_x) * Ni); //TODO: Padding
                    wmma::load_matrix_sync(toeplitz_frag, &neuron_i_smem + toeplitz_idx, Ni);

                    // load k_row from smem into registers
                    int filter_idx = (k_col * Nn * Ni) + (i * 16 * Ni) + (j * 16);
                    wmma::load_matrix_sync(filter_frag, active_synapse_row_smem_buf + filter_idx, Ni);

                    // dispatch wmma
                    wmma::mma_sync(c_frag, toeplitz_frag, filter_frag, c_frag);
                }
            }

            __syncthreads();

            if (has_next){
                idle_synapse_smem_buf = (active_buf_idx == 0) ? synapse_smem_buf2 : synapse_smem_buf1;
                load_synapse_xy_from_local(idle_synapse_smem_buf, synapse_reg_buf, next_k_row, next_k_col, t_idx, blockDim.x, Ni, Nn, KX, KY);
            }
            __syncthreads();
            
            active_buf_idx ^= 1;
            
        }

        // gather wmma
        int out_idx = (batch_idx * NYSCL * NXSCL * Nn) + (y_out * NXSCL * Nn) + (x_out * Nn);
        wmma::store_matrix_sync(neuron_n, c_frag, Nn, wmma::mem_row_major);

        // write output tile to global memory
    }
}

__device__ void load_neuron_i_tile(
    float* neuron_i_smem,
    float* neuron_i_global,
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
    float* synapse_reg_buf,
    float* synapse_global,
    int k_row, int k_col, int t_idx, int blockDim_x, 
    int Ni, int Nn, int KX, int KY
)
{
    int Nn_over_2 = (Nn+1)/2;
    for (int k = 0; k < Nn_over_2; k++){
        int local_idx = (k * blockDim_x) + t_idx;
        int global_idx = (k_row * KX * Nn * Ni) + (k_col * Nn * Ni) + (k * blockDim_x) + t_idx;
        synapse_reg_buf[local_idx] = synapse_global[global_idx];
    }
}

__device__ void load_synapse_xy_from_local(
    float* synapse_smem,
    float* synapse_reg_buf,
    int k_row, int k_col, int t_idx, int blockDim_x, 
    int Ni, int Nn, int KX, int KY
)
{
    int Nn_over_2 = (Nn+1)/2;
    for (int k = 0; k < Nn_over_2; k++){
        int local_idx = (k * blockDim_x) + t_idx;
        synapse_smem[local_idx] = synapse_reg_buf[local_idx];
    }
}


__device__ void load_synapse_row_tile(
    float* synapse_row_smem,
    float* synapse_global,
    int k_row, int t_idx, int blockDim_x, 
    int Ni, int Nn, int KX
)
{
    int Nn_over_2 = (Nn+1)/2;
    for (int j = 0; j < KX; j++){
        for (int k = 0; k < Nn_over_2; k++){
            int local_idx = (j * Nn * Ni) + (k * blockDim_x) + t_idx;
            int global_idx = (k_row * KX * Nn * Ni) + (j * Nn * Ni) + (k * blockDim_x) + t_idx;
            synapse_row_smem[local_idx] = synapse_global[global_idx];
        }
    }
    __syncthreads();
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

    size_t synapse_bytes = KY * KX * Nn * Ni * sizeof(half);
    size_t neuron_i_bytes = NYPAD * NXPAD * Ni * sizeof(half);
    size_t neuron_n_bytes = NYSCL * NXSCL * Ni * sizeof(half);

    half *synapse_device_half, *neuron_i_device_half;
    float *synapse_device_float, *neuron_i_device_float;
    float *neuron_n_device;

    // allocate mem
    cudaMalloc((void**)&synapse_device_half, synapse_bytes);
    cudaMalloc((void**)&neuron_i_device_half, neuron_i_bytes);
    cudaMalloc((void**)&synapse_device_float, synapse_bytes);
    cudaMalloc((void**)&neuron_i_device_float, neuron_i_bytes);
    cudaMalloc((void**)&neuron_n_device, neuron_n_bytes);

    // copy mem
    cudaMemcpy(synapse_device_float, synapse, synapse_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(neuron_i_device_float, neuron_i, neuron_i_bytes, cudaMemcpyHostToDevice);

    // convert
    int size[2] = {synapse_bytes/2, neuron_i_bytes/2};
    int conversion_threads = 256;
    for (int i = 0; i < 2; i++){
        int conversion_blockDim = min((size[i] + conversion_threads - 1) / conversion_threads, 2048);
        float2half_kernel<<<conversion_blockDim, conversion_threads>>>(
            (i==0)? synapse_device_float:neuron_i_device_float, 
            (i==0)? synapse_device_half:neuron_i_device_half, 
            size[i]);
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
        B, Ny, Nx, Ni, Nn);
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