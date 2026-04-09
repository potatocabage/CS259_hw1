/*
 * conv.cpp -- CPU reference implementation of a convolution layer.
 *
 * Two VGG configurations:
 *   Conv1:  224x224  3x3  Ni=64   Nn=64   -- large spatial, moderate channels
 *   Conv2:   14x14   3x3  Ni=512  Nn=512  -- small spatial, deep channels
 *
 * Compile: g++ -O3 -march=native -fopenmp -std=c++17 -o conv conv.cpp -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "timing.h"

/* Convolution parameters shared across all VGG configurations */
#define KY  3   /* kernel height               */
#define KX  3   /* kernel width                */
#define SY  1   /* stride y                    */
#define SX  1   /* stride x                    */
#define TN 64   /* output-channel tile size    */

typedef float VTYPE;

static VTYPE relu(VTYPE x) { return x > 0.f ? x : 0.f; }

static void fill(VTYPE *m, long long n, float scale, int seed)
{
    for (long long i = 0; i < n; i++)
        m[i] = scale * sinf((float)(i * 3 + seed * 7));
}

/* ------------------------------------------------------------------ */
/* Convolution layer                                                   */
/*                                                                     */
/* synapse layout : [KY][KX][Nn][Ni]                                  */
/* neuron_i layout: [B*NYPAD][NXPAD][Ni]  (pre-padded input)          */
/* neuron_n layout: [B*NYSCL][NXSCL][Nn] (output)                     */
/*                                                                     */
/* The flat buffers are reinterpreted as typed VLA pointers so the    */
/* access pattern reads as plain array indexing.                       */
/* ------------------------------------------------------------------ */
static void convolution_layer(
    VTYPE *synapse,    /* [KY * KX * Nn * Ni]         */
    VTYPE *neuron_i,   /* [B * NYPAD * NXPAD * Ni]    */
    VTYPE *neuron_n,   /* [B * NYSCL * NXSCL * Nn]    */
    int B, int Ny, int Nx, int Ni, int Nn)
{
    int NXPAD = Nx + KX - 1,  NYPAD = Ny + KY - 1;
    int NXSCL = (Nx + SX - 1) / SX,  NYSCL = (Ny + SY - 1) / SY;

    /* reinterpret flat buffers as multi-dim arrays (GCC VLA extension) */
    VTYPE (*syn)[KX][Nn][Ni]   = (VTYPE(*)[KX][Nn][Ni])   synapse;
    VTYPE (*inp)[NXPAD][Ni]    = (VTYPE(*)[NXPAD][Ni])     neuron_i;
    VTYPE (*out)[NXSCL][Nn]    = (VTYPE(*)[NXSCL][Nn])     neuron_n;

    #pragma omp parallel for schedule(static) collapse(3)
    for (int b = 0; b < B; b++)
    for (int y = 0; y < Ny; y += SY)
    for (int x = 0; x < Nx; x += SX) {

        int yout = y / SY, xout = x / SX;

        for (int nn = 0; nn < Nn; nn += TN) {

            VTYPE sum[TN] = {};

            for (int ky = 0; ky < KY; ky++)
            for (int kx = 0; kx < KX; kx++)
            for (int n  = 0; n  < TN; n++)
            for (int i  = 0; i  < Ni; i++)
                sum[n] += syn[ky][kx][nn+n][i] * inp[b*NYPAD + ky+y][kx+x][i];

            for (int n = 0; n < TN; n++)
                out[b*NYSCL + yout][xout][nn+n] = relu(sum[n]);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Run one configuration and print timing                             */
/* ------------------------------------------------------------------ */
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
    bench(label, [&]{ convolution_layer(synapse, neuron_i, neuron_n, B, Ny, Nx, Ni, Nn); }, flops);

    free(synapse); free(neuron_i); free(neuron_n);
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */
int main(void)
{
    printf("=== Convolution CPU Reference  threads=%d ===\n\n",
           omp_get_max_threads());

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
