/*
 * attention.cpp -- CPU reference implementations of attention.
 *
 * Three functions:
 *   standard_prefill  -- full sequence, two-pass softmax
 *   flash_prefill     -- full sequence, online softmax (no score matrix)
 *   standard_decode   -- single query vs KV cache, two-pass softmax
 *
 * Prefill functions are parallelised with OpenMP (query rows are independent).
 * Decode is single-threaded (only one query row).
 *
 * Compile: g++ -O3 -march=native -fopenmp -std=c++17 -o attention attention.cpp -lm
 *
 * Test cases run by main():
 *   Prefill  S = 4096   (both standard and flash)
 *   Prefill  S = 65536  (flash only -- standard is O(S^2))
 *   Decode   C = 4096
 *   Decode   C = 65536
 */

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "timing.h"

#define D 64   /* head dimension */

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

static float dot(const float *a, const float *b)
{
    float sum = 0.f;
    for (int i = 0; i < D; i++) sum += a[i] * b[i];
    return sum;
}

static void fill(float *m, long long n, int seed)
{
    for (long long i = 0; i < n; i++)
        m[i] = 0.1f * sinf((float)(i * 3 + seed * 7));
}

/* ------------------------------------------------------------------ */
/* standard_prefill                                                    */
/*                                                                     */
/* For each query i:                                                   */
/*   pass 1 -- score all keys j <= i, find max                        */
/*   pass 2 -- exp(score - max), sum, divide                          */
/*   pass 3 -- weighted sum of value rows                             */
/* ------------------------------------------------------------------ */
void standard_prefill(
    const float Q[][D],   /* [S, D] */
    const float K[][D],   /* [S, D] */
    const float V[][D],   /* [S, D] */
    float       O[][D],   /* [S, D] */
    int S)
{
    float inv_sqrt = 1.f / sqrtf((float)D);

    /* one scores[S] buffer per thread -- no per-iteration malloc */
    int nthreads = omp_get_max_threads();
    float *score_bufs = (float *)malloc((long long)nthreads * S * sizeof(float));

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < S; i++) {

        float *scores = score_bufs + (long long)omp_get_thread_num() * S;

        for (int j = 0; j <= i; j++)
            scores[j] = dot(Q[i], K[j]) * inv_sqrt;

        float mx = scores[0];
        for (int j = 1; j <= i; j++)
            if (scores[j] > mx) mx = scores[j];

        float sum = 0.f;
        for (int j = 0; j <= i; j++) { scores[j] = expf(scores[j] - mx); sum += scores[j]; }
        for (int j = 0; j <= i; j++) scores[j] /= sum;

        for (int d = 0; d < D; d++) {
            float acc = 0.f;
            for (int j = 0; j <= i; j++) acc += scores[j] * V[j][d];
            O[i][d] = acc;
        }
    }

    free(score_bufs);
}

/* ------------------------------------------------------------------ */
/* flash_prefill                                                       */
/*                                                                     */
/* For each query i:                                                   */
/*   stream keys/values one at a time, maintaining:                   */
/*     m   -- running max score                                        */
/*     d   -- running normalizer                                       */
/*     out -- running output accumulator                               */
/*   when a new score exceeds m, rescale previous accumulations       */
/*   by exp(m_old - m_new) before adding the new contribution.        */
/* ------------------------------------------------------------------ */
void flash_prefill(
    const float Q[][D],   /* [S, D] */
    const float K[][D],   /* [S, D] */
    const float V[][D],   /* [S, D] */
    float       O[][D],   /* [S, D] */
    int S)
{
    float inv_sqrt = 1.f / sqrtf((float)D);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < S; i++) {

        float out[D] = {};
        float m = -FLT_MAX, d = 0.f;

        for (int j = 0; j <= i; j++) {

            float score = dot(Q[i], K[j]) * inv_sqrt;

            float m_new      = (score > m) ? score : m;
            float correction = expf(m - m_new);
            float exp_score  = expf(score - m_new);

            d = d * correction + exp_score;
            for (int h = 0; h < D; h++)
                out[h] = out[h] * correction + exp_score * V[j][h];

            m = m_new;
        }

        for (int h = 0; h < D; h++) O[i][h] = out[h] / d;
    }
}

/* ------------------------------------------------------------------ */
/* standard_decode                                                     */
/*                                                                     */
/* Single query q attends to all C positions in the KV cache.         */
/* ------------------------------------------------------------------ */
void standard_decode(
    const float  q[D],    /* [D]    -- single query vector */
    const float  K[][D],  /* [C, D] -- key cache           */
    const float  V[][D],  /* [C, D] -- value cache         */
    float        o[D],    /* [D]    -- output vector       */
    int C)
{
    float inv_sqrt = 1.f / sqrtf((float)D);
    float *scores = (float *)malloc(C * sizeof(float));

    for (int j = 0; j < C; j++)
        scores[j] = dot(q, K[j]) * inv_sqrt;

    float mx = scores[0];
    for (int j = 1; j < C; j++) if (scores[j] > mx) mx = scores[j];

    float sum = 0.f;
    for (int j = 0; j < C; j++) { scores[j] = expf(scores[j] - mx); sum += scores[j]; }
    for (int j = 0; j < C; j++) scores[j] /= sum;

    for (int d = 0; d < D; d++) {
        float acc = 0.f;
        for (int j = 0; j < C; j++) acc += scores[j] * V[j][d];
        o[d] = acc;
    }

    free(scores);
}

/* ------------------------------------------------------------------ */
/* Benchmark runners                                                   */
/* ------------------------------------------------------------------ */

static void run_prefill(int S, bool skip_standard = false)
{
    float (*Q)[D] = (float(*)[D]) malloc((long long)S*D*sizeof(float));
    float (*K)[D] = (float(*)[D]) malloc((long long)S*D*sizeof(float));
    float (*V)[D] = (float(*)[D]) malloc((long long)S*D*sizeof(float));
    float (*O)[D] = (float(*)[D]) malloc((long long)S*D*sizeof(float));
    fill(Q[0], (long long)S*D, 1);
    fill(K[0], (long long)S*D, 2);
    fill(V[0], (long long)S*D, 3);

    printf("PREFILL  S=%d\n", S);
    if (skip_standard)
        printf("  %-38s  %8s  (O(S^2) -- skipped)\n", "standard_prefill", "---");
    else
        bench("standard_prefill", [&]{ standard_prefill(Q, K, V, O, S); });
    bench("flash_prefill", [&]{ flash_prefill(Q, K, V, O, S); });
    printf("\n");

    free(Q); free(K); free(V); free(O);
}

static void run_decode(int C)
{
    float (*q)[D] = (float(*)[D]) malloc(D * sizeof(float));
    float (*K)[D] = (float(*)[D]) malloc((long long)C*D*sizeof(float));
    float (*V)[D] = (float(*)[D]) malloc((long long)C*D*sizeof(float));
    float (*o)[D] = (float(*)[D]) malloc(D * sizeof(float));
    fill(q[0], D, 1);
    fill(K[0], (long long)C*D, 2);
    fill(V[0], (long long)C*D, 3);

    printf("DECODE   C=%d\n", C);
    bench("standard_decode", [&]{ standard_decode(q[0], K, V, o[0], C); });
    printf("\n");

    free(q); free(K); free(V); free(o);
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */
int main(void)
{
    printf("=== Attention CPU Reference  D=%d  threads=%d ===\n\n",
           D, omp_get_max_threads());

    run_prefill(4096);
    run_prefill(65536, /*skip_standard=*/true);
    run_decode(4096);
    run_decode(65536);

    return 0;
}
