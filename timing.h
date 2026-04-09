/*
 * timing.h -- Single-run timing helper.
 *
 * Usage:
 *   bench("label", [&]{ my_function(args...); });          // time only
 *   bench("label", [&]{ my_function(args...); }, flops);   // time + GFLOPS
 */

#pragma once
#include <stdio.h>
#include <time.h>

static double now_ms()
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}

template<typename F>
static void bench(const char *label, F fn, long long flops = 0)
{
    double t0 = now_ms();
    fn();
    double ms = now_ms() - t0;
    printf("  %-38s  %8.2f ms", label, ms);
    if (flops > 0)
        printf("  %8.2f GFLOPS", (double)flops * 1e-9 / (ms * 1e-3));
    printf("\n");
}
