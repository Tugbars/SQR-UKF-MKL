/**
 * @file bench_srukf.c
 * @brief Performance benchmark for Student-t SQR UKF
 *
 * Demonstrates performance difference between:
 *   1. Default MKL settings (unconfigured)
 *   2. i9-14900K optimized settings
 */

#include "student_t_srukf.h"
#include "mkl_config_14900k.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*─────────────────────────────────────────────────────────────────────────────
 * Benchmark core: measures a single configuration
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct
{
    double predict_ns;
    double update_ns;
    double step_ns;
} BenchResult;

static BenchResult run_benchmark(int nx, int nz, int iterations)
{
    BenchResult result = {0};

    StudentT_SRUKF *ukf = srukf_create(nx, nz, 4.0);
    if (!ukf)
    {
        fprintf(stderr, "Failed to create UKF\n");
        return result;
    }

    /* Initialize */
    double *x0 = calloc(nx, sizeof(double));
    double *S0 = calloc(nx * nx, sizeof(double));
    double *F = calloc(nx * nx, sizeof(double));
    double *H = calloc(nz * nx, sizeof(double));
    double *Sq = calloc(nx * nx, sizeof(double));
    double *R0 = calloc(nz * nz, sizeof(double));
    double *z = calloc(nz, sizeof(double));

    x0[nx - 1] = -3.0; /* log-vol */

    for (int i = 0; i < nx; i++)
    {
        S0[i + i * nx] = 0.1;
        F[i + i * nx] = 1.0;
        Sq[i + i * nx] = 0.01;
    }
    for (int i = 0; i < nz; i++)
    {
        H[i + i * nx] = 1.0;
        R0[i + i * nz] = 0.1;
    }

    srukf_set_state(ukf, x0);
    srukf_set_sqrt_cov(ukf, S0);
    srukf_set_dynamics(ukf, F);
    srukf_set_measurement(ukf, H);
    srukf_set_process_noise(ukf, Sq);
    srukf_set_measurement_noise(ukf, R0);

    /* Warmup */
    for (int i = 0; i < 1000; i++)
    {
        srukf_step(ukf, z);
    }

    /* Benchmark predict */
    double start = mkl_14900k_get_time_ns();
    for (int i = 0; i < iterations; i++)
    {
        srukf_predict(ukf);
    }
    double end = mkl_14900k_get_time_ns();
    result.predict_ns = (end - start) / iterations;

    /* Benchmark update */
    start = mkl_14900k_get_time_ns();
    for (int i = 0; i < iterations; i++)
    {
        srukf_update(ukf, z);
    }
    end = mkl_14900k_get_time_ns();
    result.update_ns = (end - start) / iterations;

    /* Benchmark full step */
    start = mkl_14900k_get_time_ns();
    for (int i = 0; i < iterations; i++)
    {
        srukf_step(ukf, z);
    }
    end = mkl_14900k_get_time_ns();
    result.step_ns = (end - start) / iterations;

    free(x0);
    free(S0);
    free(F);
    free(H);
    free(Sq);
    free(R0);
    free(z);
    srukf_destroy(ukf);

    return result;
}

/*─────────────────────────────────────────────────────────────────────────────
 * Print comparison table row
 *───────────────────────────────────────────────────────────────────────────*/

static void print_comparison(const char *label,
                             BenchResult before, BenchResult after)
{
    double step_speedup = before.step_ns / after.step_ns;

    printf("║ %-20s │ %7.0f ns │ %7.0f ns │ %5.2fx      ║\n",
           label, before.step_ns, after.step_ns, step_speedup);
}

/*─────────────────────────────────────────────────────────────────────────────
 * Main
 *───────────────────────────────────────────────────────────────────────────*/

int main(void)
{
    int iterations = 100000;

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║         Student-t SQR UKF Benchmark - MKL Configuration Test         ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════╣\n");
    printf("║ Target: Intel Core i9-14900K                                         ║\n");
    printf("║ Iterations per test: %d                                          ║\n", iterations);
    printf("╚══════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    /*─────────────────────────────────────────────────────────────────────────
     * PHASE 1: Benchmark with DEFAULT MKL settings (unconfigured)
     *───────────────────────────────────────────────────────────────────────*/

    printf("┌──────────────────────────────────────────────────────────────────────┐\n");
    printf("│ PHASE 1: Default MKL Settings (unconfigured)                         │\n");
    printf("│   - Threading: MKL default (may use multiple threads)                │\n");
    printf("│   - SIMD: Auto-detected                                              │\n");
    printf("│   - CNR: Disabled                                                    │\n");
    printf("│   - Core affinity: None (OS scheduler decides)                       │\n");
    printf("└──────────────────────────────────────────────────────────────────────┘\n");
    printf("\n");

    printf("Running benchmarks with default settings...\n");

    BenchResult default_3_1 = run_benchmark(3, 1, iterations);
    BenchResult default_5_1 = run_benchmark(5, 1, iterations);
    BenchResult default_10_3 = run_benchmark(10, 3, iterations);

    printf("  nx=3,  nz=1:  step = %.0f ns (%.2f µs)\n",
           default_3_1.step_ns, default_3_1.step_ns / 1000.0);
    printf("  nx=5,  nz=1:  step = %.0f ns (%.2f µs)\n",
           default_5_1.step_ns, default_5_1.step_ns / 1000.0);
    printf("  nx=10, nz=3:  step = %.0f ns (%.2f µs)\n",
           default_10_3.step_ns, default_10_3.step_ns / 1000.0);
    printf("\n");

    /*─────────────────────────────────────────────────────────────────────────
     * PHASE 2: Apply i9-14900K optimized configuration
     *───────────────────────────────────────────────────────────────────────*/

    printf("┌──────────────────────────────────────────────────────────────────────┐\n");
    printf("│ PHASE 2: Applying i9-14900K Optimized Configuration                  │\n");
    printf("└──────────────────────────────────────────────────────────────────────┘\n");
    printf("\n");

    /* Apply configuration */
    mkl_14900k_init_full();

    /* Print what we configured */
    mkl_14900k_print_info();

    /* Warmup after configuration change */
    printf("Warming up caches...\n");
    mkl_14900k_warmup(10, 3, 100);
    printf("\n");

    /*─────────────────────────────────────────────────────────────────────────
     * PHASE 3: Benchmark with OPTIMIZED settings
     *───────────────────────────────────────────────────────────────────────*/

    printf("┌──────────────────────────────────────────────────────────────────────┐\n");
    printf("│ PHASE 3: Optimized MKL Settings                                      │\n");
    printf("│   - Threading: Sequential (1 thread, no sync overhead)               │\n");
    printf("│   - SIMD: AVX2 locked (optimal for 14900K)                           │\n");
    printf("│   - CNR: AVX2 (deterministic results)                                │\n");
    printf("│   - Core affinity: P-core 0 (best single-thread boost)               │\n");
    printf("└──────────────────────────────────────────────────────────────────────┘\n");
    printf("\n");

    printf("Running benchmarks with optimized settings...\n");

    BenchResult opt_3_1 = run_benchmark(3, 1, iterations);
    BenchResult opt_5_1 = run_benchmark(5, 1, iterations);
    BenchResult opt_10_3 = run_benchmark(10, 3, iterations);

    printf("  nx=3,  nz=1:  step = %.0f ns (%.2f µs)\n",
           opt_3_1.step_ns, opt_3_1.step_ns / 1000.0);
    printf("  nx=5,  nz=1:  step = %.0f ns (%.2f µs)\n",
           opt_5_1.step_ns, opt_5_1.step_ns / 1000.0);
    printf("  nx=10, nz=3:  step = %.0f ns (%.2f µs)\n",
           opt_10_3.step_ns, opt_10_3.step_ns / 1000.0);
    printf("\n");

    /*─────────────────────────────────────────────────────────────────────────
     * PHASE 4: Comparison Summary
     *───────────────────────────────────────────────────────────────────────*/

    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║                        COMPARISON SUMMARY                            ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════╣\n");
    printf("║ Configuration          │   Default   │  Optimized  │  Speedup    ║\n");
    printf("╠════════════════════════╪═════════════╪═════════════╪═════════════╣\n");

    print_comparison("UKF step (nx=3, nz=1)", default_3_1, opt_3_1);
    print_comparison("UKF step (nx=5, nz=1)", default_5_1, opt_5_1);
    print_comparison("UKF step (nx=10, nz=3)", default_10_3, opt_10_3);

    printf("╠══════════════════════════════════════════════════════════════════════╣\n");

    /* Calculate averages */
    double avg_default = (default_3_1.step_ns + default_5_1.step_ns + default_10_3.step_ns) / 3.0;
    double avg_opt = (opt_3_1.step_ns + opt_5_1.step_ns + opt_10_3.step_ns) / 3.0;
    double avg_speedup = avg_default / avg_opt;

    printf("║ Average                │ %7.0f ns │ %7.0f ns │ %5.2fx      ║\n",
           avg_default, avg_opt, avg_speedup);
    printf("╚══════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    /*─────────────────────────────────────────────────────────────────────────
     * PHASE 5: Detailed breakdown for typical use case
     *───────────────────────────────────────────────────────────────────────*/

    printf("╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("║              DETAILED BREAKDOWN (nx=3, nz=1) - Typical UKF           ║\n");
    printf("╠══════════════════════════════════════════════════════════════════════╣\n");
    printf("║ Operation              │   Default   │  Optimized  │  Speedup    ║\n");
    printf("╠════════════════════════╪═════════════╪═════════════╪═════════════╣\n");
    printf("║ Predict                │ %7.0f ns │ %7.0f ns │ %5.2fx      ║\n",
           default_3_1.predict_ns, opt_3_1.predict_ns,
           default_3_1.predict_ns / opt_3_1.predict_ns);
    printf("║ Update                 │ %7.0f ns │ %7.0f ns │ %5.2fx      ║\n",
           default_3_1.update_ns, opt_3_1.update_ns,
           default_3_1.update_ns / opt_3_1.update_ns);
    printf("║ Full Step              │ %7.0f ns │ %7.0f ns │ %5.2fx      ║\n",
           default_3_1.step_ns, opt_3_1.step_ns,
           default_3_1.step_ns / opt_3_1.step_ns);
    printf("╠══════════════════════════════════════════════════════════════════════╣\n");
    printf("║ Throughput             │ %5.2f M/s  │ %5.2f M/s  │             ║\n",
           1e9 / default_3_1.step_ns / 1e6, 1e9 / opt_3_1.step_ns / 1e6);
    printf("╚══════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    /*─────────────────────────────────────────────────────────────────────────
     * Verification
     *───────────────────────────────────────────────────────────────────────*/

    printf("Configuration verification:\n");
    mkl_14900k_verify();
    printf("\n");

    printf("Benchmark complete.\n");
    printf("\n");

    return 0;
}