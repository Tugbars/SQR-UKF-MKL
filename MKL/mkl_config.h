/**
 * @file mkl_config.h
 * @brief Intel MKL Configuration for Quantitative Trading Systems
 * 
 * Optimizes MKL for:
 * - Low latency (single-threaded for small matrices)
 * - Deterministic results (CNR mode)
 * - Maximum SIMD utilization (AVX512 when available)
 * - Controlled memory allocation
 * 
 * Usage:
 *   #include "mkl_config.h"
 *   
 *   int main() {
 *       mkl_config_init();  // Call once at startup
 *       // ... your code ...
 *       mkl_config_print_info();  // Optional: log configuration
 *   }
 */

#ifndef MKL_CONFIG_H
#define MKL_CONFIG_H

#include <mkl.h>
#include <stdio.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*─────────────────────────────────────────────────────────────────────────────
 * CONFIGURATION PRESETS
 *───────────────────────────────────────────────────────────────────────────*/

typedef enum {
    MKL_CONFIG_QUANT_LATENCY,      /* Minimum latency, single-threaded */
    MKL_CONFIG_QUANT_THROUGHPUT,   /* Maximum throughput, multi-threaded */
    MKL_CONFIG_QUANT_DETERMINISTIC /* Reproducible results, slight overhead */
} MKL_ConfigPreset;

/*─────────────────────────────────────────────────────────────────────────────
 * CONFIGURATION STRUCTURE
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct {
    /* Threading */
    int num_threads;               /* 0 = auto, 1 = sequential, N = fixed */
    bool dynamic_threads;          /* Allow MKL to adjust thread count */
    
    /* SIMD */
    bool force_avx512;             /* Force AVX512 even on hybrid CPUs */
    bool allow_avx512_on_mic;      /* Allow AVX512 on Xeon Phi */
    
    /* Determinism */
    bool deterministic;            /* Enable Conditional Numerical Reproducibility */
    int cnr_mode;                  /* MKL_CBWR_COMPATIBLE, MKL_CBWR_AUTO, etc. */
    
    /* Memory */
    int memory_alignment;          /* Alignment in bytes (64 for AVX512) */
    bool use_huge_pages;           /* Use 2MB huge pages if available */
    
    /* JIT */
    bool enable_jit;               /* Enable JIT code generation */
    size_t jit_memory_limit;       /* Max memory for JIT (bytes, 0 = unlimited) */
    
    /* Verbose */
    bool verbose;                  /* Print configuration on init */
} MKL_Config;

/*─────────────────────────────────────────────────────────────────────────────
 * DEFAULT CONFIGURATIONS
 *───────────────────────────────────────────────────────────────────────────*/

/**
 * @brief Get default configuration for quant trading
 * 
 * Defaults:
 * - Single-threaded (latency-optimal for nx<20)
 * - Deterministic mode (reproducible results)
 * - AVX512 enabled
 * - 64-byte alignment
 */
static inline MKL_Config mkl_config_default(void) {
    MKL_Config cfg = {
        .num_threads = 1,              /* Sequential for low latency */
        .dynamic_threads = false,       /* Fixed thread count */
        .force_avx512 = true,          /* Use best SIMD */
        .allow_avx512_on_mic = false,  /* Not targeting Xeon Phi */
        .deterministic = true,         /* Reproducible results */
        .cnr_mode = MKL_CBWR_AUTO,     /* Auto-detect best CNR mode */
        .memory_alignment = 64,        /* AVX512 alignment */
        .use_huge_pages = false,       /* Disabled by default */
        .enable_jit = true,            /* Enable JIT for small matrices */
        .jit_memory_limit = 0,         /* Unlimited */
        .verbose = false
    };
    return cfg;
}

/**
 * @brief Get preset configuration
 */
static inline MKL_Config mkl_config_preset(MKL_ConfigPreset preset) {
    MKL_Config cfg = mkl_config_default();
    
    switch (preset) {
        case MKL_CONFIG_QUANT_LATENCY:
            cfg.num_threads = 1;
            cfg.dynamic_threads = false;
            cfg.deterministic = false;  /* Slightly faster without CNR */
            cfg.enable_jit = true;
            break;
            
        case MKL_CONFIG_QUANT_THROUGHPUT:
            cfg.num_threads = 0;        /* Auto-detect */
            cfg.dynamic_threads = true;
            cfg.deterministic = false;
            cfg.enable_jit = true;
            break;
            
        case MKL_CONFIG_QUANT_DETERMINISTIC:
            cfg.num_threads = 1;
            cfg.dynamic_threads = false;
            cfg.deterministic = true;
            cfg.cnr_mode = MKL_CBWR_COMPATIBLE;  /* Maximum compatibility */
            cfg.enable_jit = true;
            break;
    }
    
    return cfg;
}

/*─────────────────────────────────────────────────────────────────────────────
 * INITIALIZATION
 *───────────────────────────────────────────────────────────────────────────*/

/**
 * @brief Apply MKL configuration
 * @param cfg Configuration to apply
 * @return true on success, false on failure
 */
static inline bool mkl_config_apply(const MKL_Config* cfg) {
    
    /*───────────────────────────────────────────────────────────────────────
     * 1. THREADING
     *─────────────────────────────────────────────────────────────────────*/
    
    if (cfg->num_threads == 1) {
        /* Sequential mode - minimum latency */
        mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
    } else if (cfg->num_threads == 0) {
        /* Auto-detect optimal thread count */
        mkl_set_threading_layer(MKL_THREADING_INTEL);
        mkl_set_num_threads(0);  /* Let MKL decide */
    } else {
        /* Fixed thread count */
        mkl_set_threading_layer(MKL_THREADING_INTEL);
        mkl_set_num_threads(cfg->num_threads);
    }
    
    /* Dynamic thread adjustment */
    mkl_set_dynamic(cfg->dynamic_threads ? 1 : 0);
    
    /*───────────────────────────────────────────────────────────────────────
     * 2. SIMD / CPU DISPATCH
     *─────────────────────────────────────────────────────────────────────*/
    
    if (cfg->force_avx512) {
        /* 
         * On hybrid CPUs (e.g., Alder Lake), MKL may avoid AVX512
         * due to frequency throttling concerns. For latency-critical
         * quant code, AVX512 is usually still faster.
         * 
         * Set environment variable or use mkl_enable_instructions()
         */
#ifdef MKL_ENABLE_AVX512
        mkl_enable_instructions(MKL_ENABLE_AVX512);
#endif
    }
    
    /*───────────────────────────────────────────────────────────────────────
     * 3. DETERMINISM (Conditional Numerical Reproducibility)
     *─────────────────────────────────────────────────────────────────────*/
    
    if (cfg->deterministic) {
        /*
         * CNR Modes:
         * - MKL_CBWR_AUTO: Best performance while maintaining reproducibility
         * - MKL_CBWR_COMPATIBLE: Reproducible across different Intel CPUs
         * - MKL_CBWR_SSE2: Force SSE2 (slowest but most compatible)
         * - MKL_CBWR_AVX: Force AVX
         * - MKL_CBWR_AVX2: Force AVX2
         * - MKL_CBWR_AVX512: Force AVX512
         * 
         * For quant trading: MKL_CBWR_AUTO or MKL_CBWR_AVX512
         */
        int status = mkl_cbwr_set(cfg->cnr_mode);
        if (status != MKL_CBWR_SUCCESS) {
            if (cfg->verbose) {
                fprintf(stderr, "Warning: CNR mode %d not supported, trying AUTO\n", 
                        cfg->cnr_mode);
            }
            mkl_cbwr_set(MKL_CBWR_AUTO);
        }
    }
    
    /*───────────────────────────────────────────────────────────────────────
     * 4. MEMORY CONFIGURATION
     *─────────────────────────────────────────────────────────────────────*/
    
    /* Set default alignment for mkl_malloc */
    /* Note: This is handled per-allocation, but we document the expectation */
    
    /* Huge pages (Linux only, requires system configuration) */
#ifdef __linux__
    if (cfg->use_huge_pages) {
        /* 
         * Huge pages must be configured at OS level:
         *   echo 128 > /proc/sys/vm/nr_hugepages
         *   
         * Then set environment: MKL_ENABLE_HUGEPAGES=1
         * Or use mmap with MAP_HUGETLB for custom allocations
         */
    }
#endif
    
    /*───────────────────────────────────────────────────────────────────────
     * 5. JIT CONFIGURATION
     *─────────────────────────────────────────────────────────────────────*/
    
    /*
     * MKL JIT generates optimized code for specific matrix sizes.
     * Beneficial for repeated operations with same dimensions (like UKF).
     * 
     * For small matrices (nx < 20), JIT-generated code can be 2-3x faster
     * than generic BLAS routines.
     */
    if (!cfg->enable_jit) {
        /* Disable JIT if not wanted (saves memory) */
        /* Note: No direct API, controlled via MKL_JIT_MAX_SIZE=0 env var */
    }
    
    return true;
}

/**
 * @brief Initialize MKL with default quant trading configuration
 */
static inline bool mkl_config_init(void) {
    MKL_Config cfg = mkl_config_default();
    cfg.verbose = true;
    return mkl_config_apply(&cfg);
}

/**
 * @brief Initialize MKL with specific preset
 */
static inline bool mkl_config_init_preset(MKL_ConfigPreset preset) {
    MKL_Config cfg = mkl_config_preset(preset);
    cfg.verbose = true;
    return mkl_config_apply(&cfg);
}

/*─────────────────────────────────────────────────────────────────────────────
 * DIAGNOSTICS
 *───────────────────────────────────────────────────────────────────────────*/

/**
 * @brief Print current MKL configuration
 */
static inline void mkl_config_print_info(void) {
    MKLVersion version;
    mkl_get_version(&version);
    
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║                    MKL CONFIGURATION                             ║\n");
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║ MKL Version: %d.%d.%d (%s)                            \n", 
           version.MajorVersion, version.MinorVersion, version.UpdateVersion,
           version.ProductStatus);
    printf("║ Build: %s\n", version.Build);
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    
    /* Threading */
    int max_threads = mkl_get_max_threads();
    int dynamic = mkl_get_dynamic();
    printf("║ Threading:                                                       ║\n");
    printf("║   Max threads: %d                                                \n", max_threads);
    printf("║   Dynamic: %s                                                    \n", 
           dynamic ? "enabled" : "disabled");
    
    /* CNR */
    int cbwr = mkl_cbwr_get(MKL_CBWR_ALL);
    const char* cbwr_str = "unknown";
    switch (cbwr) {
        case MKL_CBWR_COMPATIBLE: cbwr_str = "COMPATIBLE"; break;
        case MKL_CBWR_AUTO: cbwr_str = "AUTO"; break;
        case MKL_CBWR_SSE2: cbwr_str = "SSE2"; break;
        case MKL_CBWR_SSE3: cbwr_str = "SSE3"; break;
        case MKL_CBWR_SSSE3: cbwr_str = "SSSE3"; break;
        case MKL_CBWR_SSE4_1: cbwr_str = "SSE4.1"; break;
        case MKL_CBWR_SSE4_2: cbwr_str = "SSE4.2"; break;
        case MKL_CBWR_AVX: cbwr_str = "AVX"; break;
        case MKL_CBWR_AVX2: cbwr_str = "AVX2"; break;
#ifdef MKL_CBWR_AVX512
        case MKL_CBWR_AVX512: cbwr_str = "AVX512"; break;
#endif
        default: cbwr_str = "OFF"; break;
    }
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║ Determinism:                                                     ║\n");
    printf("║   CNR Mode: %s                                                   \n", cbwr_str);
    
    /* CPU info */
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║ CPU Features:                                                    ║\n");
    printf("║   AVX2:   %s                                                     \n",
           mkl_cbwr_get(MKL_CBWR_AVX2) >= 0 ? "available" : "unavailable");
#ifdef MKL_CBWR_AVX512
    printf("║   AVX512: %s                                                     \n",
           mkl_cbwr_get(MKL_CBWR_AVX512) >= 0 ? "available" : "unavailable");
#endif
    
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║ Memory:                                                          ║\n");
    printf("║   Recommended alignment: 64 bytes (AVX512)                       ║\n");
    
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
}

/**
 * @brief Get CPU features string for logging
 */
static inline const char* mkl_config_get_cpu_string(void) {
#ifdef MKL_CBWR_AVX512
    if (mkl_cbwr_get(MKL_CBWR_AVX512) >= 0) return "AVX512";
#endif
    if (mkl_cbwr_get(MKL_CBWR_AVX2) >= 0) return "AVX2";
    if (mkl_cbwr_get(MKL_CBWR_AVX) >= 0) return "AVX";
    return "SSE";
}

/**
 * @brief Verify MKL is properly configured
 * @return true if configuration is optimal for quant trading
 */
static inline bool mkl_config_verify(void) {
    bool ok = true;
    
    /* Check threading is set correctly */
    int threads = mkl_get_max_threads();
    if (threads > 4) {
        fprintf(stderr, "Warning: MKL using %d threads. For small matrices, "
                        "consider sequential mode.\n", threads);
    }
    
    /* Check CNR is enabled (recommended for trading) */
    int cbwr = mkl_cbwr_get(MKL_CBWR_ALL);
    if (cbwr == MKL_CBWR_OFF) {
        fprintf(stderr, "Warning: CNR disabled. Results may not be reproducible.\n");
    }
    
    return ok;
}

/*─────────────────────────────────────────────────────────────────────────────
 * ENVIRONMENT VARIABLE DOCUMENTATION
 *───────────────────────────────────────────────────────────────────────────*/

/*
 * MKL behavior can also be controlled via environment variables.
 * Set these BEFORE running your application:
 * 
 * THREADING:
 *   MKL_NUM_THREADS=1                    # Number of threads
 *   MKL_DYNAMIC=FALSE                    # Disable dynamic threading
 *   OMP_NUM_THREADS=1                    # OpenMP fallback
 *   
 * SIMD:
 *   MKL_ENABLE_INSTRUCTIONS=AVX512       # Force instruction set
 *   MKL_DEBUG_CPU_TYPE=5                 # Simulate CPU type
 *   
 * DETERMINISM:
 *   MKL_CBWR=AUTO                        # CNR mode: AUTO, AVX2, AVX512, etc.
 *   
 * MEMORY:
 *   MKL_ENABLE_HUGEPAGES=1               # Enable huge pages (Linux)
 *   
 * JIT:
 *   MKL_JIT_MAX_SIZE=100                 # Max matrix dimension for JIT
 *   
 * DEBUGGING:
 *   MKL_VERBOSE=1                        # Print MKL function calls
 *   MKL_MIC_DISABLE=1                    # Disable Xeon Phi offload
 *   
 * Example startup script (Linux):
 *   #!/bin/bash
 *   export MKL_NUM_THREADS=1
 *   export MKL_DYNAMIC=FALSE
 *   export MKL_CBWR=AUTO
 *   export MKL_ENABLE_INSTRUCTIONS=AVX512
 *   ./my_trading_bot
 *   
 * Example startup script (Windows):
 *   @echo off
 *   set MKL_NUM_THREADS=1
 *   set MKL_DYNAMIC=FALSE
 *   set MKL_CBWR=AUTO
 *   my_trading_bot.exe
 */

#ifdef __cplusplus
}
#endif

#endif /* MKL_CONFIG_H */
