/**
 * @file test_kelly.c
 * @brief Unit tests for Kelly criterion with UKF integration
 */

#include "kelly.h"
#include "student_t_srukf.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/*============================================================================
 * Test utilities
 *============================================================================*/

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s\n", msg); \
        printf("        at %s:%d\n", __FILE__, __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define TEST_ASSERT_NEAR(a, b, tol, msg) do { \
    double _a = (a), _b = (b), _tol = (tol); \
    if (fabs(_a - _b) > _tol) { \
        printf("  FAIL: %s\n", msg); \
        printf("        expected: %.6f, got: %.6f (diff: %.2e, tol: %.2e)\n", \
               _b, _a, fabs(_a - _b), _tol); \
        printf("        at %s:%d\n", __FILE__, __LINE__); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define RUN_TEST(test_fn) do { \
    printf("Running %s...\n", #test_fn); \
    test_fn(); \
    tests_passed++; \
    printf("  PASS\n"); \
} while(0)

/*============================================================================
 * Test 1: Basic Kelly formula
 *============================================================================*/

static void test_basic_kelly(void) {
    /* Basic Kelly: f* = μ / σ²
     * 
     * μ = 0.10 (10% return)
     * σ = 0.20 (20% volatility)
     * f* = 0.10 / 0.04 = 2.5
     * 
     * Half Kelly = 1.25
     * But capped at KELLY_MAX_LEVERAGE = 2.0
     */
    
    double mu = 0.10;
    double sigma = 0.20;
    double nu = INFINITY;  /* Gaussian */
    double fraction = 0.5;
    
    double f = kelly_simple(mu, sigma, nu, fraction);
    
    /* Expected: 0.5 × 0.10 / 0.04 = 1.25 */
    TEST_ASSERT_NEAR(f, 1.25, 1e-6, "Half Kelly calculation");
    
    /* Full Kelly would be 2.5, but capped at 2.0 */
    double f_full = kelly_simple(mu, sigma, nu, 1.0);
    TEST_ASSERT_NEAR(f_full, KELLY_MAX_LEVERAGE, 1e-6, "Full Kelly capped");
    
    /* Negative μ → short position */
    double f_short = kelly_simple(-0.05, sigma, nu, 0.5);
    /* Expected: 0.5 × (-0.05) / 0.04 = -0.625, then × KELLY_SHORT_PENALTY = -0.3125 */
    TEST_ASSERT_NEAR(f_short, -0.3125, 1e-6, "Short position with penalty");
}

/*============================================================================
 * Test 2: Tail-adjusted Kelly (Student-t)
 *============================================================================*/

static void test_tail_kelly(void) {
    /* Student-t inflates variance by ν/(ν-2)
     * 
     * ν = 4: multiplier = 4/2 = 2.0
     * ν = 6: multiplier = 6/4 = 1.5
     * ν = ∞: multiplier = 1.0 (Gaussian)
     */
    
    double mu = 0.10;
    double sigma = 0.20;
    double fraction = 1.0;
    
    /* Gaussian */
    double f_gauss = kelly_simple(mu, sigma, INFINITY, fraction);
    /* f* = 0.10 / 0.04 = 2.5, capped at 2.0 */
    TEST_ASSERT_NEAR(f_gauss, 2.0, 1e-6, "Gaussian Kelly (capped)");
    
    /* ν = 4: variance × 2, so f* = 2.5 / 2 = 1.25 */
    double f_nu4 = kelly_simple(mu, sigma, 4.0, fraction);
    TEST_ASSERT_NEAR(f_nu4, 1.25, 1e-6, "Student-t ν=4 Kelly");
    
    /* ν = 6: variance × 1.5, so f* = 2.5 / 1.5 = 1.667 */
    double f_nu6 = kelly_simple(mu, sigma, 6.0, fraction);
    TEST_ASSERT_NEAR(f_nu6, 1.6667, 1e-3, "Student-t ν=6 Kelly");
    
    /* Very heavy tails (ν = 2.5): should be very conservative */
    double f_heavy = kelly_simple(mu, sigma, 2.5, fraction);
    TEST_ASSERT(f_heavy < f_nu4, "Heavy tails → smaller position");
}

/*============================================================================
 * Test 3: Bayesian variance from log-volatility
 *============================================================================*/

static void test_bayesian_variance(void) {
    /* E[σ²] = exp(2μ_lv + 2σ_lv²)
     * 
     * If log_vol = -2 (so σ = exp(-2) ≈ 0.135):
     *   With no uncertainty (σ_lv = 0): E[σ²] = exp(-4) ≈ 0.0183
     *   With uncertainty (σ_lv = 0.5): E[σ²] = exp(-4 + 0.5) = exp(-3.5) ≈ 0.0302
     */
    
    double mu_lv = -2.0;
    
    /* No uncertainty */
    double E_var_certain = kelly_expected_variance(mu_lv, 0.0);
    double expected_certain = exp(-4.0);
    TEST_ASSERT_NEAR(E_var_certain, expected_certain, 1e-6, "E[σ²] with no uncertainty");
    
    /* With uncertainty */
    double sigma_lv = 0.5;
    double E_var_uncertain = kelly_expected_variance(mu_lv, sigma_lv);
    double expected_uncertain = exp(2.0 * mu_lv + 2.0 * sigma_lv * sigma_lv);
    TEST_ASSERT_NEAR(E_var_uncertain, expected_uncertain, 1e-6, "E[σ²] with uncertainty");
    
    /* Uncertainty increases expected variance */
    TEST_ASSERT(E_var_uncertain > E_var_certain, "Uncertainty increases E[σ²]");
}

/*============================================================================
 * Test 4: Extract std from Cholesky factor
 *============================================================================*/

static void test_extract_std(void) {
    /* For Σ = SSᵀ with lower triangular S:
     * Var(x_i) = Σ_{k=0}^{i} S[i,k]²
     */
    
    int nx = 3;
    
    /* S = [2  0  0]
     *     [1  3  0]
     *     [1  1  2]
     */
    double S[9] = {
        2, 1, 1,   /* column 0 */
        0, 3, 1,   /* column 1 */
        0, 0, 2    /* column 2 */
    };
    
    /* Var(x0) = 2² = 4, std = 2 */
    double std0 = kelly_extract_std(S, nx, 0);
    TEST_ASSERT_NEAR(std0, 2.0, 1e-6, "Std of x0");
    
    /* Var(x1) = 1² + 3² = 10, std = sqrt(10) */
    double std1 = kelly_extract_std(S, nx, 1);
    TEST_ASSERT_NEAR(std1, sqrt(10.0), 1e-6, "Std of x1");
    
    /* Var(x2) = 1² + 1² + 2² = 6, std = sqrt(6) */
    double std2 = kelly_extract_std(S, nx, 2);
    TEST_ASSERT_NEAR(std2, sqrt(6.0), 1e-6, "Std of x2");
}

/*============================================================================
 * Test 5: Weak signal filtering
 *============================================================================*/

static void test_weak_signal_filter(void) {
    /* If |μ| < KELLY_MIN_SIGNAL_RATIO × σ_μ, position should be zero */
    
    int nx = 3;
    int vel_idx = 1;
    int vol_idx = 2;
    
    /* State: weak signal (μ = 0.01, but σ_μ = 0.02) */
    double x[3] = {0.0, 0.01, -3.0};  /* level, velocity, log_vol */
    
    /* S with high uncertainty in velocity:
     * S = [0.1   0    0  ]
     *     [0    0.02  0  ]  ← σ_μ = 0.02
     *     [0     0   0.1 ]
     */
    double S[9] = {
        0.1, 0, 0,
        0, 0.02, 0,
        0, 0, 0.1
    };
    
    KellyResult result;
    kelly_from_ukf(x, S, nx, vel_idx, vol_idx, INFINITY, 0.5, &result);
    
    /* |μ| / σ_μ = 0.01 / 0.02 = 0.5 < KELLY_MIN_SIGNAL_RATIO (1.0) */
    TEST_ASSERT_NEAR(result.f_final, 0.0, 1e-6, "Weak signal → zero position");
    
    /* Strong signal (μ = 0.05, σ_μ = 0.02) */
    x[1] = 0.05;  /* Now |μ| / σ_μ = 2.5 > 1.0 */
    kelly_from_ukf(x, S, nx, vel_idx, vol_idx, INFINITY, 0.5, &result);
    TEST_ASSERT(fabs(result.f_final) > 0.0, "Strong signal → non-zero position");
}

/*============================================================================
 * Test 6: Asymmetric Kelly (short penalty)
 *============================================================================*/

static void test_asymmetric_kelly(void) {
    double mu_long = 0.05;
    double mu_short = -0.05;
    double sigma = 0.20;
    double nu = INFINITY;
    double fraction = 1.0;
    
    double f_long = kelly_simple(mu_long, sigma, nu, fraction);
    double f_short = kelly_simple(mu_short, sigma, nu, fraction);
    
    /* Long: f* = 0.05 / 0.04 = 1.25 */
    TEST_ASSERT_NEAR(f_long, 1.25, 1e-6, "Long position");
    
    /* Short: f* = -1.25, then × KELLY_SHORT_PENALTY (0.5) = -0.625 */
    TEST_ASSERT_NEAR(f_short, -0.625, 1e-6, "Short position with penalty");
    
    /* Asymmetry: |f_short| < |f_long| */
    TEST_ASSERT(fabs(f_short) < fabs(f_long), "Short penalty reduces short size");
}

/*============================================================================
 * Test 7: Multi-asset Kelly
 *============================================================================*/

static void test_multi_asset_kelly(void) {
    /* 2-asset case with known solution
     * 
     * μ = [0.10, 0.05]
     * Σ = [0.04  0.01]  (σ₁=0.2, σ₂=0.2, ρ=0.25)
     *     [0.01  0.04]
     * 
     * S (Cholesky of Σ):
     * S = [0.2     0    ]
     *     [0.05  0.1936]
     */
    
    int n = 2;
    double mu[2] = {0.10, 0.05};
    
    double S[4] = {
        0.2, 0.05,      /* column 0 */
        0.0, 0.1936     /* column 1 */
    };
    
    double f_out[2];
    double work[2];
    
    bool ok = kelly_multi_asset(mu, S, n, INFINITY, 0.5, f_out, work);
    TEST_ASSERT(ok, "Multi-asset Kelly succeeded");
    
    /* Verify positions are reasonable */
    TEST_ASSERT(f_out[0] > 0, "Asset 1 long (μ > 0)");
    TEST_ASSERT(f_out[1] > 0, "Asset 2 long (μ > 0)");
    TEST_ASSERT(f_out[0] > f_out[1], "Asset 1 larger (higher μ)");
    
    /* With heavy tails, positions should shrink */
    double f_out_tail[2];
    kelly_multi_asset(mu, S, n, 4.0, 0.5, f_out_tail, work);
    TEST_ASSERT(f_out_tail[0] < f_out[0], "Heavy tails shrink positions");
}

/*============================================================================
 * Test 8: Transaction cost filtering
 *============================================================================*/

static void test_transaction_costs(void) {
    int n = 2;
    double mu[2] = {0.10, 0.05};
    double S[4] = {0.2, 0.0, 0.0, 0.2};
    double f_prev[2] = {0.5, 0.3};
    double f_out[2];
    double work[4];
    
    /* High transaction cost: should stay near f_prev */
    double lambda_high = 0.5;  /* Very high */
    kelly_multi_asset_with_costs(mu, S, f_prev, n, INFINITY, 0.5, lambda_high, f_out, work);
    
    /* Changes should be damped */
    TEST_ASSERT_NEAR(f_out[0], f_prev[0], 0.01, "High tcost keeps position 0");
    TEST_ASSERT_NEAR(f_out[1], f_prev[1], 0.01, "High tcost keeps position 1");
    
    /* Low transaction cost: should move toward optimal */
    double lambda_low = 0.001;
    kelly_multi_asset_with_costs(mu, S, f_prev, n, INFINITY, 0.5, lambda_low, f_out, work);
    
    /* Should be closer to unconstrained Kelly */
    double f_kelly[2];
    kelly_multi_asset(mu, S, n, INFINITY, 0.5, f_kelly, work);
    
    double dist_to_kelly = fabs(f_out[0] - f_kelly[0]) + fabs(f_out[1] - f_kelly[1]);
    double dist_prev_to_kelly = fabs(f_prev[0] - f_kelly[0]) + fabs(f_prev[1] - f_kelly[1]);
    TEST_ASSERT(dist_to_kelly < dist_prev_to_kelly, "Low tcost moves toward optimal");
}

/*============================================================================
 * Test 9: Health scaling (kill switch)
 *============================================================================*/

static void test_health_scaling(void) {
    double soft = 0.1;
    double hard = 0.3;
    
    /* Healthy: full position */
    double scale_healthy = kelly_health_scale(0.05, soft, hard);
    TEST_ASSERT_NEAR(scale_healthy, 1.0, 1e-6, "Healthy → full scale");
    
    /* Degraded: partial position */
    double scale_degraded = kelly_health_scale(0.2, soft, hard);
    TEST_ASSERT(scale_degraded > 0.0 && scale_degraded < 1.0, "Degraded → partial scale");
    TEST_ASSERT_NEAR(scale_degraded, 0.5, 1e-6, "Degraded scale = 0.5");
    
    /* Unhealthy: zero position */
    double scale_unhealthy = kelly_health_scale(0.35, soft, hard);
    TEST_ASSERT_NEAR(scale_unhealthy, 0.0, 1e-6, "Unhealthy → zero scale");
}

/*============================================================================
 * Test 10: Growth rate calculations
 *============================================================================*/

static void test_growth_rate(void) {
    double mu = 0.10;
    double sigma = 0.20;
    
    /* Full Kelly: f* = μ/σ² = 2.5 */
    double f_full = mu / (sigma * sigma);
    
    /* Growth at full Kelly: g* = μ²/(2σ²) = 0.01/0.08 = 0.125 */
    double g_optimal = kelly_optimal_growth(mu, sigma);
    TEST_ASSERT_NEAR(g_optimal, 0.125, 1e-6, "Optimal growth rate");
    
    /* Growth at f=2.5: g = μf - ½σ²f² = 0.25 - 0.5×0.04×6.25 = 0.25 - 0.125 = 0.125 */
    double g_full = kelly_growth_rate(mu, sigma, f_full);
    TEST_ASSERT_NEAR(g_full, g_optimal, 1e-6, "Growth at full Kelly");
    
    /* Growth at half Kelly: 75% of optimal */
    double g_fraction_half = kelly_growth_fraction(0.5);
    TEST_ASSERT_NEAR(g_fraction_half, 0.75, 1e-6, "Half Kelly gives 75% growth");
    
    /* Growth at quarter Kelly: 2×0.25 - 0.25² = 0.4375 */
    double g_fraction_quarter = kelly_growth_fraction(0.25);
    TEST_ASSERT_NEAR(g_fraction_quarter, 0.4375, 1e-6, "Quarter Kelly growth fraction");
}

/*============================================================================
 * Test 11: Integration with UKF
 *============================================================================*/

static void test_ukf_integration(void) {
    /* Create a simple UKF */
    int nx = 3;  /* level, velocity, log_vol */
    int nz = 1;
    double nu = 4.0;
    
    StudentT_SRUKF* ukf = srukf_create(nx, nz, nu);
    TEST_ASSERT(ukf != NULL, "UKF created");
    
    /* Set initial state */
    double x0[3] = {100.0, 0.05, -3.0};  /* level=100, velocity=5%, log_vol=-3 (σ≈5%) */
    double S0[9] = {0};
    S0[0] = 1.0;   /* S[0,0] */
    S0[4] = 0.02;  /* S[1,1] */
    S0[8] = 0.1;   /* S[2,2] */
    
    srukf_set_state(ukf, x0);
    srukf_set_sqrt_cov(ukf, S0);
    
    /* Set dynamics: simple random walk */
    double F[9] = {0};
    F[0] = 1.0; F[4] = 1.0; F[8] = 0.95;  /* Diagonal, log_vol mean-reverts */
    srukf_set_dynamics(ukf, F);
    
    /* Set measurement: observe level */
    double H[3] = {1.0, 0.0, 0.0};
    srukf_set_measurement(ukf, H);
    
    /* Set noise */
    double Sq[9] = {0};
    Sq[0] = 0.1; Sq[4] = 0.01; Sq[8] = 0.05;
    srukf_set_process_noise(ukf, Sq);
    
    double Sr[1] = {1.0};
    srukf_set_measurement_noise(ukf, Sr);
    
    /* Run a few predict/update cycles */
    double z[1] = {100.5};
    for (int i = 0; i < 10; i++) {
        z[0] = 100.0 + 0.1 * i + 0.5 * ((double)rand() / RAND_MAX - 0.5);
        srukf_step(ukf, z);
    }
    
    /* Now compute Kelly from UKF state */
    const double* x = srukf_get_state(ukf);
    const double* S = srukf_get_sqrt_cov(ukf);
    
    KellyResult result;
    kelly_from_ukf(x, S, nx, 1, 2, nu, 0.5, &result);
    
    /* Basic sanity checks */
    TEST_ASSERT(isfinite(result.f_final), "Kelly position is finite");
    TEST_ASSERT(result.f_final >= KELLY_MIN_LEVERAGE, "Position above min");
    TEST_ASSERT(result.f_final <= KELLY_MAX_LEVERAGE, "Position below max");
    TEST_ASSERT(result.volatility > 0, "Volatility is positive");
    
    /* Print result for inspection */
    printf("    UKF state: level=%.2f, vel=%.4f, log_vol=%.2f\n", x[0], x[1], x[2]);
    printf("    Kelly: f_final=%.4f, μ=%.4f, σ=%.4f, Sharpe=%.2f\n",
           result.f_final, result.expected_return, result.volatility, result.sharpe);
    
    srukf_destroy(ukf);
}

/*============================================================================
 * Test 12: Full pipeline with NIS health check
 *============================================================================*/

static void test_full_pipeline(void) {
    /* Create UKF */
    int nx = 3;
    int nz = 1;
    double nu = 4.0;
    
    StudentT_SRUKF* ukf = srukf_create(nx, nz, nu);
    TEST_ASSERT(ukf != NULL, "UKF created");
    
    /* Initialize */
    double x0[3] = {100.0, 0.02, -2.5};
    double S0[9] = {0};
    S0[0] = 1.0; S0[4] = 0.01; S0[8] = 0.1;
    
    srukf_set_state(ukf, x0);
    srukf_set_sqrt_cov(ukf, S0);
    
    double F[9] = {0};
    F[0] = 1.0; F[4] = 1.0; F[8] = 0.95;
    srukf_set_dynamics(ukf, F);
    
    double H[3] = {1.0, 0.0, 0.0};
    srukf_set_measurement(ukf, H);
    
    double Sq[9] = {0};
    Sq[0] = 0.1; Sq[4] = 0.01; Sq[8] = 0.05;
    srukf_set_process_noise(ukf, Sq);
    
    double Sr[1] = {1.0};
    srukf_set_measurement_noise(ukf, Sr);
    
    /* Run normal updates */
    double z[1];
    for (int i = 0; i < 20; i++) {
        z[0] = 100.0 + 0.05 * i + 0.3 * ((double)rand() / RAND_MAX - 0.5);
        srukf_step(ukf, z);
    }
    
    /* Get NIS stats */
    double nis_mean, nis_var, nis_above;
    srukf_nis_stats(ukf, &nis_mean, &nis_var, &nis_above);
    
    /* Full pipeline */
    const double* x = srukf_get_state(ukf);
    const double* S = srukf_get_sqrt_cov(ukf);
    
    double position = kelly_full_pipeline(
        x, S, nx, 1, 2, nu,
        0.5,        /* half Kelly */
        nis_above,  /* from UKF */
        0.1, 0.3    /* soft/hard thresholds */
    );
    
    TEST_ASSERT(isfinite(position), "Pipeline position is finite");
    printf("    NIS stats: mean=%.2f, above_threshold=%.1f%%\n", nis_mean, nis_above * 100);
    printf("    Pipeline position: %.4f\n", position);
    
    /* Inject outliers to trigger health degradation */
    printf("    Injecting outliers...\n");
    for (int i = 0; i < 10; i++) {
        z[0] = 150.0;  /* Big outlier */
        srukf_step(ukf, z);
    }
    
    srukf_nis_stats(ukf, &nis_mean, &nis_var, &nis_above);
    
    x = srukf_get_state(ukf);
    S = srukf_get_sqrt_cov(ukf);
    
    double position_after = kelly_full_pipeline(
        x, S, nx, 1, 2, nu,
        0.5, nis_above, 0.1, 0.3
    );
    
    printf("    After outliers: NIS above=%.1f%%, position=%.4f\n", 
           nis_above * 100, position_after);
    
    /* Position should be reduced if model health degraded */
    if (nis_above > 0.1) {
        TEST_ASSERT(fabs(position_after) <= fabs(position) + 0.01, 
                    "Degraded health reduces or maintains position");
    }
    
    srukf_destroy(ukf);
}

/*============================================================================
 * Test 13: Edge cases
 *============================================================================*/

static void test_edge_cases(void) {
    /* Zero volatility (floored) */
    double f = kelly_simple(0.10, 0.0, INFINITY, 0.5);
    TEST_ASSERT(isfinite(f), "Zero vol handled");
    TEST_ASSERT_NEAR(f, KELLY_MAX_LEVERAGE, 1e-6, "Zero vol → max leverage");
    
    /* Zero return */
    f = kelly_simple(0.0, 0.20, INFINITY, 0.5);
    TEST_ASSERT_NEAR(f, 0.0, 1e-6, "Zero return → zero position");
    
    /* Very heavy tails (ν near 2) */
    f = kelly_simple(0.10, 0.20, 2.1, 0.5);
    TEST_ASSERT(isfinite(f), "Very heavy tails handled");
    TEST_ASSERT(f < 0.5, "Very heavy tails → small position");
    
    /* Infinite ν (Gaussian) */
    f = kelly_simple(0.10, 0.20, INFINITY, 0.5);
    TEST_ASSERT(isfinite(f), "Infinite ν handled");
}

/*============================================================================
 * Test 14: Position limits
 *============================================================================*/

static void test_position_limits(void) {
    /* Very high Sharpe → capped long */
    double f = kelly_simple(0.50, 0.10, INFINITY, 1.0);
    /* f* = 0.50 / 0.01 = 50, capped at 2.0 */
    TEST_ASSERT_NEAR(f, KELLY_MAX_LEVERAGE, 1e-6, "Long position capped");
    
    /* Very negative Sharpe → capped short (after penalty) */
    f = kelly_simple(-0.50, 0.10, INFINITY, 1.0);
    /* f* = -50, × penalty = -25, capped at -1.0 */
    TEST_ASSERT_NEAR(f, KELLY_MIN_LEVERAGE, 1e-6, "Short position capped");
}

/*============================================================================
 * Main
 *============================================================================*/

int main(void) {
    printf("\n");
    printf("========================================\n");
    printf("  Kelly Criterion Tests\n");
    printf("========================================\n\n");
    
    RUN_TEST(test_basic_kelly);
    RUN_TEST(test_tail_kelly);
    RUN_TEST(test_bayesian_variance);
    RUN_TEST(test_extract_std);
    RUN_TEST(test_weak_signal_filter);
    RUN_TEST(test_asymmetric_kelly);
    RUN_TEST(test_multi_asset_kelly);
    RUN_TEST(test_transaction_costs);
    RUN_TEST(test_health_scaling);
    RUN_TEST(test_growth_rate);
    RUN_TEST(test_ukf_integration);
    RUN_TEST(test_full_pipeline);
    RUN_TEST(test_edge_cases);
    RUN_TEST(test_position_limits);
    
    printf("\n========================================\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n\n");
    
    return tests_failed > 0 ? 1 : 0;
}
