/**
 * @file kelly.h
 * @brief Kelly Criterion with UKF Integration
 * 
 * Computes optimal position sizing from UKF state estimates.
 * Reuses same MKL triangular solves as UKF for efficiency.
 * 
 * ============================================================================
 * KELLY CRITERION OVERVIEW
 * ============================================================================
 * 
 * Basic Kelly (single asset):
 *   f* = μ / σ²
 * 
 * Multi-asset Kelly:
 *   f* = Σ⁻¹ μ
 * 
 * With sqrt covariance S (where Σ = SSᵀ):
 *   f* = S⁻ᵀ (S⁻¹ μ)   ← Two triangular solves, no matrix inversion
 * 
 * ============================================================================
 * BAYESIAN ADJUSTMENT FOR LOG-VOLATILITY
 * ============================================================================
 * 
 * When volatility is modeled as log-normal (log_vol ~ N(μ_lv, σ_lv²)):
 *   
 *   E[σ²] = exp(2μ_lv + 2σ_lv²)   (log-normal moment)
 * 
 * This properly accounts for volatility uncertainty:
 *   - Higher σ_lv → larger E[σ²] → smaller position
 *   - Principled Bayesian shrinkage
 * 
 * ============================================================================
 * TAIL-ROBUST KELLY (Student-t)
 * ============================================================================
 * 
 * For Student-t returns with ν degrees of freedom:
 *   
 *   Var = σ² × ν/(ν-2)   for ν > 2
 * 
 * Inflates Kelly denominator for heavy-tailed returns:
 *   f*_tail = μ / (σ² × ν/(ν-2))
 * 
 * As ν → ∞, recovers Gaussian Kelly.
 * 
 * ============================================================================
 * UKF INTEGRATION
 * ============================================================================
 * 
 * UKF state x = [level, velocity, log_vol]ᵀ
 *   - velocity  → expected return μ
 *   - log_vol   → log volatility, σ = exp(log_vol)
 * 
 * UKF sqrt covariance S gives estimation uncertainty:
 *   - S[1,1] → σ_μ (uncertainty in μ estimate)
 *   - S[2,2] → σ_lv (uncertainty in log-vol estimate)
 * 
 * Combined with Student-t ν from UKF, gives fully Bayesian Kelly.
 * 
 * ============================================================================
 */

#ifndef KELLY_H
#define KELLY_H

#include <mkl.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * CONFIGURATION
 *============================================================================*/

/* Default fractional Kelly (full Kelly is too aggressive) */
#define KELLY_DEFAULT_FRACTION      0.5

/* Position limits */
#define KELLY_MAX_LEVERAGE          2.0     /* Maximum long position */
#define KELLY_MIN_LEVERAGE         -1.0     /* Maximum short position */
#define KELLY_MIN_VOLATILITY        1e-6    /* Floor to avoid division by zero */

/* Weak signal filter: zero position if |μ| < k × σ_μ */
#define KELLY_MIN_SIGNAL_RATIO      1.0     /* Require μ > 1σ to trade */

/* Asymmetric Kelly: short penalty (shorts are structurally harder) */
#define KELLY_SHORT_PENALTY         0.5     /* Scale down short positions */

/* Tail-Kelly: additional variance from heavy tails */
#define KELLY_TAIL_FLOOR_NU         2.1     /* Below this ν, infinite variance */

/*============================================================================
 * TIMESCALE
 *============================================================================
 * 
 * CRITICAL: Kelly requires μ and σ on the SAME timescale.
 * 
 * If your UKF estimates:
 *   μ_tick  = expected return per tick
 *   σ_tick  = volatility per tick
 * 
 * Then Kelly is:
 *   f* = μ_tick / σ_tick²   (correct, same timescale)
 * 
 * To annualize (if needed for reporting):
 *   μ_annual = μ_tick × ticks_per_year
 *   σ_annual = σ_tick × sqrt(ticks_per_year)
 *   Sharpe_annual = (μ_tick / σ_tick) × sqrt(ticks_per_year)
 * 
 * Kelly fraction is SCALE INVARIANT:
 *   μ_annual / σ_annual² = μ_tick / σ_tick²
 * 
 * So Kelly doesn't care about timescale, but your μ and σ must match.
 *============================================================================*/

/*============================================================================
 * RESULT STRUCTURE
 *============================================================================*/

typedef struct {
    double f_full;          /* Full Kelly fraction */
    double f_half;          /* Half Kelly (f_full × 0.5) */
    double f_adjusted;      /* Uncertainty-adjusted Kelly */
    double f_final;         /* Final position after limits */
    
    double expected_return; /* μ used in calculation */
    double volatility;      /* σ used in calculation */
    double sharpe;          /* μ/σ (annualize externally if needed) */
    
    double mu_uncertainty;  /* Std dev of μ estimate */
    double vol_uncertainty; /* Std dev of σ estimate */
    
    bool capped_long;       /* Was position capped at max long? */
    bool capped_short;      /* Was position capped at max short? */
} KellyResult;

/*============================================================================
 * SINGLE ASSET KELLY (Most common case for your UKF)
 *============================================================================*/

/**
 * @brief Extract standard deviation from sqrt covariance (Cholesky factor)
 * 
 * For lower triangular S where Σ = SSᵀ:
 *   Var(x_i) = Σ_{k=0}^{i} S[i,k]²
 *   Std(x_i) = sqrt(Var(x_i))
 * 
 * NOTE: S[i,i] alone is NOT the standard deviation!
 */
static inline double kelly_extract_std(const double* S, int nx, int idx) {
    double var = 0.0;
    for (int k = 0; k <= idx; k++) {
        double v = S[idx + k * nx];  /* S[idx, k] in column-major */
        var += v * v;
    }
    return sqrt(var);
}

/**
 * @brief Compute E[σ²] from log-volatility distribution
 * 
 * If log_vol ~ N(μ_lv, σ_lv²), then:
 *   σ = exp(log_vol)
 *   σ² = exp(2 × log_vol)
 *   E[σ²] = exp(2μ_lv + 2σ_lv²)   (log-normal moment)
 * 
 * This is the Bayesian-correct expected variance accounting for
 * uncertainty in volatility estimation.
 */
static inline double kelly_expected_variance(double mu_lv, double sigma_lv) {
    return exp(2.0 * mu_lv + 2.0 * sigma_lv * sigma_lv);
}

/**
 * @brief Compute tail variance adjustment from Student-t ν
 * 
 * Student-t with ν degrees of freedom has variance:
 *   Var = σ² × ν/(ν-2)  for ν > 2
 * 
 * This inflates Kelly denominator for heavy-tailed returns.
 * 
 * @param base_var  Base variance (σ²)
 * @param nu        Student-t degrees of freedom
 * @return Tail-adjusted variance
 */
static inline double kelly_tail_variance(double base_var, double nu) {
    if (nu <= KELLY_TAIL_FLOOR_NU) {
        /* ν ≤ 2: infinite variance, use large multiplier */
        return base_var * 10.0;
    }
    return base_var * nu / (nu - 2.0);
}

/**
 * @brief Compute Kelly fraction from UKF state (Bayesian formulation)
 * 
 * Proper handling of:
 *   - Log-volatility: E[σ²] = exp(2μ_lv + 2σ_lv²)
 *   - Correlation: uses full covariance, not independent CVs
 *   - Weak signals: zeros position if μ < k × σ_μ
 *   - Asymmetry: penalizes short positions
 *   - Heavy tails: inflates variance by ν/(ν-2)
 * 
 * @param x         UKF state vector [level, velocity, log_vol]
 * @param S         UKF sqrt covariance (nx × nx, column-major, lower triangular)
 * @param nx        State dimension (typically 3)
 * @param vel_idx   Index of velocity state (typically 1)
 * @param vol_idx   Index of log-volatility state (typically 2)
 * @param nu        Student-t degrees of freedom (use INFINITY for Gaussian)
 * @param fraction  Kelly fraction (0.5 = half Kelly)
 * @param result    Output result structure
 */
static inline void kelly_from_ukf(
    const double* x,
    const double* S,
    int nx,
    int vel_idx,
    int vol_idx,
    double nu,
    double fraction,
    KellyResult* result
) {
    /* Extract estimates from UKF state */
    double mu = x[vel_idx];                    /* Expected return */
    double mu_lv = x[vol_idx];                 /* Log-volatility mean */
    
    /* Extract uncertainties from sqrt covariance
     * IMPORTANT: S is Cholesky factor, not covariance!
     * Var(x_i) = Σ_{k=0}^{i} S[i,k]², not S[i,i]² */
    double sigma_mu = kelly_extract_std(S, nx, vel_idx);
    double sigma_lv = kelly_extract_std(S, nx, vol_idx);
    
    /* Point estimate of volatility */
    double sigma_point = exp(mu_lv);
    
    /* Bayesian expected variance: E[σ²] accounting for log-vol uncertainty
     * This is more principled than the naive exp(mu_lv)² */
    double expected_var = kelly_expected_variance(mu_lv, sigma_lv);
    
    /* Tail adjustment: inflate variance for heavy-tailed returns */
    double tail_var = kelly_tail_variance(expected_var, nu);
    
    /* Floor variance */
    if (tail_var < KELLY_MIN_VOLATILITY * KELLY_MIN_VOLATILITY) {
        tail_var = KELLY_MIN_VOLATILITY * KELLY_MIN_VOLATILITY;
    }
    
    /* Weak signal filter: require |μ| > k × σ_μ to trade
     * This prevents noisy near-zero signals from generating positions */
    double signal_strength = (sigma_mu > 0) ? fabs(mu) / sigma_mu : INFINITY;
    
    if (signal_strength < KELLY_MIN_SIGNAL_RATIO) {
        /* Signal too weak relative to estimation uncertainty */
        result->f_full = 0.0;
        result->f_half = 0.0;
        result->f_adjusted = 0.0;
        result->f_final = 0.0;
        result->expected_return = mu;
        result->volatility = sigma_point;
        result->sharpe = (sigma_point > 0) ? mu / sigma_point : 0.0;
        result->mu_uncertainty = sigma_mu;
        result->vol_uncertainty = sigma_lv;
        result->capped_long = false;
        result->capped_short = false;
        return;
    }
    
    /* Bayesian Kelly: f* = E[μ] / E[σ²]
     * Using tail-adjusted variance for crash robustness */
    double f_full = mu / tail_var;
    
    /* Half Kelly */
    double f_half = f_full * 0.5;
    
    /* Apply fraction */
    double f_adjusted = f_full * fraction;
    
    /* Asymmetric penalty for shorts
     * Shorting is structurally harder: borrow costs, squeeze risk, unlimited loss */
    if (f_adjusted < 0) {
        f_adjusted *= KELLY_SHORT_PENALTY;
    }
    
    /* Apply position limits */
    double f_final = f_adjusted;
    bool capped_long = false;
    bool capped_short = false;
    
    if (f_final > KELLY_MAX_LEVERAGE) {
        f_final = KELLY_MAX_LEVERAGE;
        capped_long = true;
    } else if (f_final < KELLY_MIN_LEVERAGE) {
        f_final = KELLY_MIN_LEVERAGE;
        capped_short = true;
    }
    
    /* Populate result */
    result->f_full = f_full;
    result->f_half = f_half;
    result->f_adjusted = f_adjusted;
    result->f_final = f_final;
    
    result->expected_return = mu;
    result->volatility = sigma_point;
    result->sharpe = (sigma_point > 0) ? mu / sigma_point : 0.0;
    
    result->mu_uncertainty = sigma_mu;
    result->vol_uncertainty = sigma_lv;
    
    result->capped_long = capped_long;
    result->capped_short = capped_short;
}

/**
 * @brief Simple Kelly from μ and σ directly (no uncertainty adjustment)
 * 
 * Use when you have point estimates only, no UKF covariance.
 * 
 * @param mu        Expected return
 * @param sigma     Volatility (same timescale as mu)
 * @param nu        Student-t degrees of freedom (INFINITY for Gaussian)
 * @param fraction  Kelly fraction
 */
static inline double kelly_simple(double mu, double sigma, double nu, double fraction) {
    if (sigma < KELLY_MIN_VOLATILITY) {
        sigma = KELLY_MIN_VOLATILITY;
    }
    
    double var = sigma * sigma;
    
    /* Tail adjustment */
    var = kelly_tail_variance(var, nu);
    
    double f = fraction * mu / var;
    
    /* Asymmetric short penalty */
    if (f < 0) {
        f *= KELLY_SHORT_PENALTY;
    }
    
    /* Clamp */
    if (f > KELLY_MAX_LEVERAGE) f = KELLY_MAX_LEVERAGE;
    if (f < KELLY_MIN_LEVERAGE) f = KELLY_MIN_LEVERAGE;
    
    return f;
}

/*============================================================================
 * MULTI-ASSET KELLY (Using triangular solves like UKF)
 *============================================================================*/

/**
 * @brief Multi-asset Kelly: f* = Σ⁻¹ μ via triangular solves
 * 
 * Given sqrt covariance S where Σ = SSᵀ:
 *   f* = S⁻ᵀ (S⁻¹ μ)
 * 
 * Two dtrsv calls - same as UKF uses internally.
 * 
 * @param mu        Expected returns vector (n × 1)
 * @param S         Sqrt covariance matrix (n × n, column-major, lower triangular)
 * @param n         Number of assets
 * @param nu        Student-t degrees of freedom (INFINITY for Gaussian)
 * @param fraction  Kelly fraction
 * @param f_out     Output: optimal fractions (n × 1)
 * @param work      Workspace (n doubles)
 * @return true on success
 */
static inline bool kelly_multi_asset(
    const double* mu,
    const double* S,
    int n,
    double nu,
    double fraction,
    double* f_out,
    double* work
) {
    /* Copy μ to workspace (dtrsv overwrites) */
    memcpy(work, mu, n * sizeof(double));
    
    /* Step 1: Solve S y = μ (lower triangular) */
    cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                n, S, n, work, 1);
    
    /* Step 2: Solve Sᵀ f = y (lower triangular, transposed) */
    cblas_dtrsv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit,
                n, S, n, work, 1);
    
    /* Tail adjustment factor */
    double tail_factor = 1.0;
    if (nu > KELLY_TAIL_FLOOR_NU && isfinite(nu)) {
        tail_factor = (nu - 2.0) / nu;  /* Shrink positions for heavy tails */
    } else if (nu <= KELLY_TAIL_FLOOR_NU) {
        tail_factor = 0.1;  /* Very heavy tails, be very conservative */
    }
    
    /* Apply fraction, tail adjustment, and clamp */
    for (int i = 0; i < n; i++) {
        double f = fraction * tail_factor * work[i];
        
        /* Asymmetric short penalty */
        if (f < 0) {
            f *= KELLY_SHORT_PENALTY;
        }
        
        /* Clamp each position */
        if (f > KELLY_MAX_LEVERAGE) f = KELLY_MAX_LEVERAGE;
        if (f < KELLY_MIN_LEVERAGE) f = KELLY_MIN_LEVERAGE;
        
        f_out[i] = f;
    }
    
    return true;
}

/**
 * @brief Multi-asset Kelly with transaction cost penalty
 * 
 * Solves the turnover-penalized problem:
 *   f* = argmax_f { μᵀf - ½fᵀΣf - λ‖f - f_prev‖₁ }
 * 
 * This implementation uses soft-thresholding approximation:
 *   1. Compute unconstrained Kelly: f_kelly = Σ⁻¹μ
 *   2. Apply shrinkage toward f_prev: f* = f_prev + shrink(f_kelly - f_prev)
 * 
 * For exact L1 solution, use a proper QP solver (OSQP, etc.)
 * 
 * @param mu        Expected returns (n × 1)
 * @param S         Sqrt covariance (n × n, lower triangular)
 * @param f_prev    Previous position (n × 1)
 * @param n         Number of assets
 * @param nu        Student-t degrees of freedom
 * @param fraction  Kelly fraction
 * @param lambda    Transaction cost per unit turnover
 * @param f_out     Output: new position (n × 1)
 * @param work      Workspace (2n doubles)
 * @return true on success
 */
static inline bool kelly_multi_asset_with_costs(
    const double* mu,
    const double* S,
    const double* f_prev,
    int n,
    double nu,
    double fraction,
    double lambda,
    double* f_out,
    double* work
) {
    double* f_kelly = work;      /* n doubles */
    double* temp = work + n;     /* n doubles */
    
    /* Compute unconstrained Kelly */
    if (!kelly_multi_asset(mu, S, n, nu, fraction, f_kelly, temp)) {
        return false;
    }
    
    /* Soft-thresholding: shrink changes toward zero
     * 
     * For each asset i:
     *   delta = f_kelly[i] - f_prev[i]
     *   if |delta| < λ: keep f_prev (don't trade)
     *   else: f_out = f_prev + sign(delta) × (|delta| - λ)
     */
    for (int i = 0; i < n; i++) {
        double delta = f_kelly[i] - f_prev[i];
        double abs_delta = fabs(delta);
        
        if (abs_delta <= lambda) {
            /* Change too small to justify transaction cost */
            f_out[i] = f_prev[i];
        } else {
            /* Shrink the change by λ */
            double shrunk = abs_delta - lambda;
            f_out[i] = f_prev[i] + (delta > 0 ? shrunk : -shrunk);
        }
        
        /* Still enforce position limits */
        if (f_out[i] > KELLY_MAX_LEVERAGE) f_out[i] = KELLY_MAX_LEVERAGE;
        if (f_out[i] < KELLY_MIN_LEVERAGE) f_out[i] = KELLY_MIN_LEVERAGE;
    }
    
    return true;
}

/**
 * @brief Compute total transaction cost for position change
 * 
 * @param f_new     New position (n × 1)
 * @param f_prev    Previous position (n × 1)
 * @param n         Number of assets
 * @param lambda    Cost per unit turnover
 * @return Total transaction cost
 */
static inline double kelly_transaction_cost(
    const double* f_new,
    const double* f_prev,
    int n,
    double lambda
) {
    double cost = 0.0;
    for (int i = 0; i < n; i++) {
        cost += lambda * fabs(f_new[i] - f_prev[i]);
    }
    return cost;
}

/**
 * @brief Multi-asset Kelly with full covariance (not sqrt form)
 * 
 * If you have Σ directly, we compute Cholesky first.
 * Less efficient than sqrt form - prefer kelly_multi_asset() when possible.
 * 
 * @param mu        Expected returns (n × 1)
 * @param Sigma     Full covariance matrix (n × n, column-major)
 * @param n         Number of assets
 * @param nu        Student-t degrees of freedom
 * @param fraction  Kelly fraction
 * @param f_out     Output: optimal fractions (n × 1)
 * @param work      Workspace (n × n + n doubles)
 * @return true on success, false if Cholesky fails
 */
static inline bool kelly_multi_asset_full_cov(
    const double* mu,
    const double* Sigma,
    int n,
    double nu,
    double fraction,
    double* f_out,
    double* work
) {
    double* S_work = work;          /* n × n for Cholesky factor */
    double* vec_work = work + n*n;  /* n for triangular solves */
    
    /* Copy Sigma to workspace */
    memcpy(S_work, Sigma, n * n * sizeof(double));
    
    /* Cholesky: Σ = LLᵀ */
    lapack_int info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, S_work, n);
    if (info != 0) {
        return false;  /* Not positive definite */
    }
    
    /* Now use sqrt form */
    return kelly_multi_asset(mu, S_work, n, nu, fraction, f_out, vec_work);
}

/*============================================================================
 * EXPECTED GROWTH RATE
 *============================================================================*/

/**
 * @brief Compute expected log growth rate for a given Kelly fraction
 * 
 * g(f) = μf - ½σ²f²
 * 
 * Maximum at f* = μ/σ² (full Kelly)
 */
static inline double kelly_growth_rate(double mu, double sigma, double f) {
    double sigma_sq = sigma * sigma;
    return mu * f - 0.5 * sigma_sq * f * f;
}

/**
 * @brief Compute optimal growth rate (at full Kelly)
 * 
 * g* = μ²/(2σ²) = ½ × Sharpe²
 */
static inline double kelly_optimal_growth(double mu, double sigma) {
    if (sigma < KELLY_MIN_VOLATILITY) {
        sigma = KELLY_MIN_VOLATILITY;
    }
    double sharpe = mu / sigma;
    return 0.5 * sharpe * sharpe;
}

/**
 * @brief Fraction of optimal growth achieved at given Kelly fraction
 * 
 * At half-Kelly: 75% of optimal growth, 50% of variance
 * At quarter-Kelly: 43.75% of optimal growth, 25% of variance
 */
static inline double kelly_growth_fraction(double kelly_frac) {
    /* g(f)/g* = 2f - f² where f is fraction of full Kelly */
    return 2.0 * kelly_frac - kelly_frac * kelly_frac;
}

/*============================================================================
 * DRAWDOWN ESTIMATES (ADVISORY HEURISTICS - NOT EXACT)
 *============================================================================
 * 
 * WARNING: These are approximations for intuition only.
 * Actual drawdowns depend on:
 *   - Path dependency
 *   - Non-normality (tails, skew)
 *   - Time horizon
 *   - Correlation structure
 * 
 * Use for rough guidance, not risk management.
 *============================================================================*/

/**
 * @brief Estimate maximum drawdown for given Kelly fraction
 * 
 * Approximation based on continuous-time Kelly analysis.
 * Full Kelly has ~87% expected max drawdown over infinite horizon.
 * Half Kelly has ~50% expected max drawdown.
 * 
 * @param kelly_frac Fraction of full Kelly (0.5 = half)
 * @return Approximate expected maximum drawdown (0-1)
 */
static inline double kelly_expected_max_drawdown(double kelly_frac) {
    /* Rough approximation: MDD ≈ 1 - exp(-2×f) for f = Kelly fraction */
    /* At f=1 (full Kelly): 1 - exp(-2) ≈ 0.86 */
    /* At f=0.5 (half Kelly): 1 - exp(-1) ≈ 0.63 */
    return 1.0 - exp(-2.0 * kelly_frac);
}

/**
 * @brief Find Kelly fraction that targets specific max drawdown
 * 
 * @param target_mdd Target maximum drawdown (e.g., 0.20 for 20%)
 * @return Kelly fraction to use
 */
static inline double kelly_fraction_for_drawdown(double target_mdd) {
    if (target_mdd <= 0.0) return 0.0;
    if (target_mdd >= 1.0) return 1.0;
    
    /* Invert: target = 1 - exp(-2f) → f = -ln(1-target)/2 */
    return -log(1.0 - target_mdd) / 2.0;
}

/*============================================================================
 * KILL SWITCH INTEGRATION
 *============================================================================*/

/**
 * @brief Check if Kelly position should be zeroed by kill switch
 * 
 * Integrates with UKF's windowed NIS statistics.
 * 
 * @param nis_mean     Mean NIS from windowed stats
 * @param nis_above    Fraction of NIS above threshold
 * @param threshold    Kill threshold (e.g., 0.3 = 30% above)
 * @return true if should kill (zero position)
 */
static inline bool kelly_kill_check(double nis_mean, double nis_above, double threshold) {
    (void)nis_mean;  /* Could use for graduated response */
    return nis_above > threshold;
}

/**
 * @brief Graduated position scaling based on model health
 * 
 * Instead of binary kill, scale position down as model degrades.
 * 
 * @param nis_above Fraction of NIS above threshold (0-1)
 * @param soft_threshold Start reducing at this level (e.g., 0.1)
 * @param hard_threshold Zero position at this level (e.g., 0.3)
 * @return Scale factor (0-1) to multiply Kelly position
 */
static inline double kelly_health_scale(double nis_above, 
                                         double soft_threshold,
                                         double hard_threshold) {
    if (nis_above <= soft_threshold) {
        return 1.0;  /* Full position */
    }
    if (nis_above >= hard_threshold) {
        return 0.0;  /* Zero position */
    }
    
    /* Linear interpolation */
    return (hard_threshold - nis_above) / (hard_threshold - soft_threshold);
}

/*============================================================================
 * CONVENIENCE: FULL PIPELINE
 *============================================================================*/

/**
 * @brief Complete Kelly calculation from UKF output
 * 
 * Combines: state extraction → Kelly → uncertainty adjustment → limits → health scaling
 * 
 * @param x              UKF state
 * @param S              UKF sqrt covariance
 * @param nx             State dimension
 * @param vel_idx        Velocity state index
 * @param vol_idx        Log-vol state index
 * @param nu             Student-t degrees of freedom (INFINITY for Gaussian)
 * @param fraction       Base Kelly fraction
 * @param nis_above      NIS health metric (0 if not using)
 * @param soft_threshold Health soft threshold
 * @param hard_threshold Health hard threshold
 * @return Final position size
 */
static inline double kelly_full_pipeline(
    const double* x,
    const double* S,
    int nx,
    int vel_idx,
    int vol_idx,
    double nu,
    double fraction,
    double nis_above,
    double soft_threshold,
    double hard_threshold
) {
    KellyResult result;
    kelly_from_ukf(x, S, nx, vel_idx, vol_idx, nu, fraction, &result);
    
    /* Apply health scaling */
    double health = kelly_health_scale(nis_above, soft_threshold, hard_threshold);
    
    return result.f_final * health;
}

/*============================================================================
 * LOGGING / DEBUG
 *============================================================================*/

/**
 * @brief Print Kelly result for logging/debugging
 */
static inline void kelly_print_result(const KellyResult* r) {
    printf("Kelly Result:\n");
    printf("  Expected Return: %.6f\n", r->expected_return);
    printf("  Volatility:      %.6f\n", r->volatility);
    printf("  Sharpe:          %.3f\n", r->sharpe);
    printf("  Full Kelly:      %.3f\n", r->f_full);
    printf("  Half Kelly:      %.3f\n", r->f_half);
    printf("  Adjusted:        %.3f\n", r->f_adjusted);
    printf("  Final Position:  %.3f%s%s\n", r->f_final,
           r->capped_long ? " (capped long)" : "",
           r->capped_short ? " (capped short)" : "");
    printf("  Uncertainty (mu): %.6f\n", r->mu_uncertainty);
    printf("  Uncertainty (vol): %.6f\n", r->vol_uncertainty);
}

#ifdef __cplusplus
}
#endif

#endif /* KELLY_H */
