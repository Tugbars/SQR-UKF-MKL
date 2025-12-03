"""
Comparison: Student-t SQR UKF vs FilterPy UKF

Tests:
1. Correctness - Both filters should track the same signal
2. Speed - MKL-accelerated C vs pure Python
3. Numerical stability - Under stress conditions
"""

import numpy as np
import time
from typing import Tuple

# Our implementation
from srukf import StudentTSRUKF, Kelly, create_trend_filter

# =============================================================================
# FilterPy Reference Implementation
# =============================================================================

try:
    from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
    HAS_FILTERPY = True
except ImportError:
    HAS_FILTERPY = False
    print("FilterPy not installed. Install with: pip install filterpy")
    print("Running without comparison.\n")


def create_filterpy_ukf(nx: int = 3, nz: int = 1) -> "UnscentedKalmanFilter":
    """Create FilterPy UKF with same configuration as our implementation."""
    
    # Sigma point selection (Merwe's scaled points - standard choice)
    points = MerweScaledSigmaPoints(n=nx, alpha=0.1, beta=2.0, kappa=0.0)
    
    def fx(x, dt):
        """State transition function."""
        F = np.array([
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.95],
        ])
        return F @ x
    
    def hx(x):
        """Measurement function."""
        return np.array([x[0]])
    
    ukf = UnscentedKalmanFilter(
        dim_x=nx,
        dim_z=nz,
        dt=1.0,
        fx=fx,
        hx=hx,
        points=points,
    )
    
    # Initial state
    ukf.x = np.array([0.0, 0.0, -3.0])
    
    # Initial covariance
    ukf.P = np.diag([1.0, 0.01, 0.01])
    
    # Process noise
    ukf.Q = np.diag([0.01, 0.0001, 0.0025])  # Sq @ Sq.T where Sq = diag([0.1, 0.01, 0.05])
    
    # Measurement noise
    ukf.R = np.array([[1.0]])
    
    return ukf


# =============================================================================
# Test Data Generation
# =============================================================================

def generate_test_data(T: int = 1000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic test data.
    
    Returns
    -------
    measurements : ndarray
        Noisy measurements (T,)
    true_states : ndarray
        True underlying states (T, 3)
    """
    np.random.seed(seed)
    
    true_states = np.zeros((T, 3))
    measurements = np.zeros(T)
    
    # Initial state
    level = 100.0
    velocity = 0.1
    log_vol = -3.0  # σ ≈ 5%
    
    for t in range(T):
        # State evolution
        level += velocity + np.random.randn() * np.exp(log_vol)
        velocity += np.random.randn() * 0.01
        log_vol = 0.95 * log_vol + (-3.0) * 0.05 + np.random.randn() * 0.05
        
        true_states[t] = [level, velocity, log_vol]
        
        # Noisy measurement
        measurements[t] = level + np.random.randn() * 1.0
    
    return measurements, true_states


def generate_stress_data(T: int = 1000, seed: int = 42) -> np.ndarray:
    """
    Generate stress test data with outliers and regime changes.
    """
    np.random.seed(seed)
    
    measurements = np.zeros(T)
    level = 100.0
    velocity = 0.0
    
    for t in range(T):
        # Regime changes
        if t == 300:
            velocity = 0.5  # Trend starts
        elif t == 600:
            velocity = -0.3  # Reversal
        elif t == 800:
            velocity = 0.0  # Flat
        
        # Outliers (5% of data)
        if np.random.rand() < 0.05:
            level += np.random.randn() * 10.0  # Big jump
        else:
            level += velocity + np.random.randn() * 0.5
        
        measurements[t] = level + np.random.randn() * 1.0
    
    return measurements


# =============================================================================
# Correctness Test
# =============================================================================

def test_correctness():
    """Compare filter outputs for correctness."""
    
    print("=" * 70)
    print("CORRECTNESS TEST")
    print("=" * 70)
    
    if not HAS_FILTERPY:
        print("Skipping (FilterPy not installed)")
        return
    
    # Generate data
    measurements, true_states = generate_test_data(T=500)
    
    # Our implementation
    our_ukf = create_trend_filter(nu=1000.0)  # High ν ≈ Gaussian for fair comparison
    
    # FilterPy
    fp_ukf = create_filterpy_ukf()
    
    # Run both filters
    our_states = np.zeros((len(measurements), 3))
    fp_states = np.zeros((len(measurements), 3))
    
    for t, z in enumerate(measurements):
        # Our filter
        our_ukf.step([z])
        our_states[t] = our_ukf.state
        
        # FilterPy
        fp_ukf.predict()
        fp_ukf.update([z])
        fp_states[t] = fp_ukf.x
    
    # Compare
    diff = np.abs(our_states - fp_states)
    
    print(f"\nState estimation difference (absolute):")
    print(f"  Level:    mean={diff[:, 0].mean():.4f}, max={diff[:, 0].max():.4f}")
    print(f"  Velocity: mean={diff[:, 1].mean():.6f}, max={diff[:, 1].max():.6f}")
    print(f"  Log-vol:  mean={diff[:, 2].mean():.4f}, max={diff[:, 2].max():.4f}")
    
    # Track error relative to truth
    our_rmse = np.sqrt(np.mean((our_states[:, 0] - true_states[:, 0]) ** 2))
    fp_rmse = np.sqrt(np.mean((fp_states[:, 0] - true_states[:, 0]) ** 2))
    
    print(f"\nRMSE vs true level:")
    print(f"  Our UKF:     {our_rmse:.4f}")
    print(f"  FilterPy:    {fp_rmse:.4f}")
    print(f"  Difference:  {abs(our_rmse - fp_rmse):.4f}")
    
    # Pass if within 10% of each other
    if abs(our_rmse - fp_rmse) / fp_rmse < 0.1:
        print("\n✓ PASS: Filters produce similar results")
    else:
        print("\n✗ FAIL: Significant difference in filter outputs")


# =============================================================================
# Speed Benchmark
# =============================================================================

def benchmark_speed():
    """Benchmark filter speed."""
    
    print("\n" + "=" * 70)
    print("SPEED BENCHMARK")
    print("=" * 70)
    
    # Test sizes
    sizes = [100, 1000, 10000, 100000]
    
    print(f"\n{'T':>10} | {'Our UKF':>15} | {'FilterPy':>15} | {'Speedup':>10}")
    print("-" * 60)
    
    for T in sizes:
        measurements = np.random.randn(T) * 10 + 100
        
        # Our implementation
        our_ukf = create_trend_filter(nu=4.0)
        
        start = time.perf_counter()
        for z in measurements:
            our_ukf.step([z])
        our_time = time.perf_counter() - start
        
        # FilterPy (if available)
        if HAS_FILTERPY:
            fp_ukf = create_filterpy_ukf()
            
            start = time.perf_counter()
            for z in measurements:
                fp_ukf.predict()
                fp_ukf.update([z])
            fp_time = time.perf_counter() - start
            
            speedup = fp_time / our_time
            print(f"{T:>10} | {our_time*1000:>12.2f} ms | {fp_time*1000:>12.2f} ms | {speedup:>9.1f}x")
        else:
            print(f"{T:>10} | {our_time*1000:>12.2f} ms | {'N/A':>15} | {'N/A':>10}")
    
    # Per-step timing
    print(f"\nPer-step timing (T=100000):")
    print(f"  Our UKF:  {our_time/T*1e6:.2f} μs/step")
    if HAS_FILTERPY:
        print(f"  FilterPy: {fp_time/T*1e6:.2f} μs/step")


# =============================================================================
# Numerical Stability Test
# =============================================================================

def test_numerical_stability():
    """Test numerical stability under stress conditions."""
    
    print("\n" + "=" * 70)
    print("NUMERICAL STABILITY TEST")
    print("=" * 70)
    
    measurements = generate_stress_data(T=1000)
    
    # Our implementation
    ukf = create_trend_filter(nu=4.0)
    ukf.enable_nis_tracking(window_size=50)
    
    n_repairs = 0
    max_nis = 0.0
    
    for t, z in enumerate(measurements):
        ukf.step([z])
        
        # Check for NaN/Inf
        if not np.all(np.isfinite(ukf.state)):
            print(f"  ✗ NaN/Inf in state at t={t}")
            break
        
        if not np.all(np.isfinite(ukf.sqrt_cov)):
            print(f"  ✗ NaN/Inf in covariance at t={t}")
            break
        
        # Track NIS
        max_nis = max(max_nis, ukf.nis)
        
        # Check covariance health
        if not ukf.check_cov_health():
            ukf.repair_cov()
            n_repairs += 1
    else:
        print(f"  ✓ Completed {len(measurements)} steps without NaN/Inf")
    
    print(f"  Covariance repairs: {n_repairs}")
    print(f"  Max NIS: {max_nis:.2f}")
    
    # Final state sanity check
    state = ukf.state
    print(f"\n  Final state:")
    print(f"    Level:    {state[0]:.2f}")
    print(f"    Velocity: {state[1]:.4f}")
    print(f"    Log-vol:  {state[2]:.2f}")
    
    # Check NIS health
    stats = ukf.get_nis_stats()
    print(f"\n  NIS statistics:")
    print(f"    Mean: {stats.mean:.2f} (expected: ~1)")
    print(f"    Outliers: {stats.fraction_above*100:.1f}%")
    
    if stats.fraction_above < 0.5:
        print("\n✓ PASS: Filter remained stable under stress")
    else:
        print("\n✗ WARNING: High outlier rate under stress")


# =============================================================================
# Student-t vs Gaussian Comparison
# =============================================================================

def test_student_t_robustness():
    """Compare Student-t vs Gaussian UKF under outliers."""
    
    print("\n" + "=" * 70)
    print("STUDENT-T ROBUSTNESS TEST")
    print("=" * 70)
    
    np.random.seed(42)
    T = 500
    
    # Generate data with heavy outliers
    true_level = 100.0
    true_velocity = 0.1
    
    measurements = np.zeros(T)
    true_states = np.zeros(T)
    
    for t in range(T):
        true_level += true_velocity
        true_states[t] = true_level
        
        # 10% outliers from heavy-tailed distribution
        if np.random.rand() < 0.1:
            measurements[t] = true_level + np.random.standard_t(2) * 5.0
        else:
            measurements[t] = true_level + np.random.randn() * 0.5
    
    # Gaussian UKF (high ν)
    gauss_ukf = create_trend_filter(nu=1000.0)
    
    # Student-t UKF (ν = 4)
    studentt_ukf = create_trend_filter(nu=4.0)
    
    # Run both
    gauss_states = np.zeros(T)
    studentt_states = np.zeros(T)
    
    for t, z in enumerate(measurements):
        gauss_ukf.step([z])
        gauss_states[t] = gauss_ukf.state[0]
        
        studentt_ukf.step([z])
        studentt_states[t] = studentt_ukf.state[0]
    
    # Compute errors
    gauss_rmse = np.sqrt(np.mean((gauss_states - true_states) ** 2))
    studentt_rmse = np.sqrt(np.mean((studentt_states - true_states) ** 2))
    
    print(f"\nRMSE (level tracking):")
    print(f"  Gaussian UKF:  {gauss_rmse:.4f}")
    print(f"  Student-t UKF: {studentt_rmse:.4f}")
    print(f"  Improvement:   {(gauss_rmse - studentt_rmse) / gauss_rmse * 100:.1f}%")
    
    # Max error (captures outlier sensitivity)
    gauss_max = np.max(np.abs(gauss_states - true_states))
    studentt_max = np.max(np.abs(studentt_states - true_states))
    
    print(f"\nMax error:")
    print(f"  Gaussian UKF:  {gauss_max:.4f}")
    print(f"  Student-t UKF: {studentt_max:.4f}")
    print(f"  Improvement:   {(gauss_max - studentt_max) / gauss_max * 100:.1f}%")
    
    if studentt_rmse < gauss_rmse:
        print("\n✓ Student-t UKF more robust to outliers")
    else:
        print("\n✗ No improvement from Student-t (check data)")


# =============================================================================
# Kelly Integration Test
# =============================================================================

def test_kelly_integration():
    """Test Kelly criterion integration with UKF."""
    
    print("\n" + "=" * 70)
    print("KELLY INTEGRATION TEST")
    print("=" * 70)
    
    # Generate trending data
    np.random.seed(42)
    T = 200
    
    measurements = 100.0 + np.cumsum(0.1 + np.random.randn(T) * 0.5)
    
    # Run filter
    ukf = create_trend_filter(nu=4.0)
    ukf.enable_nis_tracking(window_size=50)
    
    positions = []
    sharpes = []
    
    for z in measurements:
        ukf.step([z])
        
        kelly = Kelly.from_ukf(ukf, fraction=0.5)
        positions.append(kelly.position)
        sharpes.append(kelly.sharpe)
    
    positions = np.array(positions)
    sharpes = np.array(sharpes)
    
    print(f"\nKelly statistics over {T} steps:")
    print(f"  Position: mean={positions.mean():.4f}, std={positions.std():.4f}")
    print(f"  Sharpe:   mean={sharpes.mean():.2f}, std={sharpes.std():.2f}")
    print(f"  Max position: {positions.max():.4f}")
    print(f"  Min position: {positions.min():.4f}")
    
    # Final Kelly result
    final_kelly = Kelly.from_ukf(ukf, fraction=0.5)
    print(f"\nFinal Kelly calculation:")
    print(f"  Expected return: {final_kelly.expected_return:.4f}")
    print(f"  Volatility:      {final_kelly.volatility:.4f}")
    print(f"  Full Kelly:      {final_kelly.f_full:.4f}")
    print(f"  Half Kelly:      {final_kelly.f_half:.4f}")
    print(f"  Final position:  {final_kelly.position:.4f}")
    print(f"  μ uncertainty:   {final_kelly.mu_uncertainty:.4f}")
    
    # Check for reasonable values
    if abs(final_kelly.position) <= Kelly.MAX_LEVERAGE:
        print("\n✓ Kelly positions within limits")
    else:
        print("\n✗ Kelly position exceeded limits")


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " Student-t SQR UKF - Comparison & Validation ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    test_correctness()
    benchmark_speed()
    test_numerical_stability()
    test_student_t_robustness()
    test_kelly_integration()
    
    print("\n" + "=" * 70)
    print("All tests complete.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()