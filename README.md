# Square-Root UKF

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![MKL](https://img.shields.io/badge/Intel-MKL-0071C5.svg)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html)

MKL-accelerated Square-Root Unscented Kalman Filter with Student-t outlier rejection.

<img width="1386" alt="UKF filtering SPY price data showing level tracking, velocity regime detection, volatility spike during crash, and NIS model health monitoring" src="https://github.com/user-attachments/assets/762e2920-98de-422e-b222-6dbb9adc4915" />

<img width="1387" alt="Student-t vs Gaussian comparison and Kelly strategy backtest on SPY data" src="https://github.com/user-attachments/assets/6d41b921-26e7-45ea-a902-0e39bcbe8778" />

## Features

- **Student-t likelihood** — Robust to measurement outliers (configurable ν)
- **Square-root covariance** — Numerical stability via Cholesky factorization
- **Fast** — 1.7 μs/step on Intel CPUs (590K updates/sec)
- **NIS health monitoring** — Windowed statistics for filter divergence detection
- **Python bindings** — ctypes wrapper with NumPy integration
- **Kelly criterion** — Position sizing module that integrates with UKF output

## Performance

| Implementation | Time/step | Steps/sec | vs FilterPy |
|----------------|-----------|-----------|-------------|
| C (MKL batch)  | 1.72 μs   | 590K      | 38× faster  |
| Python ctypes  | 3.86 μs   | 260K      | 17× faster  |
| FilterPy       | 64.6 μs   | 15K       | baseline    |

*Benchmarked on Intel i9-14900K, nx=3, nz=1, Student-t ν=4*

**Note:** "Batch" processes N measurements in a single Python→C call, eliminating per-call overhead (~2.1 μs). Use `srukf_step_batch()` for maximum throughput when processing historical data.

## Quick Start

### C

```c
#include "student_t_srukf.h"

// Create filter: 3 states, 1 measurement, Student-t ν=4
StudentT_SRUKF* ukf = srukf_create(3, 1, 4.0);

// Configure dynamics: x_{k+1} = F @ x_k
double F[9] = {1,0,0, 1,1,0, 0,0,0.95};
srukf_set_dynamics(ukf, F);

// Configure measurement: z_k = H @ x_k  
double H[3] = {1, 0, 0};
srukf_set_measurement(ukf, H);

// Set noise covariances (sqrt form)
double Sq[9] = {0.1,0,0, 0,0.01,0, 0,0,0.05};
double Sr[1] = {1.0};
srukf_set_process_noise(ukf, Sq);
srukf_set_measurement_noise(ukf, Sr);

// Run filter
double z[1];
for (int i = 0; i < n_measurements; i++) {
    z[0] = measurements[i];
    srukf_step(ukf, z);
    
    const double* x = srukf_get_state(ukf);
    printf("State: %.2f, %.4f, %.2f\n", x[0], x[1], x[2]);
}

srukf_destroy(ukf);
```

### Python

```python
import numpy as np
from srukf import StudentTSRUKF, Kelly, create_trend_filter

# Quick start with pre-configured trend filter
ukf = create_trend_filter(nu=4.0)

# Run filter
z = np.zeros(1)
for measurement in measurements:
    z[0] = measurement
    ukf.step(z)
    print(f"State: {ukf.state}, NIS: {ukf.nis:.2f}")

# Kelly position sizing
kelly = Kelly.from_ukf(ukf, fraction=0.5)
print(f"Position: {kelly.position:.3f}, Sharpe: {kelly.sharpe:.2f}")

# Batch processing (faster - single Python→C call)
states, covariances, nis_values = ukf.filter(measurements)
```

### Demo Notebook

See [`demo_ukf_finance.ipynb`](python/demo_ukf_finance.ipynb) for a complete example on real market data (SPY), including:
- UKF state visualization (level, velocity, volatility)
- Student-t vs Gaussian robustness comparison
- Kelly criterion position sizing
- Simple backtest

## Building

### Prerequisites

**Intel oneAPI Math Kernel Library (MKL)**

Download from [Intel oneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) (free).

After installation, MKL is typically located at:
- Windows: `C:\Program Files (x86)\Intel\oneAPI\mkl\<version>\`
- Linux: `/opt/intel/oneapi/mkl/<version>/`

**Python 3.8+** (required for `os.add_dll_directory()` on Windows)

### CMake Build

```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build static library
cmake --build build --config Release

# Build shared library (for Python)
cmake --build build --target student_t_srukf_shared --config Release

# Run tests
cd build
ctest -C Release
```

### Windows Setup

Intel provides a script that sets all required environment variables:

```bash
# Before running Python scripts, call:
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
```

**Helper scripts included:**

| Script | Purpose |
|--------|---------|
| `run.bat` | Calls `setvars.bat` then runs Python script |
| `deploy_python.bat` | Copies DLL to python folder |

Usage:
```bash
# From project root
deploy_python.bat              # Copy DLL to python/
cd python
..\run.bat compare_ukf.py      # Run with MKL environment
```

### Linux Setup

```bash
# Source MKL environment
source /opt/intel/oneapi/setvars.sh

# Or add to ~/.bashrc:
export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/latest/lib/intel64:$LD_LIBRARY_PATH
```

## Project Structure

```
Square-Root-UKF/
├── MKL/
│   ├── student_t_srukf.c      # Main implementation
│   ├── student_t_srukf.h      # Public API
│   ├── student_t_srukf.def    # Windows DLL exports
│   └── kelly.h                # Kelly criterion (header-only)
├── python/
│   ├── srukf.py               # Python bindings
│   ├── compare_ukf.py         # FilterPy comparison
│   ├── bench_overhead.py      # Python overhead benchmark
│   └── demo_ukf_finance.ipynb # Real data demo
├── test/
│   ├── test_srukf.c           # UKF unit tests (11 tests)
│   ├── test_kelly.c           # Kelly unit tests (14 tests)
│   └── bench_srukf.c          # C performance benchmark
├── CMakeLists.txt
├── run.bat                    # Windows Python launcher
├── deploy_python.bat          # DLL deployment script
└── README.md
```

## Tests

### C Tests

```bash
# Build and run
cmake --build build --target test_srukf test_kelly --config Release
./build/Release/test_srukf
./build/Release/test_kelly
```

**test_srukf** (11 tests):
- Filter creation/destruction
- State propagation
- Measurement update
- Student-t weighting
- NIS computation
- Covariance health checks
- Missing data handling
- Serialization/deserialization

**test_kelly** (14 tests):
- Basic Kelly formula
- Tail-adjusted Kelly (Student-t)
- Bayesian variance from log-volatility
- Weak signal filtering
- Multi-asset Kelly
- Transaction cost adjustment
- Health scaling

### Python Tests

```bash
cd python

# Compare with FilterPy (correctness + speed)
python compare_ukf.py

# Measure Python ctypes overhead
python bench_overhead.py
```

**compare_ukf.py**:
- Correctness test vs FilterPy
- Speed benchmark (100 to 100K steps)
- Numerical stability under outliers
- Student-t vs Gaussian robustness

## API Reference

### Core Functions

```c
// Lifecycle
StudentT_SRUKF* srukf_create(int nx, int nz, double nu);
void srukf_destroy(StudentT_SRUKF* ukf);

// Configuration
void srukf_set_state(StudentT_SRUKF* ukf, const double* x0);
void srukf_set_sqrt_cov(StudentT_SRUKF* ukf, const double* S0);
void srukf_set_dynamics(StudentT_SRUKF* ukf, const double* F);
void srukf_set_measurement(StudentT_SRUKF* ukf, const double* H);
void srukf_set_process_noise(StudentT_SRUKF* ukf, const double* Sq);
void srukf_set_measurement_noise(StudentT_SRUKF* ukf, const double* Sr);

// Filtering
void srukf_predict(StudentT_SRUKF* ukf);
void srukf_update(StudentT_SRUKF* ukf, const double* z);
void srukf_step(StudentT_SRUKF* ukf, const double* z);  // predict + update
void srukf_step_batch(StudentT_SRUKF* ukf, const double* z_all, int n_steps);

// State access
const double* srukf_get_state(const StudentT_SRUKF* ukf);
const double* srukf_get_sqrt_cov(const StudentT_SRUKF* ukf);
double srukf_get_nis(const StudentT_SRUKF* ukf);

// Health monitoring
void srukf_enable_nis_tracking(StudentT_SRUKF* ukf, int window_size, double threshold);
bool srukf_nis_healthy(const StudentT_SRUKF* ukf);
```

### Kelly Criterion

```c
#include "kelly.h"

// From UKF state
KellyResult result;
kelly_from_ukf(ukf->x, ukf->S, nx, vel_idx, vol_idx, nu, fraction, &result);
printf("Position: %.3f\n", result.f_final);

// Simple calculation
double f = kelly_simple(mu, sigma, nu, fraction);

// Multi-asset
kelly_multi_asset(mu, S, n_assets, nu, fraction, positions, workspace);
```

## Why Square-Root?

Standard UKF propagates covariance P directly, which can lose positive-definiteness due to numerical errors. Square-root UKF propagates the Cholesky factor S where P = SSᵀ, guaranteeing positive-definiteness by construction.

## Why Student-t?

Gaussian likelihood assigns near-zero probability to outliers, causing the filter to "chase" them. Student-t likelihood with low ν (4-6) has heavier tails — outliers get down-weighted automatically:

```
weight = (ν + nz) / (ν + NIS)
```

Large NIS (outlier) → small weight → measurement ignored.

## Known Limitations

- **Velocity lags reversals** — The trend estimate is backward-looking. V-shaped recoveries will be detected late. Consider adding regime change detection (e.g., BOCPD) for faster adaptation.

- **Log-volatility is unobservable** — With H = [1, 0, 0], the filter cannot directly observe volatility. The log-vol state decays to its prior. Use realized volatility for risk estimates.

- **Linear dynamics only** — This implementation uses linear F and H matrices. For nonlinear dynamics, the sigma point propagation would need modification.

## References

- Van der Merwe, R. (2004). *Sigma-Point Kalman Filters for Probabilistic Inference in Dynamic State-Space Models*. PhD thesis, Oregon Health & Science University.

- Roth, M., Özkan, E., & Gustafsson, F. (2013). *A Student's t Filter for Heavy Tailed Process and Measurement Noise*. ICASSP.

- Arasaratnam, I., & Haykin, S. (2009). *Cubature Kalman Filters*. IEEE Transactions on Automatic Control.

## License

GPL-3.0 — see [LICENSE](LICENSE) for details.

## Author

[Tugbars](https://github.com/Tugbars)
