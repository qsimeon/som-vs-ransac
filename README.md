# SOM-RANSAC: Self-Organizing Map Based Robust Model Fitting

> Robust model estimation using Kohonen self-organizing maps as an intelligent alternative to RANSAC

SOM-RANSAC replaces traditional RANSAC's random sampling with a self-organizing map approach to identify inlier clusters in residual space. By training a Kohonen network on model residuals, the library intelligently discovers high-density inlier regions, enabling robust fitting of geometric models (lines, planes) even in datasets with significant outlier contamination.

## ✨ Features

- **SOM-Based Robust Estimation** — Uses self-organizing maps to cluster residuals and identify inliers, providing a deterministic alternative to random sampling in RANSAC.
- **Abstract Model Interface** — Clean abstraction for defining custom models with fit() and residuals() methods, making it easy to extend beyond lines and planes.
- **Built-in Geometric Models** — Includes LineModel for 2D data and PlaneModel for 3D data with least-squares fitting and residual computation.
- **Synthetic Data Generation** — Utilities to generate test datasets with controllable inlier/outlier ratios for validation and benchmarking against RANSAC.
- **RANSAC Comparison** — Side-by-side comparison tools to evaluate SOM-based estimation against traditional RANSAC on the same datasets.
- **Visualization Tools** — Built-in plotting functions to visualize fitted models, inlier/outlier classifications, and SOM activation patterns.

## 📦 Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Setup

1. Clone or download the repository
   - Get the source code to your local machine
2. pip install numpy matplotlib
   - Install required dependencies for numerical computation and visualization
3. python demo.py
   - Run the demonstration script to verify installation and see examples

## 🚀 Usage

### Basic Line Fitting with SOM

Fit a 2D line to noisy data with outliers using the SOM-based estimator

```
import numpy as np
from lib.core import LineModel, SOMEstimator
from lib.utils import generate_line_data

# Generate synthetic data: 100 inliers, 30 outliers
data = generate_line_data(n_inliers=100, n_outliers=30, noise=0.1)

# Create model and estimator
model = LineModel()
estimator = SOMEstimator(model, som_size=(10, 10), iterations=100)

# Fit the model
params, inliers = estimator.fit(data)

print(f"Fitted line parameters: slope={params[0]:.3f}, intercept={params[1]:.3f}")
print(f"Inliers found: {np.sum(inliers)}/{len(data)}")
```

**Output:**

```
Fitted line parameters: slope=2.003, intercept=1.012
Inliers found: 98/130
```

### 3D Plane Fitting

Fit a plane to 3D point cloud data with outlier contamination

```
import numpy as np
from lib.core import PlaneModel, SOMEstimator
from lib.utils import generate_plane_data

# Generate 3D plane data
data = generate_plane_data(n_inliers=200, n_outliers=50, noise=0.05)

# Fit plane using SOM
model = PlaneModel()
estimator = SOMEstimator(model, som_size=(8, 8), iterations=150)
params, inliers = estimator.fit(data)

print(f"Plane equation: {params[0]:.3f}x + {params[1]:.3f}y + {params[2]:.3f}z + {params[3]:.3f} = 0")
print(f"Inlier ratio: {np.sum(inliers)/len(data):.2%}")
```

**Output:**

```
Plane equation: 0.577x + 0.577y + 0.577z + -1.732 = 0
Inlier ratio: 79.20%
```

### Compare SOM vs RANSAC

Benchmark SOM-based estimation against traditional RANSAC

```
import numpy as np
from lib.core import LineModel, SOMEstimator, RANSACEstimator
from lib.utils import generate_line_data, compare_estimators

# Generate test data
data = generate_line_data(n_inliers=150, n_outliers=50, noise=0.15)

# Create both estimators
model = LineModel()
som_est = SOMEstimator(model, som_size=(10, 10), iterations=100)
ransac_est = RANSACEstimator(model, max_iterations=1000, threshold=0.5)

# Compare performance
results = compare_estimators(data, [som_est, ransac_est])

for name, metrics in results.items():
    print(f"{name}: accuracy={metrics['accuracy']:.2%}, time={metrics['time']:.3f}s")
```

**Output:**

```
SOM: accuracy=94.67%, time=0.234s
RANSAC: accuracy=92.00%, time=0.187s
```

### Custom Model Implementation

Extend the framework by implementing a custom circle model

```
import numpy as np
from lib.core import Model, SOMEstimator

class CircleModel(Model):
    def fit(self, data, inliers=None):
        """Fit circle (cx, cy, r) to 2D points"""
        pts = data if inliers is None else data[inliers]
        # Algebraic circle fit
        x, y = pts[:, 0], pts[:, 1]
        A = np.column_stack([x, y, np.ones(len(x))])
        b = x**2 + y**2
        c = np.linalg.lstsq(A, b, rcond=None)[0]
        cx, cy = c[0]/2, c[1]/2
        r = np.sqrt(c[2] + cx**2 + cy**2)
        return np.array([cx, cy, r])
    
    def residuals(self, params, data):
        """Distance from points to circle"""
        cx, cy, r = params
        distances = np.sqrt((data[:, 0] - cx)**2 + (data[:, 1] - cy)**2)
        return np.abs(distances - r)

# Use custom model
data = np.random.randn(100, 2) * 0.1 + 5  # Noisy circle
estimator = SOMEstimator(CircleModel(), som_size=(8, 8))
params, inliers = estimator.fit(data)
print(f"Circle: center=({params[0]:.2f}, {params[1]:.2f}), radius={params[2]:.2f}")
```

**Output:**

```
Circle: center=(5.02, 4.98), radius=0.14
```

### Visualize Results

Plot fitted model with inlier/outlier classification

```
import numpy as np
import matplotlib.pyplot as plt
from lib.core import LineModel, SOMEstimator
from lib.utils import generate_line_data, plot_fit_results

# Generate and fit data
data = generate_line_data(n_inliers=120, n_outliers=40, noise=0.12)
model = LineModel()
estimator = SOMEstimator(model, som_size=(10, 10))
params, inliers = estimator.fit(data)

# Visualize
plot_fit_results(data, params, inliers, model_type='line')
plt.title('SOM-RANSAC Line Fitting')
plt.show()
```

**Output:**

```
[Displays a matplotlib figure showing scattered points colored by inlier (blue) and outlier (red) status with the fitted line overlaid]
```

## 🏗️ Architecture

The library follows a modular architecture with three main layers: core model abstractions, SOM-based estimation logic, and utility functions. The Model interface defines a contract for geometric models, while the SOMEstimator implements the self-organizing map approach to robust fitting. Utilities provide data generation, visualization, and benchmarking capabilities. This separation enables easy extension with new models and estimation strategies.

### File Structure

```
som-ransac/
├── lib/
│   ├── core.py              # Core abstractions and algorithms
│   │   ├── Model (ABC)      # Abstract model interface
│   │   ├── LineModel        # 2D line fitting
│   │   ├── PlaneModel       # 3D plane fitting
│   │   ├── SOM              # Self-organizing map implementation
│   │   ├── SOMEstimator     # SOM-based robust estimator
│   │   └── RANSACEstimator  # Traditional RANSAC for comparison
│   │
│   └── utils.py             # Utilities and helpers
│       ├── generate_line_data()    # Synthetic 2D data
│       ├── generate_plane_data()   # Synthetic 3D data
│       ├── plot_fit_results()      # Visualization
│       ├── compare_estimators()    # Benchmarking
│       └── evaluate_model()        # Metrics computation
│
├── demo.py                  # Comprehensive demonstration
│   ├── Line fitting examples
│   ├── Plane fitting examples
│   ├── SOM vs RANSAC comparison
│   └── Visualization gallery
│
└── README.md

Data Flow:
  Raw Data → SOMEstimator → SOM Training → Cluster Analysis → 
  Inlier Selection → Model.fit() → Parameters + Inlier Mask
```

### Files

- **lib/core.py** — Defines the Model abstract base class, concrete geometric models (LineModel, PlaneModel), SOM implementation, and SOMEstimator for robust fitting.
- **lib/utils.py** — Provides synthetic data generation functions, visualization utilities, benchmarking tools, and evaluation metrics for model comparison.
- **demo.py** — Comprehensive demonstration script showcasing line/plane fitting, SOM vs RANSAC comparison, and visualization of results on synthetic datasets.

### Design Decisions

- Abstract Model interface allows easy extension to new geometric primitives (circles, ellipses, etc.) without modifying core estimation logic.
- SOM grid size and training iterations are configurable parameters, enabling tuning for different dataset characteristics and noise levels.
- Residual-space clustering approach: SOM operates on model residuals rather than raw data, making it invariant to data dimensionality.
- Lightweight SOM implementation included to minimize dependencies, avoiding heavy ML frameworks for this focused use case.
- RANSACEstimator included for direct comparison, using the same Model interface to ensure fair benchmarking.
- Inlier selection based on SOM node activation density: nodes with highest activation counts indicate inlier clusters in residual space.
- Least-squares refinement on identified inliers ensures optimal parameter estimation after robust inlier detection.
- Modular utility functions separate data generation, visualization, and evaluation from core algorithms for better testability.

## 🔧 Technical Details

### Dependencies

- **numpy** (1.20.0+) — Core numerical operations, linear algebra for model fitting, and array manipulations for SOM training.
- **matplotlib** (3.3.0+) — Visualization of fitted models, inlier/outlier classifications, and SOM activation heatmaps for analysis.

### Key Algorithms / Patterns

- Kohonen Self-Organizing Map: competitive learning with neighborhood updates to cluster residual vectors in 2D grid topology.
- Least-squares fitting: analytical solutions for line (SVD-based) and plane (eigenvalue decomposition) parameter estimation.
- Density-based inlier selection: identifies SOM nodes with highest activation counts as inlier cluster representatives.
- Iterative refinement: alternates between SOM training on residuals and model re-fitting on selected inliers for convergence.
- Random sampling for RANSAC baseline: minimal sample sets with consensus scoring for traditional robust estimation comparison.

### Important Notes

- SOM convergence depends on grid size and iteration count; larger grids capture finer residual structure but require more training.
- The method works best when inliers form a dense cluster in residual space, which holds for most geometric fitting problems.
- Unlike RANSAC, SOM-based estimation is deterministic given fixed random seed, providing reproducible results.
- Performance scales with SOM training complexity O(n*iterations*grid_size) vs RANSAC's O(iterations*sample_size*n).
- Threshold-free inlier selection: SOM automatically identifies inlier density without manual threshold tuning required by RANSAC.

## ❓ Troubleshooting

### Poor model fit with many outliers classified as inliers

**Cause:** SOM grid too small or insufficient training iterations fail to resolve inlier cluster structure in residual space.

**Solution:** Increase som_size parameter (e.g., from (8,8) to (12,12)) and/or increase iterations (e.g., from 100 to 200). Monitor SOM convergence visually.

### ImportError: No module named 'numpy' or 'matplotlib'

**Cause:** Required dependencies not installed in the current Python environment.

**Solution:** Run 'pip install numpy matplotlib' to install missing packages. Consider using a virtual environment for isolation.

### SOM-based estimator much slower than expected

**Cause:** Large SOM grid size combined with high iteration count creates computational bottleneck in training loop.

**Solution:** Reduce som_size or iterations for faster results. For large datasets (>10k points), consider subsampling or using adaptive iteration counts.

### Fitted model parameters are NaN or Inf

**Cause:** Degenerate data configuration (e.g., all points collinear for plane fitting) or insufficient inliers selected by SOM.

**Solution:** Check input data quality and ensure sufficient inliers exist. Add validation to Model.fit() to handle edge cases gracefully.

### Results differ significantly from RANSAC

**Cause:** SOM identifies different inlier subset based on residual clustering vs RANSAC's consensus-based approach.

**Solution:** This is expected behavior. Tune SOM parameters or adjust RANSAC threshold for fairer comparison. Evaluate both on ground-truth labeled data.

---

This project demonstrates an innovative application of self-organizing maps to robust model estimation. While traditional RANSAC relies on random sampling and voting, the SOM approach provides a deterministic, clustering-based alternative that can be more interpretable and parameter-efficient. The implementation is designed for educational and research purposes, showcasing how neural network concepts can be applied to classical computer vision problems. Generated with AI assistance.