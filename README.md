# SCOPE: Sparse Clustering On Periodic-box Estimator

[![PyPI version](https://img.shields.io/pypi/v/scope-corr.svg)](https://pypi.org/project/scope-corr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SCOPE** is a high-performance hybrid Rust/Python package designed for computing two-point and N-point correlation functions from galaxy catalogues. It is specifically optimized for datasets drawn from sparse sub-volumes of periodic N-body simulations.

## Key Features

- **Fast Correlation Functions:** Leverages Rust for heavy lifting with a clean Python interface.
- **Real-Space & Redshift-Space:** Support for $\xi(r)$ and $\xi(s, \mu)$ multipoles.
- **N-Point Support:** High-order correlation counters ($N=3, 4, \dots$) with optimized recursive pruning.
- **Unbiased Estimators:** Built-in weights to correct for statistical under-sampling in sparse sub-volumes.
- **Hardware Accelerated:** Automatic CPU/GPU routing based on task complexity and dataset size.

## Installation

You can install SCOPE directly from PyPI using \`uv\` or \`pip\`:

\`\`\`bash
uv pip install scope-corr
\`\`\`

or

\`\`\`bash
pip install scope-corr
\`\`\`

## Quick Start

\`\`\`python
import numpy as np
import scope

# Generate some random data
n_galaxies = 100_000
box_size = 500.0
coords = np.random.uniform(0, box_size, size=(n_galaxies, 3))

# Compute the 2-point correlation function
r_bins = np.linspace(0.1, 50.0, 20)
xi = scope.pairs_1d(coords, r_bins, box_size=box_size)

print(f"Computed xi(r) at {len(r_bins)-1} bins")
\`\`\`

## Advanced Usage

For more detailed examples, including Redshift-Space Distortions (RSD) and 3-point correlation functions, please refer to the \`examples/\` directory in the repository.

## Citation

If you use SCOPE in your research, please cite:

> Hickman, O. et al. (2026). "Fast and Unbiased Clustering Estimators for Sparse Sub-volumes." (In Prep).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
