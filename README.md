# SUGC

[![PyPI version](https://img.shields.io/pypi/v/sugc.svg)](https://pypi.org/project/sugc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Correlation function counters for galaxy catalogues drawn from sparse sub-volumes of periodic N-body simulations. Written in Rust with a Python interface via PyO3.

Supports real-space ξ(r), redshift-space ξ(s, μ) multipoles, and high-order N-point functions (N ≥ 3). Pair counts are corrected for the statistical under-sampling that arises when working with m-of-k independent realisations of a simulation box.

## Installation

```bash
uv add sugc
```

## Usage

```python
import numpy as np
from sugc import count_pairs_1d

coords = np.random.uniform(0, 500.0, size=(100_000, 3))
subvol_ids = np.zeros(100_000, dtype=np.int32)  # single sub-volume
r_bins = np.linspace(0.1, 50.0, 20)

counts, weights = count_pairs_1d(coords, subvol_ids, r_bins, box_size=500.0)
```

See `examples/` for redshift-space distortions and 3-point correlation functions.

## Citation

If you use this in published work, please cite:

> Hickman, O. et al. (2026). *Fast and Unbiased Clustering Estimators for Sparse Sub-volumes.* (In Prep).

## License

MIT — see [LICENSE](LICENSE).
