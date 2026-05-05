# SCOPE — Sub-volume Clustering for Observational and Periodic Environments

A Python package (Rust backend) for computing the 2D two-point correlation
function ξ(r_p, π) and the projected correlation function w_p(r_p) from galaxy
catalogues in periodic simulation boxes (e.g. GALFORM on P-Millennium).

---

## The problem it solves

Semi-analytic models run on large simulations that are often partitioned into
**k sub-volumes** for parallel processing. If you use only **m < k** of those
sub-volumes, naively computing the correlation function inflates the one-halo
term by a factor of k/m — because intra-halo pairs are counted only from the
selected sub-volumes but the random catalogue covers the full box.

SCOPE fixes this by:

1. Counting pairs that fall **within** the same sub-volume (`DD_auto`) and
   **across** different sub-volumes (`DD_cross`) simultaneously.
2. Applying scalar correction weights:
   - α = k/m  — rescales one-halo-dominated auto pairs
   - β = k(k−1) / [m(m−1)]  — rescales two-halo cross pairs
3. Using the **Natural Estimator** with an **analytic RR** (no Monte Carlo
   random catalogue needed for a periodic box):

```
DD_corr = α · DD_auto + β · DD_cross
ξ(r_p, π) = DD_corr / RR − 1
w_p(r_p)  = 2 ∫₀^{π_max} ξ(r_p, π) dπ
```

---

## Implementation status

| Component | Status |
|-----------|--------|
| Rust pair counter (`count_pairs_2d`) | **Complete** — anisotropic cell-list, rayon parallel, periodic min-image |
| Python wrapper (`compute_2pcf`) | **Complete** — α/β correction, analytic RR, ξ and w_p |
| Analytic RR (`analytic_rr`) | **Complete** |
| Physics verified | **Yes** — uniform random field gives mean \|ξ\| < 0.02 |
| Jackknife / bootstrap errors | Not yet implemented |
| Multiple LOS axes | Not yet — z is hardcoded as LOS |
| 1D isotropic ξ(r) | Not yet implemented |

---

## Requirements

- Python ≥ 3.9
- Rust toolchain — install via [rustup](https://rustup.rs): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- `maturin` ≥ 1.7 — install via pip or uv (see below)

No C compiler or external libraries are required beyond the above.

---

## Installation

Both `pip` and `uv` are supported. `uv` is recommended for new projects as it is
significantly faster and creates reproducible environments.

### With uv (recommended)

```bash
# install uv itself if needed
pip install uv

cd /path/to/SCOPE/scope

# create a virtual environment and activate it
uv venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# install maturin into the venv, then build + install scope in one step
uv pip install maturin
maturin develop --release      # maturin detects the active .venv automatically
```

To install the pre-built wheel into any uv-managed environment:

```bash
uv pip install /path/to/SCOPE/scope/target/wheels/scope-*.whl
```

To add scope as a dependency in another uv project:

```bash
# from your other project's directory:
uv add scope --find-links /path/to/SCOPE/scope/target/wheels/
```

Or point directly at the source tree (uv will call maturin's PEP 517 backend):

```bash
uv add scope --editable /path/to/SCOPE/scope
```

### With pip

```bash
cd /path/to/SCOPE/scope

# Option A: build a wheel and install it
pip install maturin
maturin build --release
pip install target/wheels/scope-*.whl --user

# Option B: install directly from source (maturin is invoked automatically)
pip install . --user

# Option C: editable install (rebuilds on source changes)
pip install maturin
maturin develop --release
```

### On Cosma without a virtual environment

```bash
pip install maturin --user
cd /path/to/SCOPE/scope
maturin build --release
pip install target/wheels/scope-*.whl --user
```

### Hardlink warning on network filesystems

If you see:

```
warning: Failed to hardlink files; falling back to full copy.
```

this is harmless — uv is working around a limitation of network-mounted
filesystems (e.g. Lustre on Cosma). Suppress it permanently with:

```bash
export UV_LINK_MODE=copy
```

---

## Quick start

```python
import numpy as np
from scope import compute_2pcf

# --- your galaxy catalogue ---
# coords   : (N, 3) float64, units Mpc/h, z-axis is the line of sight
# subvol_ids : (N,) int32, integer label 0 … m-1 identifying which of the
#              m selected sub-volumes each galaxy belongs to
coords     = np.load("coords.npy")          # shape (N, 3)
subvol_ids = np.load("subvol_ids.npy")      # shape (N,), dtype int32
n_total    = 15_000_000   # total galaxies in the FULL simulation

# --- binning ---
r_p_bins = np.logspace(-1, 1.5, 16)   # 15 log-spaced bins, 0.1–31 Mpc/h
pi_bins  = np.linspace(0, 60, 13)     # 12 linear bins, 0–60 Mpc/h

# --- run ---
result = compute_2pcf(
    coords        = coords,
    subvol_ids    = subvol_ids,
    r_p_bins      = r_p_bins,
    pi_bins       = pi_bins,
    box_size      = 800.0,   # P-Millennium box, Mpc/h
    n_subvols     = 512,     # k  — total sub-volumes in the full simulation
    n_subvols_selected = 64, # m  — sub-volumes you actually loaded
    n_total       = n_total,
)

xi = result["xi"]   # shape (n_rp, n_pi)  — ξ(r_p, π)
wp = result["wp"]   # shape (n_rp,)       — w_p(r_p)

# r_p bin centres for plotting
r_p_centres = np.sqrt(r_p_bins[:-1] * r_p_bins[1:])  # geometric mean
```

### Return dictionary

| Key | Shape | Description |
|-----|-------|-------------|
| `xi` | (n_rp, n_pi) | Two-dimensional ξ(r_p, π) |
| `wp` | (n_rp,) | Projected w_p(r_p) |
| `dd_auto` | (n_rp, n_pi) | Raw same-subvol pair counts |
| `dd_cross` | (n_rp, n_pi) | Raw cross-subvol pair counts |
| `dd_corr` | (n_rp, n_pi) | Corrected pair counts (α·auto + β·cross) |
| `rr` | (n_rp, n_pi) | Analytic RR pair counts |

### Accessing the Rust function directly

If you only want the raw pair counts (e.g. to apply your own estimator):

```python
from scope import count_pairs_2d   # the Rust function

dd_auto, dd_cross = count_pairs_2d(
    coords,       # (N, 3) float64, C-contiguous
    subvol_ids,   # (N,) int32
    r_p_bins,     # (n_rp+1,) float64
    pi_bins,      # (n_pi+1,) float64
    box_size,     # float
)
```

---

## Sub-volume assignment — important note

The correction weights assume sub-volumes are **3D spatial cells** that are
statistically equivalent (same mean density, same clustering environment).
The sub-volume ID for each galaxy must **not** correlate with the separation
axes being measured.

**Good:** assign sub-volumes by a 3D grid cell index (e.g. octants, or a
regular nx×ny×nz tiling of the box).

**Avoid:** assigning sub-volumes as pure z-slices when z is also the
line-of-sight axis. This correlates the auto/cross split with the π coordinate
and biases the small-π signal.

---

## Building from source (development)

```bash
git clone <repo>
cd scope

# Install dependencies
pip install maturin

# Build + install in development mode (rebuilds on source changes)
maturin develop --release

# Or build a wheel manually
maturin build --release
```

The Rust source is in `src/lib.rs`. After editing it, re-run `maturin develop`
or `maturin build` to recompile.

---

## Project layout

```
scope/
├── Cargo.toml          # Rust dependencies (pyo3, numpy, rayon)
├── pyproject.toml      # Python build config (maturin backend)
├── src/
│   └── lib.rs          # Rust: cell-list pair counter, PyO3 bindings
└── python/
    └── scope/
        └── __init__.py # Python: compute_2pcf, analytic_rr, correction weights
```

---

## Algorithm

**Pair counting (Rust):**  
An anisotropic cell list is built with separate cell granularity for the
transverse (r_p) and line-of-sight (π) directions. For each particle, only
the 3×3×3 = 27 neighbouring cells are searched. Periodic boundary conditions
are applied via the minimum-image convention. Each unordered pair (i, j) with
i < j is counted exactly once, and routed to `DD_auto` or `DD_cross` depending
on whether `subvol_id[i] == subvol_id[j]`. The outer loop is parallelised with
Rayon using a thread-local fold/reduce pattern (no locking).

**Analytic RR:**  
For a uniform distribution in a periodic box of volume V = L³ with N_total
galaxies, the expected random–random pair count in a cylindrical annular shell
(r_p ∈ [a, b], |dz| ∈ [c, d]) is:

```
RR = N(N-1)/2 · π(b²-a²) · 2(d-c) / L³
```

The factor of 2 accounts for pairs with both positive and negative dz mapping
to the same |dz| bin.
