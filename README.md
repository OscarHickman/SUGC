# SCOPE — Sparse Clustering On Periodic-box Estimator

A Python package (Rust backend) for computing the real-space two-point
correlation function **ξ(r)** from galaxy catalogues drawn from a sparse
subset of sub-volumes of a periodic N-body simulation
(e.g. GALFORM on P-Millennium, 542.16 Mpc/h).

---

## The problem it solves

Semi-analytic models run on large periodic simulations partitioned into
**k sub-volumes**. If you load only **m < k** of those sub-volumes, naively
computing ξ(r) biases the one-halo signal because:

- **One-halo pairs** (two galaxies in the same halo) contribute whenever
  their host halo is selected — probability m/k.
- **Two-halo pairs** (galaxies in different haloes) contribute only when
  *both* haloes are selected — probability (m/k)².

The usual N(N−1) normalisation assumes all N_total galaxies are present.
Using only N_selected = (m/k)·N_total inflates the one-halo term by k/m
relative to the two-halo term, distorting the small-scale signal.

SCOPE fixes this by:

1. Counting pairs **within** the same sub-volume (`DD_auto`) and **across**
   different sub-volumes (`DD_cross`) in a single pass.
2. Applying scalar correction weights (Hickman et al. 2026, Eqs. 9–10):

```
α = m/k
β = m(k−1) / [k(m−1)]

DD_corr = α · DD_auto + β · DD_cross
```

   These weights satisfy ⟨DD_corr⟩ = (m/k)²·(DD_1h + DD_2h), matching the
   natural normalisation of the selected catalogue.

3. Using the **Natural Estimator** with an **analytic RR**
   (no Monte Carlo random catalogue needed for a periodic box):

```
ξ(r) = DD_corr / RR − 1
RR(r) = N(N−1)/2 · (4π/3)(r_hi³ − r_lo³) / L³
```

---

## Requirements

- Python ≥ 3.9
- Rust toolchain — install via [rustup](https://rustup.rs)
- `maturin` ≥ 1.7

---

## Installation

### On Cosma (no virtual environment)

```bash
pip install maturin --user
cd /path/to/SCOPE
maturin build --release
pip install target/wheels/scope-*.whl --user
```

Suppress the harmless Lustre hardlink warning:
```bash
export UV_LINK_MODE=copy
```

### With uv (recommended for local development)

```bash
cd /path/to/SCOPE
uv venv .venv && source .venv/bin/activate
uv pip install maturin
maturin develop --release
```

### With pip (wheel)

```bash
cd /path/to/SCOPE
pip install maturin
maturin build --release
pip install target/wheels/scope-*.whl --user
```

---

## Quick start

```python
import numpy as np
from scope import compute_xi

# Galaxy positions (Mpc/h) and sub-volume labels
coords     = np.load("coords.npy")      # shape (N, 3), float64, C-contiguous
subvol_ids = np.load("subvol_ids.npy")  # shape (N,), int32

# Radial bins: 10 kpc/h to 256 Mpc/h (box/2 for P-Millennium)
r_bins = np.logspace(np.log10(0.01), np.log10(256.0), 31)  # 30 bins

result = compute_xi(
    coords              = coords,
    subvol_ids          = subvol_ids,
    r_bins              = r_bins,
    box_size            = 542.16,   # P-Millennium, Mpc/h
    n_subvols           = 1024,    # k — total independent realisations
    n_subvols_selected  = 8,       # m — realisations actually loaded
)

xi    = result["xi"]     # (n_r,) — ξ(r)
r_mid = result["r_mid"]  # (n_r,) — geometric-mean bin centres
```

### Return dictionary

| Key | Shape | Description |
|-----|-------|-------------|
| `xi` | (n_r,) | Real-space ξ(r) |
| `r_mid` | (n_r,) | Geometric-mean bin centres |
| `dd_auto` | (n_r,) | Raw same-subvol pair counts |
| `dd_cross` | (n_r,) | Raw cross-subvol pair counts |
| `dd_corr` | (n_r,) | Corrected pair counts (α·auto + β·cross) |
| `rr` | (n_r,) | Analytic RR pair counts |

### Raw pair counts only

```python
from scope import count_pairs_1d

dd_auto, dd_cross = count_pairs_1d(
    coords,       # (N, 3) float64, C-contiguous
    subvol_ids,   # (N,) int32
    r_bins,       # (n_r+1,) float64
    box_size,     # float
)
```

---

## Sub-volume assignment

Sub-volume IDs label **independent statistical realisations** of the full
simulation box — **not** spatial cells or patches. Every realisation spans
the full coordinate range [0, box_size) in all three dimensions; realisations
overlap completely. Stacking all k realisations recovers the full-density
catalogue. The correction weights assume all realisations are statistically
equivalent (same mean number density, same large-scale structure).

Minimum useful m is **2** — with m=1 there are no cross-pairs and the
intra/inter-realisation decomposition is undefined.

---

## Project layout

```
SCOPE/
├── Cargo.toml             # Rust deps: pyo3, numpy, rayon
├── pyproject.toml         # maturin build backend
├── src/
│   ├── lib.rs             # module declarations + #[pymodule]
│   ├── cell_list.rs       # CellList, HALF_SHELL, find_bin helpers
│   ├── pairs_1d.rs        # count_pairs_1d (real-space ξ(r))
│   ├── pairs_2d.rs        # count_pairs_2d (legacy r_p, π)
│   └── pairs_smu.rs       # count_pairs_smu (redshift-space s, μ)
├── python/scope/
│   └── __init__.py        # Python: compute_xi, analytic_rr_1d (primary API)
│                          #         compute_xi_smu, analytic_rr_smu (RSD)
│                          #         compute_2pcf, analytic_rr (legacy r_p, π)
├── tests/
│   ├── unit/test_pairs_1d.py  # pytest unit tests: count_pairs_1d, analytic_rr_1d, compute_xi
│   │           test_pairs_2d.py  # pytest unit tests: count_pairs_2d, analytic_rr, compute_2pcf
│   │           test_pairs_smu.py # pytest unit tests: count_pairs_smu, analytic_rr_smu, compute_xi_smu
│   └── performance/
│       ├── benchmark_corrfunc.py  # SCOPE vs Corrfunc DD — timing at fixed range
│       └── benchmark_scale.py     # SCOPE vs Corrfunc DD — timing vs r_max sweep
└── examples/
    ├── pairs_1d.ipynb         # end-to-end demo: real-space ξ(r)
    └── pairs_smu.ipynb        # end-to-end demo: redshift-space ξ(s, μ)
```

---

## Algorithm

**Pair counting (Rust, `count_pairs_1d`):**  
An isotropic cell list is built with cell size = r_max, capped at 128 cells
per dimension to bound memory at sub-Mpc search radii. For each particle, the
3×3×3 = 27 neighbouring cells are searched under periodic minimum-image
boundary conditions. Each unordered pair (i < j) is routed to `DD_auto`
(same sub-volume ID) or `DD_cross` (different IDs). The outer loop is
parallelised with Rayon via a thread-local fold/reduce — no locking.

**Analytic RR:**  
For a uniform Poisson process in a periodic box of side L with N galaxies:

```
RR(r) = N(N−1)/2 · (4π/3)(r_hi³ − r_lo³) / L³
```

No random catalogue is needed.

**Correction weights (Hickman et al. 2026):**  
With m sub-volumes selected from k:

| Weight | Formula | Role |
|--------|---------|------|
| α | m/k | Down-weights inflated one-halo auto pairs |
| β | m(k−1)/[k(m−1)] | Adjusts two-halo cross pairs |

Setting m = k gives α = β = 1, recovering the standard full-catalogue result.

---

**What drives the pattern:**

- **Small r_max (≤ 2 Mpc/h) — cells capped at 128.** The cell list is bounded
  at 128³ ≈ 2 M buckets to keep memory under 50 MB regardless of r_max. With
  cells much larger than r_max, most candidate pairs are rejected immediately
  by the distance check, but the 27-cell search still pays a fixed overhead.
  SCOPE's Rayon parallelism absorbs this cost; Corrfunc pays similar
  fixed overhead but without the parallel fold, so SCOPE wins 2–3×.

- **Mid-range (5–20 Mpc/h) — cell list well-matched.** The cell count drops
  from 128 to 25, cell size ≈ r_max, and each particle inspects only its
  natural 27-cell neighbourhood. SCOPE's thread-local accumulators avoid all
  synchronisation; it wins 3–5× over single-thread Corrfunc and also beats
  16-thread Corrfunc at r_max ≤ 20 Mpc/h.

- **Large r_max (≥ 50 Mpc/h) — cell list degenerates.** At r_max ≥ box/3
  the grid collapses to 3 cells per dimension and the loop is effectively
  O(N²). Corrfunc's SIMD/cache-blocking advantages are most significant here,
  and 16-thread Corrfunc pulls ahead. In practice, ξ(r) at r > 50 Mpc/h
  can be computed with a single call using the full r-range, so this regime
  is not the bottleneck for typical GALFORM analyses.

---

## Redshift-space distortions: ξ(s, μ)

`compute_xi_smu` computes the anisotropic correlation function in redshift-space
separation s and cosine angle to the line-of-sight μ = |Δz|/s, then projects
onto the monopole ξ₀(s) and quadrupole ξ₂(s).

The caller must pre-apply the RSD displacement before passing positions:

```python
# z_rsd = (z_real + v_pec_z / H) % box_size   (all in Mpc/h)
coords_rsd = coords.copy()
coords_rsd[:, 2] = (coords[:, 2] + v_pec_z / H) % box_size

from scope import compute_xi_smu

result = compute_xi_smu(
    coords              = coords_rsd,   # redshift-space [x, y, z_rsd]
    subvol_ids          = subvol_ids,
    s_bins              = s_bins,       # (n_s+1,) separation edges in Mpc/h
    box_size            = 542.16,
    n_subvols           = 1024,
    n_subvols_selected  = 8,
    n_mu_bins           = 100,          # μ bins in [0, mu_max]
    mu_max              = 1.0,
)

xi_smu = result["xi_smu"]   # (n_s, n_mu)  ξ(s, μ)
xi0    = result["xi0"]      # (n_s,)       monopole ξ₀(s)
xi2    = result["xi2"]      # (n_s,)       quadrupole ξ₂(s)
s_mid  = result["s_mid"]    # (n_s,)       geometric-mean bin centres
mu_mid = result["mu_mid"]   # (n_mu,)      μ bin centres
```

The same α/β sub-volume correction weights are applied as for `compute_xi`.

---

## Legacy 2D interface

`compute_2pcf` and `count_pairs_2d` are retained for backward compatibility
and compute the anisotropic ξ(r_p, π) and projected w_p(r_p).
These use different (equivalent) correction weights scaled to the full-box
count and are not the primary interface.
