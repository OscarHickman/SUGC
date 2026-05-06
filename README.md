# SCOPE — Sparse Correction Of Pair Estimators

A Python package (Rust backend) for computing the real-space two-point
correlation function **ξ(r)** from galaxy catalogues drawn from a sparse
subset of sub-volumes of a periodic N-body simulation
(e.g. GALFORM on P-Millennium, 512 Mpc/h).

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
    box_size            = 512.0,   # P-Millennium, Mpc/h
    n_subvols           = 1024,    # k — total sub-volumes tiling the full box
    n_subvols_selected  = 8,       # m — sub-volumes actually loaded
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

Sub-volume IDs must label **3D spatial cells** — e.g. a regular nx×ny×nz grid
tiling the box. The correction weights assume all sub-volumes are statistically
equivalent (same mean density, same large-scale environment).

**Good:** a 3D grid cell index such as `ix*ny*nz + iy*nz + iz`.

**Avoid:** pure z-slices or any partition that correlates the label with the
separation axis being measured.

The selected m sub-volumes should be **spatially well-distributed**, not a
contiguous block (which biases ξ at scales comparable to the sub-volume size).
Minimum useful m is **2** — with m=1 there are no cross-pairs and the
one-halo/two-halo decomposition is undefined.

---

## Project layout

```
SCOPE/
├── Cargo.toml             # Rust deps: pyo3, numpy, rayon
├── pyproject.toml         # maturin build backend
├── src/lib.rs             # Rust: cell-list + count_pairs_1d + count_pairs_2d
├── python/scope/
│   └── __init__.py        # Python: compute_xi, analytic_rr_1d (primary API)
│                          #         compute_2pcf, analytic_rr  (legacy 2D)
├── benchmark_corrfunc.py  # SCOPE vs Corrfunc DD — timing at fixed range
└── benchmark_scale.py     # SCOPE vs Corrfunc DD — timing vs r_max sweep
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

## Performance

Measured on a Cosma node (16 cores), P-Millennium geometry (box = 512 Mpc/h),
N = 100,000 particles, 10 log-spaced r bins per decade. Median of 3 runs.
Corrfunc `DD` is the reference 3D isotropic pair counter (AVX2 SIMD + OpenMP).
**SCOPE does strictly more work** — it splits every pair into `DD_auto` or
`DD_cross` simultaneously — so the comparison slightly understates SCOPE's
raw pair-counting speed.

| r_max (Mpc/h) | cells | SCOPE (ms) | CF 1-thread (ms) | CF 16-thread (ms) | vs CF-1t | vs CF-16t |
|:---:|:---:|---:|---:|---:|---:|---:|
| 0.01 | 128 | 88 | 201 | 251 | 2.3× faster | 2.9× faster |
| 0.1  | 128 | 87 | 189 | 251 | 2.2× faster | 2.9× faster |
| 0.5  | 128 | 92 | 192 | 247 | 2.1× faster | 2.7× faster |
| 1    | 128 | 87 | 187 | 246 | 2.1× faster | 2.8× faster |
| 2    | 128 | 88 | 191 | 237 | 2.2× faster | 2.7× faster |
| 5    | 102 | 61 | 192 | 246 | 3.1× faster | 4.0× faster |
| 10   |  51 | 25 |  80 |  98 | 3.2× faster | 3.9× faster |
| 20   |  25 | 17 |  86 |  63 | 5.0× faster | 3.7× faster |
| 50   |  10 | 27 | 230 |  42 | 8.5× faster | 1.6× slower |
| 100  |   5 | 131 | 637 |  50 | 4.9× faster | 2.6× slower |
| 200  |   3 | 769 | 2943 | 189 | 3.8× faster | 4.1× slower |
| 255  |   3 | 975 | 5435 | 346 | 5.6× faster | 2.8× slower |

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

## Legacy 2D interface

`compute_2pcf` and `count_pairs_2d` are retained for backward compatibility
and compute the anisotropic ξ(r_p, π) and projected w_p(r_p).
These use different (equivalent) correction weights scaled to the full-box
count and are not the primary interface.
