# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

SCOPE computes the anisotropic two-point correlation function ξ(r_p, π) and the projected correlation function w_p(r_p) for galaxy catalogues drawn from a **subset of sub-volumes** of a periodic N-body simulation box (e.g. GALFORM on P-Millennium). The core problem it solves: using only m of k sub-volumes biases the pair counts — SCOPE corrects this with scalar weights α = k/m (auto pairs) and β = k(k−1)/[m(m−1)] (cross pairs).

The package is a Rust extension module compiled with **maturin** and exposed to Python via **PyO3**.

## Repository layout

The root `/SCOPE` and the subdirectory `/SCOPE/scope` are **mirrors** of the same package at different stages of development. The canonical, active source is in `scope/` (the subdirectory). Working files:

```
scope/
├── Cargo.toml          # Rust deps: pyo3, numpy, rayon
├── pyproject.toml      # maturin build backend, module-name = scope._scope
├── src/lib.rs          # ALL Rust: cell-list builder + count_pairs_2d PyO3 fn
└── python/scope/
    └── __init__.py     # ALL Python: analytic_rr, compute_2pcf (the public API)
```

## Build commands

All commands should be run from `scope/` (the subdirectory):

```bash
# Development build (fast iteration — recompiles Rust, installs into active venv)
maturin develop --release

# Production wheel
maturin build --release
# wheel lands at: target/wheels/scope-*.whl

# Install the built wheel
pip install target/wheels/scope-*.whl --user
# or with uv:
uv pip install target/wheels/scope-*.whl
```

On Cosma (Lustre network FS), suppress a harmless hardlink warning:
```bash
export UV_LINK_MODE=copy
```

There are no lint or test commands defined — verification is done by running the package against a uniform random field and checking that mean |ξ| < 0.02.

## Architecture

**Data flow:**

1. User calls `compute_2pcf(coords, subvol_ids, r_p_bins, pi_bins, box_size, n_subvols, n_subvols_selected)` in `__init__.py`
2. Python computes α and β correction weights, then calls the Rust function:
3. `count_pairs_2d` (Rust, `src/lib.rs`) builds an anisotropic cell list and counts pairs in parallel via Rayon, routing each pair to `dd_auto` (same subvol ID) or `dd_cross` (different subvol IDs). Uses a thread-local fold/reduce pattern — no locks.
4. Python applies `dd_corr = α·dd_auto + β·dd_cross`, computes analytic RR, returns ξ and w_p.

**Cell list:** Separate granularity for transverse (r_p) and LOS (π) directions. Each particle searches its 3×3×3 = 27 neighbouring cells. Periodic BC via minimum-image convention. Each pair counted once (i < j).

**Analytic RR:** `RR = N(N-1)/2 · π(r_p_hi²−r_p_lo²) · 2Δπ / L³` — no Monte Carlo random catalogue needed for a periodic box.

## Key physics constraint

Sub-volume IDs must label **3D spatial cells** (e.g. a regular nx×ny×nz grid tiling the box), never z-slices. Using z-slices correlates the auto/cross split with the π (LOS) axis and biases ξ at small π. The correction weights assume all sub-volumes are statistically equivalent.

## CI / release

GitHub Actions (`CI.yml`) uses `maturin-action` to build wheels for Linux (glibc + musl), Windows, and macOS on push to main/tags. Releases publish to PyPI via `uv publish` when a tag is pushed.