# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

**SCOPE** — **Sparse Correction Of Pair Estimators** — computes the real-space two-point correlation function ξ(r) for galaxy catalogues drawn from **m of k independent realisations** of a periodic N-body simulation box (e.g. GALFORM on P-Millennium, 512 Mpc/h). Only real-space (isotropic) ξ(r) is supported — no redshift-space distortions, no (r_p, π) decomposition.

Each **sub-volume** (realisation) is an independent statistical sample that spans the **full simulation box** at 1/k of the total number density — sub-volume IDs are not spatial cell labels. Galaxies from any given realisation are distributed across the entire box. Stacking all k realisations recovers the full-density catalogue.

The core problem: using only m of k realisations biases the pair counts. SCOPE corrects this with scalar weights (Hickman et al. 2026 Eqs. 9–10):
- α = m/k  — down-weights same-realisation (intra-subvol) auto pairs
- β = m(k−1)/[k(m−1)]  — adjusts cross-realisation (inter-subvol) cross pairs
- Target: ⟨DD_corr⟩ = (m/k)²·(DD_1h + DD_2h), normalised to the selected catalogue.
- Minimum useful m is 2 (m=1 has no cross-pairs so intra/inter cannot be separated).

The package is a Rust extension module compiled with **maturin** and exposed to Python via **PyO3**.

## Repository layout

The repository root is the package root — there is no subdirectory mirror. Working files:

```
SCOPE/                  ← repository root
├── Cargo.toml          # Rust deps: pyo3, numpy, rayon
├── pyproject.toml      # maturin build backend, module-name = scope._scope
├── src/lib.rs          # ALL Rust: cell-list builder + count_pairs_1d + count_pairs_2d
├── python/scope/
│   └── __init__.py     # ALL Python: analytic_rr_1d, compute_xi (primary API)
│                       #             analytic_rr, compute_2pcf  (legacy 2D, kept for compat)
├── benchmark_corrfunc.py  # SCOPE vs Corrfunc DD timing at fixed r_max range
└── benchmark_scale.py     # SCOPE vs Corrfunc DD timing vs r_max sweep
```

## Build commands

All commands run from the **repository root** (`/SCOPE/`):

```bash
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

There are no lint or test commands defined — verification is done by running the package against a uniform random field and checking that mean |ξ| < 0.02 for bins with RR > 100 (bins at r < ~0.5 Mpc/h with N~200k will have too few pairs to meet the threshold and are noise-dominated).

## Architecture

**Data flow:**

1. User calls `compute_xi(coords, subvol_ids, r_bins, box_size, n_subvols, n_subvols_selected)` in `__init__.py`
2. Python computes α and β correction weights, then calls the Rust function:
3. `count_pairs_1d` (Rust, `src/lib.rs`) builds an isotropic cell list and counts pairs in parallel via Rayon, routing each pair to `dd_auto` (same subvol ID) or `dd_cross` (different subvol IDs). Uses a thread-local fold/reduce pattern — no locks.
4. Python applies `dd_corr = α·dd_auto + β·dd_cross`, computes analytic RR, returns ξ(r).

**Cell list:** Isotropic — same cell size in all 3 dimensions, set to r_max. Each particle searches its 3×3×3 = 27 neighbouring cells. Minimum 3 cells per dimension to avoid double-counting under periodic wrapping. Periodic BC via minimum-image convention. Each pair counted once (i < j).

**Analytic RR:** `RR = N(N-1)/2 · (4π/3)(r_hi³−r_lo³) / L³` — no Monte Carlo random catalogue needed for a periodic box.

## Key physics constraints

Sub-volume IDs label **independent statistical realisations**, not spatial cells. Each realisation spans the full box; the ID assigned to a galaxy carries no positional meaning. The Rust pair counter enforces this — it performs a plain integer equality check (`sv_id[i] == sv_id[j]`) with no spatial interpretation.

The correction weights assume all realisations are statistically equivalent (same number density, same clustering). The selected m realisations should be drawn randomly from the k available ones. For convergence tests, selecting realisations in a fixed deterministic order (each step a superset of the previous) is acceptable.

## CI / release

GitHub Actions (`CI.yml`) uses `maturin-action` to build wheels for Linux (glibc + musl), Windows, and macOS on push to main/tags. Releases publish to PyPI via `uv publish` when a tag is pushed.