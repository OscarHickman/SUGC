# GEMINI.md — SCOPE Project Context

## Project Overview
**SCOPE** (Sparse Clustering On Periodic-box Estimator) is a hybrid Rust/Python package designed to compute two-point correlation functions (2PCF) from galaxy catalogues drawn from a sparse subset of sub-volumes of a periodic N-body simulation (e.g., GALFORM on P-Millennium).

### The Problem It Solves
When using only $m$ out of $k$ independent realisations (sub-volumes), naive pair counting biases the small-scale (one-halo) signal. SCOPE applies scalar correction weights ($ \alpha, \beta $) to intra- and inter-sub-volume pair counts to provide an unbiased estimate of the full-density 2PCF without needing the full dataset.

### Core Technologies
- **Rust (Backend):** Performance-critical pair counting using `Rayon` for parallelism and `PyO3` for Python bindings.
- **Python (Frontend):** High-level API, analytic Random-Random (RR) calculations, and multipole projections.
- **Maturin:** Build system for compiling and installing the Rust extension as a Python module.
- **Pytest:** Unit testing framework.

---

## Building and Running

### Build & Installation
All commands should be run from the repository root.

- **Local Development (Editable):**
  ```bash
  maturin develop --release
  ```
- **Build Wheel:**
  ```bash
  maturin build --release
  ```
- **Install Wheel:**
  ```bash
  pip install target/wheels/scope-*.whl --user
  ```

### HPC Environment (COSMA)
On Lustre file systems, suppress hardlink warnings by setting:
```bash
export UV_LINK_MODE=copy
```

### Running Tests
Run the unit test suite to verify physics acceptance:
```bash
pytest tests/unit/ -v
```
**Physics Acceptance:** $|\xi| < 0.05$ in every bin with RR > 100 on a uniform Poisson field.

---

## Project Architecture

### Data Flow
1. **Python Layer (`python/scope/__init__.py`):** Calculates correction weights ($\alpha, \beta$) and invokes Rust counters.
2. **Rust Layer (`src/lib.rs`):** Orchestrates calls to specific counters.
3. **Core Logic (`src/pairs_1d.rs`, `src/pairs_smu.rs`):**
   - Builds an isotropic cell list (`src/cell_list.rs`).
   - Counts pairs in parallel using `Rayon` (thread-local fold/reduce).
   - Separates counts into `dd_auto` (same sub-volume) and `dd_cross` (different sub-volumes).
4. **Python Layer (Post-processing):** Applies weights, computes analytic RR, and returns results (e.g., $\xi(r)$, $\xi_0(s)$, $\xi_2(s)$).

### Key Components
- **Cell List (`src/cell_list.rs`):** Isotropic grid with a minimum of 3 cells per dimension for periodic wrapping. Max 128 cells/dim to bound memory.
- **Analytic RR:** No Monte Carlo random catalogues are used. RR is computed analytically for periodic boxes.
- **Sub-volume IDs:** These label independent statistical realisations, *not* spatial cells. They span the full simulation box.

---

## Development Conventions

- **Performance:** All pair-counting loops must be implemented in Rust.
- **Parallelism:** Use `Rayon` for Rust loops. Prefer thread-local accumulators over locking.
- **API Consistency:** High-level functions like `compute_xi` should return a dictionary containing raw counts (`dd_auto`, `dd_cross`), corrected counts (`dd_corr`), and the final statistic.
- **Memory Safety:** Avoid unnecessary clones in Rust; use `numpy` arrays via `PyO3` to share memory between Python and Rust.
- **Documentation:** Use NumPy-style docstrings for Python functions.
- **Testing:** New features must include unit tests in `tests/unit/` using the Poisson field validation method.
