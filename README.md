# SCOPE — Sparse Clustering On Periodic-box Estimator

A high-performance hybrid Rust/Python package for computing two-point and N-point correlation functions from galaxy catalogues drawn from a sparse subset of sub-volumes of a periodic N-body simulation.

---

## Core Capabilities

- **Real-Space 2PCF:** Compute unbiased xi(r) from partial datasets.
- **Redshift-Space RSD:** Compute ultra-optimized xi(s, mu) multipoles.
- **N-point Functions:** High-order correlation counters (N=3, 4, ...) with recursive distance pruning.
- **Sub-volume Correction:** Automatically applies Hickman et al. (2026) weights to correct for statistical under-sampling.
- **Hardware-Aware Acceleration:** Automatic CPU/GPU selection for massive 50M+ galaxy datasets.

---

## Performance Breakthroughs (Phase 2)

The current version implements several advanced optimization techniques:

1.  **Monotonic Z-Range Tracking:** Amortized O(1) candidate search for line-of-sight binning.
2.  **Constant-Time Binning (O(1)):** High-resolution lookup tables eliminate logarithmic search overhead.
3.  **SoA Linear Access:** Galaxy data reordered for contiguous cache-line reads and SIMD throughput.
4.  **Hybrid CPU/GPU Selector:** Intelligently routes N=2, 3 tasks to GPU while utilizing CPU branch-prediction for N >= 4 recursive pruning.
