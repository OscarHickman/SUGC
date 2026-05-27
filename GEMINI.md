# GEMINI.md — SUGC Project Context

## Project Overview
**SUGC** (Sparse Clustering On Periodic-box Estimator) is a hybrid Rust/Python package for unbiased galaxy clustering estimates from sparse sub-volumes.

### Phase 2 Breakthroughs:
- **N-point Capabilities:** Support for 3PCF, 4PCF, and beyond with optimized recursive pruning.
- **Ultra-Optimized RSD:** Monotonic Z-range tracking and constant-time binning lookup tables.
- **Hardware-Awareness:** Auto-detection of GPUs and scale-based routing.
- **Multi-Version Stability:** ABI3 compliance and multi-version Python CI (3.9 - 3.13).
