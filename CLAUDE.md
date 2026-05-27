# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

**SUGC** — **Sparse Correction Of Pair Estimators** — computes multi-point correlation functions for galaxy catalogues drawn from **m of k independent realisations** of a periodic simulation box.

### Key Capabilities:
- **Optimized Counters:** Phase 2 Rust backend with Z-sorting, SoA reordering, and constant-time binning.
- **Statistics:** xi(r), xi(s, mu) multipoles, and high-order NPCFs (N=3, 4, ...).
- **Hybrid Backend:** Hardware-aware selector for CPU/GPU execution.
- **Compatibility:** Fully ABI3 compatible (Python 3.9+).
