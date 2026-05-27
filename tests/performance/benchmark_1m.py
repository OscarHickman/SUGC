"""
SUGC vs Corrfunc — Benchmark up to 1 million actual particles.

This script runs benchmarks for:
1. 3D Real-space pair counting (count_pairs_1d vs DD)
2. (s, mu) Redshift-space pair counting (count_pairs_smu vs DDsmu)

For each, we scale up to N_req = 3,000,000 (which selects ~1,000,000 actual particles).
SUGC runs with Rayon (using all available threads).
Corrfunc runs with 1 thread and N threads (typically 16 threads).
"""

import os
import time

import numpy as np
from Corrfunc.theory import DD, DDsmu
from sugc._sugc import count_pairs_1d, count_pairs_smu

# ── Parameters ────────────────────────────────────────────────────────────────
RNG_SEED  = 42
BOX_SIZE  = 542.16      # P-Millennium Mpc/h
N_PARTITIONS = 27
N_PARTITIONS_SELECTED = 9  # m/k = 1/3 (fraction of particles selected)

# Redshift-space parameters
S_MAX     = 40.0
N_S_BINS  = 20
N_MU_BINS = 100
MU_MAX    = 1.0
S_BINS    = np.logspace(np.log10(0.1), np.log10(S_MAX), N_S_BINS + 1)

# Real-space parameters
R_BINS = np.logspace(np.log10(0.01), np.log10(256.0), 31)

# Particle counts to sweep (up to 3M requested -> ~1M selected)
N_PARTICLES = [10_000, 100_000, 1_000_000, 3_000_000]
REPEATS = 3

# Determine thread count
N_THREADS_CF = int(os.environ.get("OMP_NUM_THREADS", 16))

def make_catalogue(n, rng):
    coords = rng.uniform(0, BOX_SIZE, size=(n, 3)).astype(np.float64, order="C")
    part_ids = rng.integers(0, N_PARTITIONS, size=n).astype(np.int32)
    mask = part_ids < N_PARTITIONS_SELECTED
    return coords[mask], part_ids[mask]

def time_sugc_1d(coords, part_ids):
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        count_pairs_1d(coords, part_ids, R_BINS, BOX_SIZE)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))

def time_corrfunc_1d(coords, nthreads):
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        DD(
            autocorr=1,
            nthreads=nthreads,
            binfile=R_BINS,
            X1=x, Y1=y, Z1=z,
            periodic=True,
            boxsize=BOX_SIZE,
            output_ravg=False,
            verbose=False,
        )
        times.append(time.perf_counter() - t0)
    return float(np.median(times))

def time_sugc_smu(coords, part_ids):
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        count_pairs_smu(coords, part_ids, S_BINS, N_MU_BINS, MU_MAX, BOX_SIZE)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))

def time_corrfunc_smu(coords, nthreads):
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        DDsmu(
            autocorr=1,
            nthreads=nthreads,
            binfile=S_BINS,
            mu_max=MU_MAX,
            nmu_bins=N_MU_BINS,
            X1=x, Y1=y, Z1=z,
            periodic=True,
            boxsize=BOX_SIZE,
            output_savg=False,
            verbose=False,
        )
        times.append(time.perf_counter() - t0)
    return float(np.median(times))

def run_benchmark():
    print("=" * 85)
    print("  SUGC vs Corrfunc High-Scale Benchmark (Up to 1 Million Selected Particles)")
    print(f"  P-Millennium box={BOX_SIZE} Mpc/h  ·  {N_PARTITIONS_SELECTED}/{N_PARTITIONS} partitions")
    print(f"  Corrfunc multi-thread uses {N_THREADS_CF} threads  ·  SUGC uses Rayon (all cores)")
    print(f"  Median of {REPEATS} runs")
    print("=" * 85)

    # 1. Real-space 1D Benchmark
    print("\n--- 1. 3D Real-Space Pair Counting (1D r-bins) ---")
    print(f"    Bins: {len(R_BINS)-1} log bins [{R_BINS[0]:.2f}, {R_BINS[-1]:.1f}] Mpc/h")
    print("-" * 85)
    print(
        f"  {'N input':>10}  {'N actual':>10}  "
        f"{'SUGC':>10}  {'CF 1t':>10}  {'ratio':>8}  "
        f"{'CF Nt':>10}  {'ratio':>8}"
    )
    print(
        f"  {'(req)':>10}  {'(sel)':>10}  "
        f"{'[ms]':>10}  {'[ms]':>10}  {'':>8}  "
        f"{'[ms]':>10}  {'':>8}"
    )
    print("-" * 85)

    rng = np.random.default_rng(RNG_SEED)
    results_1d = []

    for n_req in N_PARTICLES:
        coords, part_ids = make_catalogue(n_req, rng)
        n_actual = len(coords)

        t_sugc = time_sugc_1d(coords, part_ids)
        t_cf_1t = time_corrfunc_1d(coords, nthreads=1)
        t_cf_nt = time_corrfunc_1d(coords, nthreads=N_THREADS_CF)

        r1 = t_sugc / t_cf_1t
        rN = t_sugc / t_cf_nt
        results_1d.append((n_req, n_actual, t_sugc, t_cf_1t, t_cf_nt))

        print(
            f"  {n_req:>10,}  {n_actual:>10,}  "
            f"{t_sugc*1e3:>10.1f}  {t_cf_1t*1e3:>10.1f}  {r1:>8.2f}x  "
            f"{t_cf_nt*1e3:>10.1f}  {rN:>8.2f}x"
        )

    # 2. Redshift-space smu Benchmark
    print("\n--- 2. Redshift-Space Pair Counting (s, mu bins) ---")
    print(f"    Bins: s_max={S_MAX} Mpc/h ({N_S_BINS} log s bins) · {N_MU_BINS} mu bins")
    print("-" * 85)
    print(
        f"  {'N input':>10}  {'N actual':>10}  "
        f"{'SUGC':>10}  {'CF 1t':>10}  {'ratio':>8}  "
        f"{'CF Nt':>10}  {'ratio':>8}"
    )
    print(
        f"  {'(req)':>10}  {'(sel)':>10}  "
        f"{'[ms]':>10}  {'[ms]':>10}  {'':>8}  "
        f"{'[ms]':>10}  {'':>8}"
    )
    print("-" * 85)

    rng = np.random.default_rng(RNG_SEED)
    results_smu = []

    for n_req in N_PARTICLES:
        coords, part_ids = make_catalogue(n_req, rng)
        n_actual = len(coords)

        t_sugc = time_sugc_smu(coords, part_ids)
        t_cf_1t = time_corrfunc_smu(coords, nthreads=1)
        t_cf_nt = time_corrfunc_smu(coords, nthreads=N_THREADS_CF)

        r1 = t_sugc / t_cf_1t
        rN = t_sugc / t_cf_nt
        results_smu.append((n_req, n_actual, t_sugc, t_cf_1t, t_cf_nt))

        print(
            f"  {n_req:>10,}  {n_actual:>10,}  "
            f"{t_sugc*1e3:>10.1f}  {t_cf_1t*1e3:>10.1f}  {r1:>8.2f}x  "
            f"{t_cf_nt*1e3:>10.1f}  {rN:>8.2f}x"
        )
    print("=" * 85)

if __name__ == "__main__":
    run_benchmark()
