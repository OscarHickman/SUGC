"""
Benchmark SUGC vs Corrfunc for 3D real-space pair counting.

SUGC counts pairs split by sub-volume ID (auto + cross separately).
Corrfunc counts all pairs in one pass with SIMD optimisation.

The timings are not directly comparing identical work — SUGC does strictly more
(it routes each pair to one of two accumulators), but the cell-list structure and
O(N) scaling approach are the same, so this shows where SUGC sits relative to a
state-of-the-art reference implementation.
"""

import time
import os
import numpy as np
from Corrfunc.theory import DD
from sugc._sugc import count_pairs_1d

# ── Fixed parameters ──────────────────────────────────────────────────────────
RNG_SEED  = 42
BOX_SIZE  = 542.16      # P-Millennium Mpc/h
N_SUBVOLS = 27         # k independent realisations
N_SUBVOLS_SELECTED = 9 # m realisations selected (m/k = 1/3)

# 30 log-spaced bins from 0.01 to 256 Mpc/h  (covers kpc/h to box/2)
R_BINS = np.logspace(np.log10(0.01), np.log10(256.0), 31)

# N values to sweep
N_PARTICLES = [10_000, 30_000, 100_000, 300_000]

REPEATS = 3

N_THREADS_CF = min(16, max(1, len(os.sched_getaffinity(0))))


def make_catalogue(n, rng):
    """Uniform random positions with realisation IDs assigned randomly.

    Each galaxy is drawn from one of N_SUBVOLS independent realisations.
    Realisation IDs are independent of position — every realisation spans
    the full box volume.
    """
    coords = rng.uniform(0, BOX_SIZE, size=(n, 3)).astype(np.float64, order="C")
    sv_ids = rng.integers(0, N_SUBVOLS, size=n).astype(np.int32)
    mask = sv_ids < N_SUBVOLS_SELECTED
    return coords[mask], sv_ids[mask]


def time_sugc(coords, sv_ids):
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        count_pairs_1d(coords, sv_ids, R_BINS, BOX_SIZE)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def time_corrfunc(coords, nthreads=1):
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


def main():
    print("=" * 75)
    print("  SUGC vs Corrfunc — 3D real-space pair counting benchmark")
    print(f"  P-Millennium box={BOX_SIZE} Mpc/h  ·  {N_SUBVOLS_SELECTED}/{N_SUBVOLS} sub-vols")
    print(f"  {len(R_BINS)-1} log bins  [{R_BINS[0]:.3f}, {R_BINS[-1]:.1f}] Mpc/h")
    print(f"  Corrfunc multi-thread uses {N_THREADS_CF} threads  ·  SUGC uses Rayon default")
    print(f"  Median of {REPEATS} runs")
    print("=" * 75)
    print(
        f"  {'N input':>8}  {'N actual':>8}  "
        f"{'SUGC':>8}  {'CF 1t':>8}  {'ratio':>6}  "
        f"{'CF Nt':>8}  {'ratio':>6}"
    )
    print(f"  {'(req)':>8}  {'(sel)':>8}  "
          f"{'[ms]':>8}  {'[ms]':>8}  {'':>6}  "
          f"{'[ms]':>8}  {'':>6}")
    print("-" * 75)

    rng = np.random.default_rng(RNG_SEED)

    for n_req in N_PARTICLES:
        coords, sv_ids = make_catalogue(n_req, rng)
        n_actual = len(coords)

        t_sugc  = time_sugc(coords, sv_ids)
        t_cf_1t  = time_corrfunc(coords, nthreads=1)
        t_cf_nt  = time_corrfunc(coords, nthreads=N_THREADS_CF)

        r1 = t_sugc / t_cf_1t
        rN = t_sugc / t_cf_nt
        print(
            f"  {n_req:>8,}  {n_actual:>8,}  "
            f"{t_sugc*1e3:>8.1f}  {t_cf_1t*1e3:>8.1f}  {r1:>6.2f}x  "
            f"{t_cf_nt*1e3:>8.1f}  {rN:>6.2f}x"
        )

    print("=" * 75)
    print()
    print("Notes:")
    print("  SUGC routes each pair to dd_auto or dd_cross — strictly more work than Corrfunc.")
    print("  Corrfunc uses AVX/AVX2 SIMD, tile-based cache blocking, and optional OpenMP.")
    print("  ratio > 1 means SUGC is slower; < 1 means SUGC is faster.")


if __name__ == "__main__":
    main()
