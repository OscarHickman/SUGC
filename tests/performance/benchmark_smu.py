"""
SCOPE vs Corrfunc — (s, μ) redshift-space pair counting benchmark.

SCOPE counts pairs split by sub-volume ID (auto + cross separately).
Corrfunc DDsmu counts all pairs in one pass with SIMD optimisation.

As with the 1D benchmark, the comparison is not apples-to-apples: SCOPE does
strictly more work (routing each pair to one of two accumulators, maintaining
sub-volume membership), while Corrfunc performs a single-accumulator count.
The intent is to show where SCOPE sits relative to a state-of-the-art reference
at a representative RSD analysis scale (s_max = 40 Mpc/h, 100 μ bins).
"""

import time
import os
import numpy as np
from Corrfunc.theory import DDsmu
from scope._scope import count_pairs_smu

# ── Fixed parameters ──────────────────────────────────────────────────────────
RNG_SEED  = 42
BOX_SIZE  = 542.16      # P-Millennium Mpc/h
N_SUBVOLS = 27
N_SUBVOLS_SELECTED = 9  # m/k = 1/3

S_MAX     = 40.0        # Mpc/h  — typical RSD analysis scale
N_S_BINS  = 20          # log-spaced s bins from 0.1 to S_MAX
N_MU_BINS = 100         # uniform μ bins in [0, 1]
MU_MAX    = 1.0

S_BINS = np.logspace(np.log10(0.1), np.log10(S_MAX), N_S_BINS + 1)

N_PARTICLES = [10_000, 30_000, 100_000, 300_000]
REPEATS = 3

N_THREADS_CF = min(16, max(1, len(os.sched_getaffinity(0))))


def make_catalogue(n, rng):
    coords = rng.uniform(0, BOX_SIZE, size=(n, 3)).astype(np.float64, order="C")
    sv_ids = rng.integers(0, N_SUBVOLS, size=n).astype(np.int32)
    mask = sv_ids < N_SUBVOLS_SELECTED
    return coords[mask], sv_ids[mask]


def time_scope(coords, sv_ids):
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        count_pairs_smu(coords, sv_ids, S_BINS, N_MU_BINS, MU_MAX, BOX_SIZE)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def time_corrfunc(coords, nthreads=1):
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


def main():
    print("=" * 75)
    print("  SCOPE vs Corrfunc — (s, μ) redshift-space pair counting benchmark")
    print(f"  P-Millennium box={BOX_SIZE} Mpc/h  ·  {N_SUBVOLS_SELECTED}/{N_SUBVOLS} sub-vols")
    print(f"  s_max={S_MAX} Mpc/h  ·  {N_S_BINS} log s bins  ·  {N_MU_BINS} μ bins")
    print(f"  Corrfunc multi-thread uses {N_THREADS_CF} threads  ·  SCOPE uses Rayon default")
    print(f"  Median of {REPEATS} runs")
    print("=" * 75)
    print(
        f"  {'N input':>8}  {'N actual':>8}  "
        f"{'SCOPE':>8}  {'CF 1t':>8}  {'ratio':>6}  "
        f"{'CF Nt':>8}  {'ratio':>6}"
    )
    print(
        f"  {'(req)':>8}  {'(sel)':>8}  "
        f"{'[ms]':>8}  {'[ms]':>8}  {'':>6}  "
        f"{'[ms]':>8}  {'':>6}"
    )
    print("-" * 75)

    rng = np.random.default_rng(RNG_SEED)

    for n_req in N_PARTICLES:
        coords, sv_ids = make_catalogue(n_req, rng)
        n_actual = len(coords)

        t_scope = time_scope(coords, sv_ids)
        t_cf_1t = time_corrfunc(coords, nthreads=1)
        t_cf_nt = time_corrfunc(coords, nthreads=N_THREADS_CF)

        r1 = t_scope / t_cf_1t
        rN = t_scope / t_cf_nt
        print(
            f"  {n_req:>8,}  {n_actual:>8,}  "
            f"{t_scope*1e3:>8.1f}  {t_cf_1t*1e3:>8.1f}  {r1:>6.2f}x  "
            f"{t_cf_nt*1e3:>8.1f}  {rN:>6.2f}x"
        )

    print("=" * 75)
    print()
    print("Notes:")
    print("  SCOPE routes each pair to dd_auto or dd_cross — strictly more work than Corrfunc.")
    print("  Corrfunc uses AVX/AVX2 SIMD, tile-based cache blocking, and optional OpenMP.")
    print("  ratio > 1 means SCOPE is slower; < 1 means SCOPE is faster.")


if __name__ == "__main__":
    main()
