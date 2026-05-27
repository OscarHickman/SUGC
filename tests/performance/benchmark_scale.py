"""
SUGC vs Corrfunc — performance vs r_max sweep.

P-Millennium box (542.16 Mpc/h).  Sweeps r_max from 0.01 Mpc/h (10 kpc/h) up to
256 Mpc/h (box/2, the physical maximum for periodic minimum-image).

At small r_max the cell list is fine-grained → O(N) fast.
At large r_max cells degenerate → O(N²) at r_max = box/2.

Fixed N=100,000 throughout so timing differences are purely due to r_max.
"""

import time
import os
import numpy as np
from Corrfunc.theory import DD
from sugc._sugc import count_pairs_1d

BOX_SIZE = 542.16       # P-Millennium Mpc/h
N        = 100_000
REPEATS  = 3
N_BINS   = 10          # 10 log r bins per decade (consistent across calls)

N_THREADS_CF = min(16, max(1, len(os.sched_getaffinity(0))))

# r_max values spanning sub-Mpc to box/2
# Corrfunc requires rmax < box/2 strictly; 255.0 is the practical maximum.
R_MAX_VALUES = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 255.0]

K_REALISATIONS = 27   # k independent realisations

rng    = np.random.default_rng(42)
coords = rng.uniform(0, BOX_SIZE, (N, 3)).astype(np.float64, order="C")
# Realisation IDs are independent of position — each realisation spans the full box.
sv_ids = rng.integers(0, K_REALISATIONS, size=N).astype(np.int32)


def timeit(fn, reps=REPEATS):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def n_cells(r_max):
    """Number of cells SUGC would build (after the 128 cap)."""
    return min(128, max(3, int(BOX_SIZE / r_max)))


print("=" * 80)
print("  SUGC vs Corrfunc — timing vs r_max  (P-Millennium, box=542.16 Mpc/h)")
print(f"  N={N:,}  {N_BINS} log r bins per decade")
print(f"  Corrfunc: {N_THREADS_CF} threads    SUGC: Rayon (all available threads)")
print(f"  Median of {REPEATS} runs")
print("=" * 80)
print(f"  {'r_max':>8}  {'n_cells':>8}  "
      f"{'SUGC':>8}  {'CF 1t':>8}  {'CF Nt':>8}  {'S/CF-1t':>8}  {'S/CF-Nt':>8}")
print(f"  {'(Mpc/h)':>8}  {'':>8}  "
      f"{'(ms)':>8}  {'(ms)':>8}  {'(ms)':>8}  {'':>8}  {'':>8}")
print("-" * 80)

for r_max in R_MAX_VALUES:
    r_min_bin = max(1e-4, r_max / 10**N_BINS)
    r_bins = np.logspace(np.log10(r_min_bin), np.log10(r_max), N_BINS + 1)
    nc = n_cells(r_max)

    t_sugc = timeit(lambda rb=r_bins:
                     count_pairs_1d(coords, sv_ids, rb, BOX_SIZE))
    t_cf_1t = timeit(lambda rb=r_bins:
                     DD(1, 1, rb,
                        coords[:,0], coords[:,1], coords[:,2],
                        periodic=True, boxsize=BOX_SIZE,
                        output_ravg=False, verbose=False))
    t_cf_nt = timeit(lambda rb=r_bins:
                     DD(1, N_THREADS_CF, rb,
                        coords[:,0], coords[:,1], coords[:,2],
                        periodic=True, boxsize=BOX_SIZE,
                        output_ravg=False, verbose=False))

    ratio_1t = t_sugc / t_cf_1t
    ratio_nt = t_sugc / t_cf_nt
    print(f"  {r_max:>8.3g}  {nc:>8d}  "
          f"{t_sugc*1e3:>8.1f}  {t_cf_1t*1e3:>8.1f}  {t_cf_nt*1e3:>8.1f}  "
          f"{ratio_1t:>8.2f}x  {ratio_nt:>8.2f}x")

print("=" * 80)
print()
print(f"Note: 'n_cells' = min(128, floor(box/r_max)) — cell count per dimension.")
print(f"      At n_cells=3 (r_max ≥ box/3 ≈ 171 Mpc/h) the cell list gives no gain.")
print(f"      box/2 = {BOX_SIZE/2:.1f} Mpc/h is the max physical separation (min-image).")
