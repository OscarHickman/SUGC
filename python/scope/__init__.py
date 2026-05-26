"""
SCOPE: Sparse Correction Of Pair Estimators (Hybrid CPU/GPU Dev).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scope._scope import (
    count_npoint,
    count_npoint_gpu,
    has_cuda,
    count_pairs_1d,
    count_pairs_2d,
    count_pairs_smu,
)

__all__ = [
    "count_pairs_1d",
    "count_pairs_2d",
    "count_pairs_smu",
    "count_npoint",
    "count_npoint_gpu",
    "has_cuda",
    "analytic_rr_1d",
    "analytic_rr",
    "analytic_rr_smu",
    "compute_xi",
    "compute_xi_smu",
    "compute_2pcf",
    "compute_npcf",
]

def compute_npcf(
    coords: NDArray[np.float64],
    subvol_ids: NDArray[np.int32],
    r_bins: NDArray[np.float64],
    box_size: float,
    n_subvols: int,
    n_subvols_selected: int,
    n_order: int,
    use_gpu: bool | None = None,
) -> dict[str, NDArray[np.float64]]:
    """
    N-point correlation function with Hardware-Aware Auto-Selection.
    """
    k = int(n_subvols)
    m = int(n_subvols_selected)
    if m < 1 or m > k:
        raise ValueError(f"n_subvols_selected={m} must be in [1, {k}]")

    coords = np.ascontiguousarray(coords, dtype=np.float64)
    subvol_ids = np.ascontiguousarray(subvol_ids, dtype=np.int32)
    r_bins = np.asarray(r_bins, dtype=np.float64)

    if use_gpu is None:
        n_gal = len(coords)
        if has_cuda() and n_gal > 10000 and n_order < 4:
            use_gpu = True
        else:
            use_gpu = False

    if use_gpu:
        try:
            t_by_s, t_total = count_npoint_gpu(coords, subvol_ids, r_bins, float(box_size), n_order)
        except Exception as e:
            print(f"Warning: GPU execution failed ({e}). Falling back to CPU.")
            t_by_s, t_total = count_npoint(coords, subvol_ids, r_bins, float(box_size), n_order)
    else:
        t_by_s, t_total = count_npoint(coords, subvol_ids, r_bins, float(box_size), n_order)

    weights = np.zeros(n_order)
    ratio = 1.0
    for s_idx in range(1, n_order + 1):
        ratio *= (m - s_idx + 1) / (k - s_idx + 1)
        if ratio != 0:
            weights[s_idx - 1] = (m / k) ** n_order / ratio
        else:
            weights[s_idx - 1] = 0.0

    t_corr = weights @ t_by_s
    r_mid = np.sqrt(r_bins[:-1] * r_bins[1:])

    return {
        "t_by_s": t_by_s,
        "t_total": t_total,
        "t_corr": t_corr,
        "weights": weights,
        "r_mid": r_mid,
        "used_gpu": use_gpu
    }

def analytic_rr_1d(
    r_bins: NDArray[np.float64],
    box_size: float,
    n_galaxies: int,
) -> NDArray[np.float64]:
    v_shell = (4.0 * np.pi / 3.0) * (r_bins[1:] ** 3 - r_bins[:-1] ** 3)
    prefactor = n_galaxies * (n_galaxies - 1) / (2.0 * box_size**3)
    return prefactor * v_shell

def analytic_rr(
    r_p_bins: NDArray[np.float64],
    pi_bins: NDArray[np.float64],
    box_size: float,
    n_total: int,
) -> NDArray[np.float64]:
    r_p_bins = np.asarray(r_p_bins, dtype=np.float64)
    pi_bins = np.asarray(pi_bins, dtype=np.float64)
    ann_areas = np.pi * (r_p_bins[1:] ** 2 - r_p_bins[:-1] ** 2)
    delta_pi = pi_bins[1:] - pi_bins[:-1]
    v_shell = 2.0 * ann_areas[:, np.newaxis] * delta_pi[np.newaxis, :]
    prefactor = n_total * (n_total - 1) / (2.0 * box_size**3)
    return prefactor * v_shell

def analytic_rr_smu(
    s_bins: NDArray[np.float64],
    mu_max: float,
    n_mu_bins: int,
    box_size: float,
    n_galaxies: int,
) -> NDArray[np.float64]:
    s_bins = np.asarray(s_bins, dtype=np.float64)
    v_shell = (4.0 * np.pi / 3.0) * (s_bins[1:] ** 3 - s_bins[:-1] ** 3)
    prefactor = n_galaxies * (n_galaxies - 1) / (2.0 * box_size**3)
    dmu = mu_max / n_mu_bins
    return (prefactor * v_shell)[:, np.newaxis] * np.full(n_mu_bins, dmu)

def compute_xi(coords, subvol_ids, r_bins, box_size, n_subvols, n_subvols_selected):
    res = compute_npcf(coords, subvol_ids, r_bins, box_size, n_subvols, n_subvols_selected, n_order=2)
    rr = analytic_rr_1d(r_bins, box_size, len(coords))
    xi = res["t_corr"] / rr - 1.0
    return {
        "xi": xi,
        "dd_auto": res["t_by_s"][0],
        "dd_cross": res["t_by_s"][1],
        "dd_corr": res["t_corr"],
        "rr": rr,
        "r_mid": res["r_mid"],
        "used_gpu": res["used_gpu"]
    }

def compute_xi_smu(
    coords: NDArray[np.float64],
    subvol_ids: NDArray[np.int32],
    s_bins: NDArray[np.float64],
    box_size: float,
    n_subvols: int,
    n_subvols_selected: int,
    n_mu_bins: int = 100,
    mu_max: float = 1.0,
) -> dict[str, NDArray[np.float64]]:
    k = int(n_subvols)
    m = int(n_subvols_selected)
    if m < 1 or m > k:
        raise ValueError(f"n_subvols_selected={m} must be in [1, {k}]")
    n_selected = len(coords)
    dd_auto, dd_cross = count_pairs_smu(coords, subvol_ids, s_bins, n_mu_bins, mu_max, float(box_size))
    alpha = m / k
    beta  = m * (k - 1) / (k * (m - 1)) if m > 1 else 0.0
    dd_corr: NDArray[np.float64] = alpha * dd_auto + beta * dd_cross
    rr = analytic_rr_smu(s_bins, mu_max, n_mu_bins, float(box_size), n_selected)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi_smu: NDArray[np.float64] = dd_corr / rr - 1.0
    xi_smu[~np.isfinite(xi_smu)] = np.nan
    dmu    = mu_max / n_mu_bins
    mu_mid = np.linspace(dmu / 2.0, mu_max - dmu / 2.0, n_mu_bins)
    L0 = np.ones(n_mu_bins)
    L2 = 0.5 * (3.0 * mu_mid**2 - 1.0)
    xi0: NDArray[np.float64] = np.nansum(xi_smu * L0[np.newaxis, :] * dmu, axis=1)
    xi2: NDArray[np.float64] = 5.0 * np.nansum(xi_smu * L2[np.newaxis, :] * dmu, axis=1)
    s_mid: NDArray[np.float64] = np.sqrt(s_bins[:-1] * s_bins[1:])
    return {
        "xi_smu":    xi_smu,
        "xi0":       xi0,
        "xi2":       xi2,
        "dd_auto":   dd_auto,
        "dd_cross":  dd_cross,
        "dd_corr":   dd_corr,
        "rr":        rr,
        "s_mid":     s_mid,
        "mu_mid":    mu_mid,
    }

def compute_2pcf(
    coords: NDArray[np.float64],
    subvol_ids: NDArray[np.int32],
    r_p_bins: NDArray[np.float64],
    pi_bins: NDArray[np.float64],
    box_size: float,
    n_subvols: int,
    n_subvols_selected: int,
    n_total: int | None = None,
) -> dict[str, NDArray[np.float64]]:
    k = int(n_subvols)
    m = int(n_subvols_selected)
    if m < 1 or m > k:
        raise ValueError(f"n_subvols_selected={m} must be in [1, {k}]")
    n_selected = len(coords)
    if n_total is None:
        n_total = int(round(n_selected * k / m))
    dd_auto, dd_cross = count_pairs_2d(coords, subvol_ids, r_p_bins, pi_bins, float(box_size))
    alpha, beta = k/m, k*(k-1)/(m*(m-1)) if m > 1 else 0.0
    dd_corr = alpha * dd_auto + beta * dd_cross
    ann_areas = np.pi * (r_p_bins[1:] ** 2 - r_p_bins[:-1] ** 2)
    delta_pi = pi_bins[1:] - pi_bins[:-1]
    v_shell = 2.0 * ann_areas[:, np.newaxis] * delta_pi[np.newaxis, :]
    prefactor = n_total * (n_total - 1) / (2.0 * box_size**3)
    rr = prefactor * v_shell
    xi: NDArray[np.float64] = dd_corr / rr - 1.0
    delta_pi = np.diff(pi_bins)
    wp: NDArray[np.float64] = 2.0 * np.sum(xi * delta_pi[np.newaxis, :], axis=1)
    return {"xi": xi, "wp": wp, "dd_auto": dd_auto, "dd_cross": dd_cross, "dd_corr": dd_corr, "rr": rr}
