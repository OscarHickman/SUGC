"""
SCOPE: Sparse Correction Of Pair Estimators.

Computes the real-space two-point correlation function ξ(r) for galaxy
catalogues drawn from m of k independent realisations of a periodic simulation
box (e.g. m GALFORM runs out of k runs on the same P-Millennium backbone).
Each realisation spans the full coordinate range of the box at 1/k of the
total number density; realisations overlap completely in space.

The estimator is the Natural Estimator  ξ = DD_corr / RR − 1,
where:
  - DD_corr = α · DD_auto + β · DD_cross  (sub-volume corrected pair count)
  - RR is computed analytically from the mean number density and bin volumes
  - α = m/k        corrects for the under-sampling of one-halo (auto) pairs
  - β = m(k−1)/[k(m−1)]  corrects for the under-sampling of two-halo (cross) pairs

Both α and β are derived by requiring ⟨DD_corr⟩ = (m/k)²(DD_1h + DD_2h),
which matches the normalisation of the selected catalogue (N_D ∝ m/k · N_full).
Setting m = k gives α = β = 1, recovering the full-catalogue result.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scope._scope import count_pairs_1d, count_pairs_2d, count_pairs_smu  # Rust extension

__all__ = [
    "count_pairs_1d",
    "count_pairs_2d",
    "count_pairs_smu",
    "analytic_rr_1d",
    "analytic_rr_smu",
    "analytic_rr",
    "compute_xi",
    "compute_xi_smu",
    "compute_2pcf",
]


def analytic_rr_1d(
    r_bins: NDArray[np.float64],
    box_size: float,
    n_galaxies: int,
) -> NDArray[np.float64]:
    """
    Analytic RR pair counts for a uniform Poisson process in a periodic box.

    RR(r) = N(N−1)/2 · V_shell(r) / V_box

    where V_shell = (4π/3)(r_hi³ − r_lo³) and V_box = box_size³.

    Parameters
    ----------
    r_bins      : (n_r+1,) array — radial bin edges in Mpc/h
    box_size    : float           — periodic box side length in Mpc/h
    n_galaxies  : int             — number of galaxies in the catalogue

    Returns
    -------
    rr : (n_r,) float64 array
    """
    r_bins = np.asarray(r_bins, dtype=np.float64)
    v_shell = (4.0 * np.pi / 3.0) * (r_bins[1:] ** 3 - r_bins[:-1] ** 3)
    prefactor = n_galaxies * (n_galaxies - 1) / (2.0 * box_size**3)
    return prefactor * v_shell


def compute_xi(
    coords: NDArray[np.float64],
    subvol_ids: NDArray[np.int32],
    r_bins: NDArray[np.float64],
    box_size: float,
    n_subvols: int,
    n_subvols_selected: int,
) -> dict[str, NDArray[np.float64]]:
    """
    Compute the real-space two-point correlation function ξ(r).

    Uses the sub-volume auto/cross correction of Hickman et al. (2026) to
    give an unbiased estimate from a partial (m-of-k sub-volume) catalogue.

    Parameters
    ----------
    coords : (N, 3) float64 array
        Galaxy positions [x, y, z] in Mpc/h. Must be C-contiguous.
    subvol_ids : (N,) int32 array
        Realisation index for each galaxy (integer labels 0..m−1, where each
        value identifies which of the m selected realisations a galaxy belongs
        to). Labels carry no spatial meaning — each realisation spans the full
        box volume.
    r_bins : (n_r+1,) array
        Radial separation bin edges in Mpc/h.
    box_size : float
        Side length of the full periodic simulation box in Mpc/h.
    n_subvols : int
        Total number of independent realisations k.
    n_subvols_selected : int
        Number of realisations m included in `coords` (1 ≤ m ≤ k).

    Returns
    -------
    dict with keys
        ``xi``       — (n_r,) correlation function ξ(r)
        ``dd_auto``  — (n_r,) same-subvol raw pair counts
        ``dd_cross`` — (n_r,) cross-subvol raw pair counts
        ``dd_corr``  — (n_r,) sub-volume corrected pair counts
        ``rr``       — (n_r,) analytic RR pair counts
        ``r_mid``    — (n_r,) bin centre radii (geometric mean of edges)
    """
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    subvol_ids = np.ascontiguousarray(subvol_ids, dtype=np.int32)
    r_bins = np.asarray(r_bins, dtype=np.float64)

    k = int(n_subvols)
    m = int(n_subvols_selected)
    if m < 1 or m > k:
        raise ValueError(f"n_subvols_selected={m} must be in [1, {k}]")

    n_selected = len(coords)

    # ------------------------------------------------------------------ #
    #  Rust pair counter                                                   #
    # ------------------------------------------------------------------ #
    dd_auto, dd_cross = count_pairs_1d(coords, subvol_ids, r_bins, float(box_size))

    # ------------------------------------------------------------------ #
    #  Sub-volume correction weights (Hickman et al. 2026 Eqs. 9–10)     #
    #                                                                      #
    #  Target: ⟨DD_corr⟩ = (m/k)²(DD_1h + DD_2h)                        #
    #  α = m/k,  β = m(k−1)/[k(m−1)]                                     #
    #  When m = 1 there are no cross pairs, so β is set to 0.             #
    # ------------------------------------------------------------------ #
    alpha = m / k
    beta = m * (k - 1) / (k * (m - 1)) if m > 1 else 0.0

    dd_corr: NDArray[np.float64] = alpha * dd_auto + beta * dd_cross

    # ------------------------------------------------------------------ #
    #  Analytic RR (normalised to the selected catalogue)                 #
    # ------------------------------------------------------------------ #
    rr = analytic_rr_1d(r_bins, box_size, n_selected)

    xi: NDArray[np.float64] = dd_corr / rr - 1.0

    r_mid: NDArray[np.float64] = np.sqrt(r_bins[:-1] * r_bins[1:])

    return {
        "xi": xi,
        "dd_auto": dd_auto,
        "dd_cross": dd_cross,
        "dd_corr": dd_corr,
        "rr": rr,
        "r_mid": r_mid,
    }


# ── Legacy 2D (r_p, π) interface kept for backward compatibility ─────────────

def analytic_rr(
    r_p_bins: NDArray[np.float64],
    pi_bins: NDArray[np.float64],
    box_size: float,
    n_total: int,
) -> NDArray[np.float64]:
    """
    Analytic RR for a 2D (r_p, π) bin in a periodic box.

    RR = N(N−1)/2 · π(r_p_hi²−r_p_lo²) · 2Δπ / V_box

    Parameters
    ----------
    r_p_bins : (n_rp+1,) array  — transverse bin edges
    pi_bins  : (n_pi+1,) array  — LOS bin edges
    box_size : float             — periodic box side length
    n_total  : int               — total number of galaxies

    Returns
    -------
    rr : (n_rp, n_pi) float64 array
    """
    r_p_bins = np.asarray(r_p_bins, dtype=np.float64)
    pi_bins = np.asarray(pi_bins, dtype=np.float64)

    ann_areas = np.pi * (r_p_bins[1:] ** 2 - r_p_bins[:-1] ** 2)
    delta_pi = pi_bins[1:] - pi_bins[:-1]
    v_shell = 2.0 * ann_areas[:, np.newaxis] * delta_pi[np.newaxis, :]
    prefactor = n_total * (n_total - 1) / (2.0 * box_size**3)
    return prefactor * v_shell


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
    """
    Compute ξ(r_p, π) and w_p(r_p) for a sub-volume selected galaxy sample.

    Parameters
    ----------
    coords : (N, 3) float64 array
        Galaxy positions [x, y, z] in Mpc/h. z is the line-of-sight axis.
    subvol_ids : (N,) int32 array
        Sub-volume index for each galaxy.
    r_p_bins : (n_rp+1,) array
        Transverse separation bin edges in Mpc/h.
    pi_bins : (n_pi+1,) array
        LOS separation bin edges in Mpc/h.
    box_size : float
        Side length of the full periodic simulation box in Mpc/h.
    n_subvols : int
        Total number of sub-volumes k.
    n_subvols_selected : int
        Number of sub-volumes m included in `coords`.
    n_total : int, optional
        Total galaxies in the full simulation. Defaults to N_sel · k/m.

    Returns
    -------
    dict with keys ``xi``, ``wp``, ``dd_auto``, ``dd_cross``, ``dd_corr``, ``rr``
    """
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    subvol_ids = np.ascontiguousarray(subvol_ids, dtype=np.int32)
    r_p_bins = np.asarray(r_p_bins, dtype=np.float64)
    pi_bins = np.asarray(pi_bins, dtype=np.float64)

    k = int(n_subvols)
    m = int(n_subvols_selected)
    if m < 1 or m > k:
        raise ValueError(f"n_subvols_selected={m} must be in [1, {k}]")

    n_selected = len(coords)
    if n_total is None:
        n_total = int(round(n_selected * k / m))

    dd_auto, dd_cross = count_pairs_2d(
        coords, subvol_ids, r_p_bins, pi_bins, float(box_size)
    )

    # Scale up to full-box counts: α=k/m, β=k(k−1)/[m(m−1)]
    alpha = k / m
    beta = k * (k - 1) / (m * (m - 1)) if m > 1 else 0.0

    dd_corr: NDArray[np.float64] = alpha * dd_auto + beta * dd_cross

    rr = analytic_rr(r_p_bins, pi_bins, box_size, n_total)
    xi: NDArray[np.float64] = dd_corr / rr - 1.0

    delta_pi = np.diff(pi_bins)
    wp: NDArray[np.float64] = 2.0 * np.sum(xi * delta_pi[np.newaxis, :], axis=1)

    return {
        "xi": xi,
        "wp": wp,
        "dd_auto": dd_auto,
        "dd_cross": dd_cross,
        "dd_corr": dd_corr,
        "rr": rr,
    }


# ── Redshift-space distortions: (s, μ) estimator ────────────────────────────

def analytic_rr_smu(
    s_bins: NDArray[np.float64],
    mu_max: float,
    n_mu_bins: int,
    box_size: float,
    n_galaxies: int,
) -> NDArray[np.float64]:
    """
    Analytic RR pair counts for a uniform Poisson process in (s, μ) bins.

    For random points in a periodic box, μ = |Δz|/s is uniformly distributed
    on [0, 1], so the expected pair count in bin (s_i, μ_j) is:

        RR(s, μ) = N(N−1)/2 · V_shell(s) · Δμ / V_box

    Parameters
    ----------
    s_bins     : (n_s+1,) array — redshift-space separation bin edges in Mpc/h
    mu_max     : float           — upper edge of the μ range (typically 1.0)
    n_mu_bins  : int             — number of uniform μ bins in [0, mu_max]
    box_size   : float           — periodic box side length in Mpc/h
    n_galaxies : int             — number of galaxies in the catalogue

    Returns
    -------
    rr : (n_s, n_mu_bins) float64 array
    """
    s_bins = np.asarray(s_bins, dtype=np.float64)
    v_shell = (4.0 * np.pi / 3.0) * (s_bins[1:] ** 3 - s_bins[:-1] ** 3)
    prefactor = n_galaxies * (n_galaxies - 1) / (2.0 * box_size**3)
    dmu = mu_max / n_mu_bins
    return (prefactor * v_shell)[:, np.newaxis] * np.full(n_mu_bins, dmu)


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
    """
    Compute ξ(s, μ) and its Legendre multipoles ξ₀(s) and ξ₂(s).

    Caller must pre-apply the RSD displacement before passing coords:
        z_rsd = (z + v_pec_z / H(z)) % box_size   (all in Mpc/h)

    The LOS is the z-axis. μ = |Δz_rsd| / s ∈ [0, mu_max].

    Parameters
    ----------
    coords : (N, 3) float64 array
        Redshift-space positions [x, y, z_rsd] in Mpc/h. Must be C-contiguous.
    subvol_ids : (N,) int32 array
        Realisation index for each galaxy.
    s_bins : (n_s+1,) array
        Redshift-space separation bin edges in Mpc/h.
    box_size : float
        Periodic box side length in Mpc/h.
    n_subvols : int
        Total number of independent realisations k.
    n_subvols_selected : int
        Number of realisations m included in `coords`.
    n_mu_bins : int
        Number of uniform μ bins in [0, mu_max]. Default 100.
    mu_max : float
        Upper edge of the μ range. Default 1.0.

    Returns
    -------
    dict with keys
        ``xi_smu``   — (n_s, n_mu) ξ(s, μ) grid
        ``xi0``      — (n_s,) monopole ξ₀(s)
        ``xi2``      — (n_s,) quadrupole ξ₂(s)
        ``dd_auto``  — (n_s, n_mu) same-subvol raw pair counts
        ``dd_cross`` — (n_s, n_mu) cross-subvol raw pair counts
        ``dd_corr``  — (n_s, n_mu) sub-volume corrected pair counts
        ``rr``       — (n_s, n_mu) analytic RR pair counts
        ``s_mid``    — (n_s,) geometric-mean bin centres in Mpc/h
        ``mu_mid``   — (n_mu,) μ bin centres
    """
    coords     = np.ascontiguousarray(coords,     dtype=np.float64)
    subvol_ids = np.ascontiguousarray(subvol_ids, dtype=np.int32)
    s_bins     = np.asarray(s_bins, dtype=np.float64)

    k = int(n_subvols)
    m = int(n_subvols_selected)
    if m < 1 or m > k:
        raise ValueError(f"n_subvols_selected={m} must be in [1, {k}]")

    n_selected = len(coords)

    # ------------------------------------------------------------------ #
    #  Rust pair counter                                                   #
    # ------------------------------------------------------------------ #
    dd_auto, dd_cross = count_pairs_smu(
        coords, subvol_ids, s_bins, n_mu_bins, mu_max, float(box_size)
    )

    # ------------------------------------------------------------------ #
    #  Sub-volume correction (same α/β as compute_xi)                     #
    # ------------------------------------------------------------------ #
    alpha = m / k
    beta  = m * (k - 1) / (k * (m - 1)) if m > 1 else 0.0
    dd_corr: NDArray[np.float64] = alpha * dd_auto + beta * dd_cross

    # ------------------------------------------------------------------ #
    #  ξ(s, μ) via natural estimator                                      #
    # ------------------------------------------------------------------ #
    rr = analytic_rr_smu(s_bins, mu_max, n_mu_bins, float(box_size), n_selected)

    with np.errstate(divide="ignore", invalid="ignore"):
        xi_smu: NDArray[np.float64] = dd_corr / rr - 1.0
    xi_smu[~np.isfinite(xi_smu)] = np.nan

    # ------------------------------------------------------------------ #
    #  Legendre projection                                                 #
    # ------------------------------------------------------------------ #
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
