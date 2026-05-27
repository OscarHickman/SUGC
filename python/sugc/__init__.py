import numpy as np

from ._sugc import *  # noqa: F403
from ._sugc import count_npoint, count_pairs_1d, count_pairs_2d, count_pairs_smu

__version__ = "0.1.0"


def analytic_rr_1d(r_bins, box, n):
    """Analytic RR for real-space estimator: N(N-1)/2 * V_shell / V_box."""
    r_bins = np.asarray(r_bins, dtype=np.float64)
    v_shell = (4.0 * np.pi / 3.0) * (r_bins[1:] ** 3 - r_bins[:-1] ** 3)
    return n * (n - 1) / 2.0 * v_shell / box**3


def analytic_rr(rp_bins, pi_bins, box, n):
    """Analytic RR for 2D (r_p, π) estimator: N(N-1)/2 * V_cyl / V_box."""
    rp_bins = np.asarray(rp_bins, dtype=np.float64)
    pi_bins = np.asarray(pi_bins, dtype=np.float64)
    ann_area = np.pi * (rp_bins[1:] ** 2 - rp_bins[:-1] ** 2)
    delta_pi = np.diff(pi_bins)
    v_cyl = 2.0 * ann_area[:, np.newaxis] * delta_pi[np.newaxis, :]
    return n * (n - 1) / 2.0 * v_cyl / box**3


def compute_xi(
    coords,
    partition_ids,
    r_bins,
    box_size,
    n_partitions,
    n_partitions_selected,
):
    """Compute ξ(r) with sparse-partition correction.

    Returns a dict with keys: xi, dd_auto, dd_cross, dd_corr, rr, r_mid.
    """
    k, m = n_partitions, n_partitions_selected
    if m > k:
        raise ValueError(
            f"n_partitions_selected ({m}) cannot exceed n_partitions ({k})"
        )
    if m <= 0:
        raise ValueError(f"n_partitions_selected must be >= 1, got {m}")

    dd_auto, dd_cross = count_pairs_1d(coords, partition_ids, r_bins, box_size)

    alpha = m / k
    beta = m * (k - 1) / (k * (m - 1)) if m > 1 else 0.0
    dd_corr = alpha * dd_auto + beta * dd_cross

    rr = analytic_rr_1d(r_bins, box_size, len(coords))
    r_bins = np.asarray(r_bins, dtype=np.float64)
    r_mid = np.sqrt(r_bins[:-1] * r_bins[1:])
    xi = dd_corr / rr - 1.0

    return {
        "xi": xi,
        "dd_auto": dd_auto,
        "dd_cross": dd_cross,
        "dd_corr": dd_corr,
        "rr": rr,
        "r_mid": r_mid,
    }


def compute_2pcf(
    coords,
    partition_ids,
    rp_bins,
    pi_bins,
    box_size,
    n_partitions,
    n_partitions_selected,
    n_total=None,
):
    """Compute ξ(r_p, π) and w_p(r_p) with sparse-partition correction.

    Returns a dict with keys: xi, wp, dd_auto, dd_cross, dd_corr, rr.
    """
    k, m = n_partitions, n_partitions_selected
    if m > k:
        raise ValueError(
            f"n_partitions_selected ({m}) cannot exceed n_partitions ({k})"
        )
    if m <= 0:
        raise ValueError(f"n_partitions_selected must be >= 1, got {m}")

    dd_auto, dd_cross = count_pairs_2d(
        coords, partition_ids, rp_bins, pi_bins, box_size
    )

    alpha = k / m
    beta = k * (k - 1) / (m * (m - 1)) if m > 1 else 0.0
    dd_corr = alpha * dd_auto + beta * dd_cross

    if n_total is None:
        n_total = int(round(len(coords) * k / m))
    rr = analytic_rr(rp_bins, pi_bins, box_size, n_total)
    xi = dd_corr / rr - 1.0
    delta_pi = np.diff(np.asarray(pi_bins, dtype=np.float64))
    wp = 2.0 * np.sum(xi * delta_pi[np.newaxis, :], axis=1)

    return {
        "xi": xi,
        "wp": wp,
        "dd_auto": dd_auto,
        "dd_cross": dd_cross,
        "dd_corr": dd_corr,
        "rr": rr,
    }


def compute_xi_smu(
    coords,
    partition_ids,
    s_bins,
    box_size,
    *,
    n_partitions,
    n_partitions_selected,
    n_mu_bins,
    mu_max=1.0,
):
    """Compute ξ(s, μ) with sparse-partition correction.

    Returns a dict with keys: dd_auto, dd_cross, dd_corr.
    """
    k, m = n_partitions, n_partitions_selected
    if m > k:
        raise ValueError(
            f"n_partitions_selected ({m}) cannot exceed n_partitions ({k})"
        )
    if m <= 0:
        raise ValueError(f"n_partitions_selected must be >= 1, got {m}")

    dd_auto, dd_cross = count_pairs_smu(
        coords, partition_ids, s_bins, n_mu_bins, mu_max, box_size
    )

    alpha = m / k
    beta = m * (k - 1) / (k * (m - 1)) if m > 1 else 0.0
    dd_corr = alpha * dd_auto + beta * dd_cross

    return {"dd_auto": dd_auto, "dd_cross": dd_cross, "dd_corr": dd_corr}


def compute_npcf(
    coords,
    partition_ids,
    r_bins,
    box_size,
    n_partitions,
    n_partitions_selected,
    n_order,
):
    """Compute the N-point correlation function with sparse-partition correction.

    Returns a dict with keys: t_corr, t_total, weights.
    """
    k, m = n_partitions, n_partitions_selected
    if m > k:
        raise ValueError(
            f"n_partitions_selected ({m}) cannot exceed n_partitions ({k})"
        )
    if m <= 0:
        raise ValueError(f"n_partitions_selected must be >= 1, got {m}")

    t_by_s, t_total = count_npoint(coords, partition_ids, r_bins, box_size, n_order)

    # weight for s distinct partitions = prod_{i=0}^{s-1} (k-i)/(m-i)
    weights = np.empty(n_order, dtype=np.float64)
    for s in range(1, n_order + 1):
        w = 1.0
        for i in range(s):
            w *= (k - i) / (m - i)
        weights[s - 1] = w

    t_corr = sum(weights[s] * t_by_s[s] for s in range(n_order))

    return {"t_corr": t_corr, "t_total": t_total, "weights": weights}
