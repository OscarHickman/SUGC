"""
Unit tests for the SCOPE correlation-function estimator.

Coverage
--------
count_pairs_1d   pair counting, auto/cross routing, periodic BCs, edge cases
count_pairs_2d   same for the 2D (r_p, π) counter
analytic_rr_1d   shell-volume RR formula
analytic_rr      2D annular-volume RR formula
compute_xi       full 1D pipeline: weights, output contract, physics
compute_2pcf     full 2D pipeline: weights, wp integral, physics

Physics check (ξ ≈ 0)
---------------------
Both compute_xi and compute_2pcf are tested on a uniform Poisson field
where the true correlation is zero.  With enough particles the Poisson
noise on ξ is small; we demand |ξ| < 0.05 in every well-populated bin
(RR > 100), consistent with the CLAUDE.md acceptance criterion.
"""

import numpy as np
import pytest

from scope._scope import count_pairs_1d, count_pairs_2d
from scope import analytic_rr_1d, analytic_rr, compute_xi, compute_2pcf


# ─── Brute-force references ───────────────────────────────────────────────────

def _bf_1d(coords, sv_ids, r_bins, box):
    """O(N²) pair counter — correct by construction, used to validate Rust code."""
    n = len(coords)
    r_sq_bins = r_bins ** 2
    n_bins = len(r_bins) - 1
    auto = np.zeros(n_bins)
    cross = np.zeros(n_bins)
    for i in range(n):
        d = coords[i + 1:] - coords[i]
        d -= box * np.round(d / box)           # minimum-image
        r_sq = (d ** 2).sum(axis=1)
        same = sv_ids[i + 1:] == sv_ids[i]
        for k in range(n_bins):
            in_bin = (r_sq >= r_sq_bins[k]) & (r_sq < r_sq_bins[k + 1])
            auto[k] += np.count_nonzero(in_bin & same)
            cross[k] += np.count_nonzero(in_bin & ~same)
    return auto, cross


def _bf_2d(coords, sv_ids, rp_bins, pi_bins, box):
    """O(N²) 2D pair counter."""
    n = len(coords)
    rp_sq_bins = rp_bins ** 2
    n_rp, n_pi = len(rp_bins) - 1, len(pi_bins) - 1
    auto = np.zeros((n_rp, n_pi))
    cross = np.zeros((n_rp, n_pi))
    for i in range(n):
        d = coords[i + 1:] - coords[i]
        d -= box * np.round(d / box)
        rp_sq = d[:, 0] ** 2 + d[:, 1] ** 2
        pi = np.abs(d[:, 2])
        same = sv_ids[i + 1:] == sv_ids[i]
        for irp in range(n_rp):
            rp_ok = (rp_sq >= rp_sq_bins[irp]) & (rp_sq < rp_sq_bins[irp + 1])
            for ipi in range(n_pi):
                pi_ok = (pi >= pi_bins[ipi]) & (pi < pi_bins[ipi + 1])
                in_bin = rp_ok & pi_ok
                auto[irp, ipi] += np.count_nonzero(in_bin & same)
                cross[irp, ipi] += np.count_nonzero(in_bin & ~same)
    return auto, cross


# ─── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_cat():
    """60-particle catalogue: small enough for exact brute-force comparison."""
    rng = np.random.default_rng(1)
    n, box = 60, 80.0
    coords = rng.uniform(0, box, (n, 3)).astype(np.float64)
    sv_ids = rng.integers(0, 4, n).astype(np.int32)
    r_bins = np.array([1.0, 5.0, 12.0, 28.0, 55.0])
    return coords, sv_ids, box, r_bins


@pytest.fixture(scope="module")
def small_cat_2d():
    """50-particle catalogue for exact 2D brute-force comparison."""
    rng = np.random.default_rng(2)
    n, box = 50, 60.0
    coords = rng.uniform(0, box, (n, 3)).astype(np.float64)
    sv_ids = rng.integers(0, 3, n).astype(np.int32)
    rp_bins = np.array([1.0, 5.0, 15.0])
    pi_bins = np.array([0.0, 5.0, 15.0, 30.0])
    return coords, sv_ids, box, rp_bins, pi_bins


@pytest.fixture(scope="module")
def uniform_cat_1d():
    """4 000-particle uniform field for ξ ≈ 0 physics check (1D)."""
    rng = np.random.default_rng(42)
    n, box, k = 4000, 300.0, 4
    coords = rng.uniform(0, box, (n, 3)).astype(np.float64)
    sv_ids = rng.integers(0, k, n).astype(np.int32)
    r_bins = np.array([5.0, 15.0, 40.0, 100.0])
    return coords, sv_ids, box, r_bins, k


@pytest.fixture(scope="module")
def uniform_cat_2d():
    """3 000-particle uniform field for ξ ≈ 0 physics check (2D)."""
    rng = np.random.default_rng(3)
    n, box, k = 3000, 200.0, 4
    coords = rng.uniform(0, box, (n, 3)).astype(np.float64)
    sv_ids = rng.integers(0, k, n).astype(np.int32)
    rp_bins = np.array([1.0, 5.0, 15.0, 40.0])
    pi_bins = np.array([0.0, 10.0, 30.0, 60.0])
    return coords, sv_ids, box, rp_bins, pi_bins, k


# ─── count_pairs_1d ───────────────────────────────────────────────────────────

class TestCountPairs1D:
    def test_matches_brute_force_auto(self, small_cat):
        coords, sv_ids, box, r_bins = small_cat
        auto, _ = count_pairs_1d(coords, sv_ids, r_bins, box)
        bf_auto, _ = _bf_1d(coords, sv_ids, r_bins, box)
        assert np.allclose(auto, bf_auto)

    def test_matches_brute_force_cross(self, small_cat):
        coords, sv_ids, box, r_bins = small_cat
        _, cross = count_pairs_1d(coords, sv_ids, r_bins, box)
        _, bf_cross = _bf_1d(coords, sv_ids, r_bins, box)
        assert np.allclose(cross, bf_cross)

    def test_auto_plus_cross_equals_total(self, small_cat):
        """auto + cross must equal every pair in range, regardless of sv routing."""
        coords, sv_ids, box, r_bins = small_cat
        auto, cross = count_pairs_1d(coords, sv_ids, r_bins, box)
        sv_ones = np.zeros(len(coords), dtype=np.int32)
        total, _ = count_pairs_1d(coords, sv_ones, r_bins, box)
        assert np.allclose(auto + cross, total)

    def test_all_same_subvol_zero_cross(self, small_cat):
        """With a single sub-volume label, every pair is auto; cross must be zero."""
        coords, _, box, r_bins = small_cat
        sv_ids = np.zeros(len(coords), dtype=np.int32)
        auto, cross = count_pairs_1d(coords, sv_ids, r_bins, box)
        assert np.all(cross == 0.0)

    def test_all_distinct_subvols_zero_auto(self, small_cat):
        """With every particle in its own sub-volume, auto must be zero."""
        coords, _, box, r_bins = small_cat
        sv_ids = np.arange(len(coords), dtype=np.int32)
        auto, cross = count_pairs_1d(coords, sv_ids, r_bins, box)
        assert np.all(auto == 0.0)

    def test_periodic_boundary(self):
        """A pair straddling the periodic boundary is counted in the correct bin."""
        box = 100.0
        # True minimum-image separation = 2 Mpc/h (across the boundary)
        coords = np.array([[1.0, 50.0, 50.0], [99.0, 50.0, 50.0]], dtype=np.float64)
        sv_ids = np.array([0, 0], dtype=np.int32)
        r_bins = np.array([1.0, 3.0, 10.0])
        auto, cross = count_pairs_1d(coords, sv_ids, r_bins, box)
        assert auto[0] == 1.0   # bin [1, 3) contains the pair
        assert auto[1] == 0.0

    def test_pair_beyond_rmax_excluded(self):
        box = 200.0
        coords = np.array([[0.0, 0.0, 0.0], [60.0, 0.0, 0.0]], dtype=np.float64)
        sv_ids = np.array([0, 1], dtype=np.int32)
        r_bins = np.array([1.0, 50.0])        # r_max = 50 < 60
        auto, cross = count_pairs_1d(coords, sv_ids, r_bins, box)
        assert auto[0] == 0.0 and cross[0] == 0.0

    def test_pair_below_rmin_excluded(self):
        box = 200.0
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        sv_ids = np.array([0, 0], dtype=np.int32)
        r_bins = np.array([5.0, 50.0])        # r_min = 5 > 1
        auto, cross = count_pairs_1d(coords, sv_ids, r_bins, box)
        assert auto[0] == 0.0

    def test_two_particles_lands_in_correct_bin(self):
        """Pair at known r contributes exactly 1 count to the expected bin."""
        box = 200.0
        coords = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)
        sv_ids = np.array([0, 1], dtype=np.int32)
        r_bins = np.array([1.0, 5.0, 15.0, 50.0])
        auto, cross = count_pairs_1d(coords, sv_ids, r_bins, box)
        # r = 10 → bin index 1, which is the [5, 15) bin
        assert cross[1] == 1.0
        assert auto[1] == 0.0
        assert np.sum(auto) + np.sum(cross) == 1.0

    def test_ordering_invariant(self, small_cat):
        """Shuffling particle order must not change pair counts."""
        coords, sv_ids, box, r_bins = small_cat
        rng = np.random.default_rng(7)
        perm = rng.permutation(len(coords))
        auto1, cross1 = count_pairs_1d(coords, sv_ids, r_bins, box)
        auto2, cross2 = count_pairs_1d(coords[perm], sv_ids[perm], r_bins, box)
        assert np.allclose(auto1, auto2)
        assert np.allclose(cross1, cross2)

    def test_output_shape(self, small_cat):
        coords, sv_ids, box, r_bins = small_cat
        auto, cross = count_pairs_1d(coords, sv_ids, r_bins, box)
        n_bins = len(r_bins) - 1
        assert auto.shape == (n_bins,)
        assert cross.shape == (n_bins,)

    def test_counts_nonnegative(self, small_cat):
        coords, sv_ids, box, r_bins = small_cat
        auto, cross = count_pairs_1d(coords, sv_ids, r_bins, box)
        assert np.all(auto >= 0) and np.all(cross >= 0)

    def test_counts_are_integer_valued(self, small_cat):
        """Pair counts are whole numbers stored as float64."""
        coords, sv_ids, box, r_bins = small_cat
        auto, cross = count_pairs_1d(coords, sv_ids, r_bins, box)
        assert np.all(auto == np.floor(auto))
        assert np.all(cross == np.floor(cross))


# ─── count_pairs_2d ───────────────────────────────────────────────────────────

class TestCountPairs2D:
    def test_matches_brute_force_auto(self, small_cat_2d):
        coords, sv_ids, box, rp_bins, pi_bins = small_cat_2d
        auto, _ = count_pairs_2d(coords, sv_ids, rp_bins, pi_bins, box)
        bf_auto, _ = _bf_2d(coords, sv_ids, rp_bins, pi_bins, box)
        assert np.allclose(auto, bf_auto)

    def test_matches_brute_force_cross(self, small_cat_2d):
        coords, sv_ids, box, rp_bins, pi_bins = small_cat_2d
        _, cross = count_pairs_2d(coords, sv_ids, rp_bins, pi_bins, box)
        _, bf_cross = _bf_2d(coords, sv_ids, rp_bins, pi_bins, box)
        assert np.allclose(cross, bf_cross)

    def test_auto_plus_cross_equals_total(self, small_cat_2d):
        coords, sv_ids, box, rp_bins, pi_bins = small_cat_2d
        auto, cross = count_pairs_2d(coords, sv_ids, rp_bins, pi_bins, box)
        sv_ones = np.zeros(len(coords), dtype=np.int32)
        total, _ = count_pairs_2d(coords, sv_ones, rp_bins, pi_bins, box)
        assert np.allclose(auto + cross, total)

    def test_all_same_subvol_zero_cross(self, small_cat_2d):
        coords, _, box, rp_bins, pi_bins = small_cat_2d
        sv_ids = np.zeros(len(coords), dtype=np.int32)
        auto, cross = count_pairs_2d(coords, sv_ids, rp_bins, pi_bins, box)
        assert np.all(cross == 0.0)

    def test_all_distinct_subvols_zero_auto(self, small_cat_2d):
        coords, _, box, rp_bins, pi_bins = small_cat_2d
        sv_ids = np.arange(len(coords), dtype=np.int32)
        auto, cross = count_pairs_2d(coords, sv_ids, rp_bins, pi_bins, box)
        assert np.all(auto == 0.0)

    def test_periodic_boundary_transverse(self):
        """A pair separated across the box in the transverse plane is counted."""
        box = 100.0
        # Transverse (x) separation = 2 Mpc/h via periodic boundary; π = 0
        coords = np.array([[1.0, 50.0, 50.0], [99.0, 50.0, 50.0]], dtype=np.float64)
        sv_ids = np.array([0, 0], dtype=np.int32)
        rp_bins = np.array([1.0, 3.0, 20.0])
        pi_bins = np.array([0.0, 5.0])
        auto, cross = count_pairs_2d(coords, sv_ids, rp_bins, pi_bins, box)
        assert auto[0, 0] == 1.0   # r_p=2 in [1,3), π=0 in [0,5)

    def test_output_shape(self, small_cat_2d):
        coords, sv_ids, box, rp_bins, pi_bins = small_cat_2d
        auto, cross = count_pairs_2d(coords, sv_ids, rp_bins, pi_bins, box)
        n_rp = len(rp_bins) - 1
        n_pi = len(pi_bins) - 1
        assert auto.shape == (n_rp, n_pi)
        assert cross.shape == (n_rp, n_pi)

    def test_counts_nonnegative(self, small_cat_2d):
        coords, sv_ids, box, rp_bins, pi_bins = small_cat_2d
        auto, cross = count_pairs_2d(coords, sv_ids, rp_bins, pi_bins, box)
        assert np.all(auto >= 0) and np.all(cross >= 0)

    def test_counts_are_integer_valued(self, small_cat_2d):
        coords, sv_ids, box, rp_bins, pi_bins = small_cat_2d
        auto, cross = count_pairs_2d(coords, sv_ids, rp_bins, pi_bins, box)
        assert np.all(auto == np.floor(auto))
        assert np.all(cross == np.floor(cross))


# ─── analytic_rr_1d ───────────────────────────────────────────────────────────

class TestAnalyticRR1D:
    def test_matches_formula(self):
        """RR(r) = N(N-1)/2 · V_shell(r) / V_box."""
        n, box = 1000, 200.0
        r_bins = np.array([2.0, 8.0, 25.0, 60.0])
        rr = analytic_rr_1d(r_bins, box, n)
        v_shell = (4 * np.pi / 3) * (r_bins[1:] ** 3 - r_bins[:-1] ** 3)
        expected = n * (n - 1) / 2 * v_shell / box ** 3
        assert np.allclose(rr, expected)

    def test_output_shape(self):
        r_bins = np.array([1.0, 5.0, 10.0, 20.0])
        rr = analytic_rr_1d(r_bins, 100.0, 500)
        assert rr.shape == (3,)

    def test_nonnegative(self):
        r_bins = np.logspace(0, 2, 11)
        rr = analytic_rr_1d(r_bins, 300.0, 1000)
        assert np.all(rr >= 0)

    def test_scales_quadratically_with_n(self):
        """RR ∝ N(N-1) — checks the pair-count prefactor."""
        r_bins = np.array([1.0, 10.0])
        box = 100.0
        rr100 = analytic_rr_1d(r_bins, box, 100)
        rr200 = analytic_rr_1d(r_bins, box, 200)
        assert np.isclose(rr200[0] / rr100[0], 200 * 199 / (100 * 99))

    def test_scales_inversely_with_box_volume(self):
        """RR ∝ 1/V_box — number density effect."""
        r_bins = np.array([1.0, 5.0])
        rr100 = analytic_rr_1d(r_bins, 100.0, 500)
        rr200 = analytic_rr_1d(r_bins, 200.0, 500)
        assert np.isclose(rr200[0] / rr100[0], (100.0 / 200.0) ** 3)

    def test_wider_bin_has_more_rr(self):
        """A wider radial bin must contain more RR than a narrower one at the same inner edge."""
        rr_narrow = analytic_rr_1d(np.array([5.0, 10.0]), 100.0, 500)
        rr_wide = analytic_rr_1d(np.array([5.0, 15.0]), 100.0, 500)
        assert rr_wide[0] > rr_narrow[0]

    def test_monotone_bins_give_more_rr_at_larger_r(self):
        """Outer shells (larger r) have more volume, hence more RR for equal-width bins."""
        r_bins = np.array([5.0, 10.0, 15.0])
        rr = analytic_rr_1d(r_bins, 200.0, 500)
        assert rr[1] > rr[0]


# ─── analytic_rr (2D) ─────────────────────────────────────────────────────────

class TestAnalyticRR2D:
    def test_matches_formula(self):
        """RR = N(N-1)/2 · π(r_p_hi²−r_p_lo²) · 2Δπ / V_box."""
        rp_bins = np.array([2.0, 6.0])
        pi_bins = np.array([3.0, 9.0])
        box, n = 100.0, 500
        rr = analytic_rr(rp_bins, pi_bins, box, n)
        ann_area = np.pi * (6.0 ** 2 - 2.0 ** 2)
        v = 2.0 * ann_area * (9.0 - 3.0)
        expected = n * (n - 1) / 2 * v / box ** 3
        assert np.isclose(rr[0, 0], expected)

    def test_output_shape(self):
        rp_bins = np.array([0.0, 5.0, 10.0])
        pi_bins = np.array([0.0, 5.0, 15.0, 30.0])
        rr = analytic_rr(rp_bins, pi_bins, 100.0, 500)
        assert rr.shape == (2, 3)

    def test_nonnegative(self):
        rp_bins = np.array([1.0, 5.0, 15.0])
        pi_bins = np.array([0.0, 10.0, 30.0])
        rr = analytic_rr(rp_bins, pi_bins, 300.0, 1000)
        assert np.all(rr >= 0)

    def test_scales_quadratically_with_n(self):
        rp_bins = np.array([1.0, 10.0])
        pi_bins = np.array([0.0, 20.0])
        box = 200.0
        rr100 = analytic_rr(rp_bins, pi_bins, box, 100)
        rr200 = analytic_rr(rp_bins, pi_bins, box, 200)
        assert np.isclose(rr200[0, 0] / rr100[0, 0], 200 * 199 / (100 * 99))


# ─── compute_xi ───────────────────────────────────────────────────────────────

class TestComputeXi:
    def test_output_keys(self, small_cat):
        coords, sv_ids, box, r_bins = small_cat
        result = compute_xi(coords, sv_ids, r_bins, box, 4, 4)
        assert {"xi", "dd_auto", "dd_cross", "dd_corr", "rr", "r_mid"} <= set(result)

    def test_output_shapes(self, small_cat):
        coords, sv_ids, box, r_bins = small_cat
        n_bins = len(r_bins) - 1
        result = compute_xi(coords, sv_ids, r_bins, box, 4, 4)
        for key in ("xi", "dd_auto", "dd_cross", "dd_corr", "rr", "r_mid"):
            assert result[key].shape == (n_bins,), f"wrong shape for '{key}'"

    def test_r_mid_is_geometric_mean(self, small_cat):
        coords, sv_ids, box, r_bins = small_cat
        result = compute_xi(coords, sv_ids, r_bins, box, 4, 4)
        assert np.allclose(result["r_mid"], np.sqrt(r_bins[:-1] * r_bins[1:]))

    def test_dd_corr_formula(self, small_cat):
        """dd_corr = α · dd_auto + β · dd_cross, with α = m/k, β = m(k-1)/[k(m-1)]."""
        coords, sv_ids, box, r_bins = small_cat
        k, m = 4, 3
        result = compute_xi(coords, sv_ids, r_bins, box, k, m)
        alpha = m / k
        beta = m * (k - 1) / (k * (m - 1))
        expected = alpha * result["dd_auto"] + beta * result["dd_cross"]
        assert np.allclose(result["dd_corr"], expected)

    def test_xi_equals_dd_corr_over_rr_minus_one(self, small_cat):
        """ξ = DD_corr / RR − 1."""
        coords, sv_ids, box, r_bins = small_cat
        result = compute_xi(coords, sv_ids, r_bins, box, 4, 4)
        assert np.allclose(result["xi"], result["dd_corr"] / result["rr"] - 1.0)

    def test_m_equals_k_gives_alpha_beta_one(self, small_cat):
        """For m = k: α = β = 1, so dd_corr = dd_auto + dd_cross."""
        coords, sv_ids, box, r_bins = small_cat
        k = 4
        result = compute_xi(coords, sv_ids, r_bins, box, k, k)
        assert np.allclose(result["dd_corr"], result["dd_auto"] + result["dd_cross"])

    def test_m_equals_1_zero_cross_contribution(self):
        """For m = 1: β = 0, all pairs are auto; dd_corr = (1/k) · dd_auto."""
        rng = np.random.default_rng(5)
        n, box, k = 80, 100.0, 5
        coords = rng.uniform(0, box, (n, 3)).astype(np.float64)
        sv_ids = np.zeros(n, dtype=np.int32)          # single sub-volume
        r_bins = np.array([1.0, 10.0, 40.0])
        result = compute_xi(coords, sv_ids, r_bins, box, k, 1)
        assert np.all(result["dd_cross"] == 0.0)
        assert np.allclose(result["dd_corr"], (1 / k) * result["dd_auto"])

    def test_rr_is_positive(self, small_cat):
        coords, sv_ids, box, r_bins = small_cat
        result = compute_xi(coords, sv_ids, r_bins, box, 4, 4)
        assert np.all(result["rr"] > 0)

    def test_rr_uses_selected_n(self):
        """RR is computed from n_selected = len(coords), not the full catalogue size."""
        rng = np.random.default_rng(9)
        n, box, k = 100, 200.0, 10
        coords = rng.uniform(0, box, (n, 3)).astype(np.float64)
        sv_ids = rng.integers(0, 2, n).astype(np.int32)  # m = 2 sub-volumes selected
        r_bins = np.array([1.0, 20.0])
        result = compute_xi(coords, sv_ids, r_bins, box, k, 2)
        expected_rr = analytic_rr_1d(r_bins, box, n)
        assert np.allclose(result["rr"], expected_rr)

    def test_invalid_m_exceeds_k(self, small_cat):
        coords, sv_ids, box, r_bins = small_cat
        with pytest.raises(ValueError):
            compute_xi(coords, sv_ids, r_bins, box, n_subvols=4, n_subvols_selected=5)

    def test_invalid_m_is_zero(self, small_cat):
        coords, sv_ids, box, r_bins = small_cat
        with pytest.raises(ValueError):
            compute_xi(coords, sv_ids, r_bins, box, n_subvols=4, n_subvols_selected=0)

    def test_alpha_varies_with_m(self):
        """Halving m halves α and therefore dd_corr_auto-part."""
        rng = np.random.default_rng(11)
        n, box = 80, 100.0
        coords = rng.uniform(0, box, (n, 3)).astype(np.float64)
        sv_ids = rng.integers(0, 4, n).astype(np.int32)
        r_bins = np.array([1.0, 30.0])
        r2 = compute_xi(coords, sv_ids, r_bins, box, 8, 2)
        r4 = compute_xi(coords, sv_ids, r_bins, box, 8, 4)
        # α(m=2) = 2/8 = 0.25,  α(m=4) = 4/8 = 0.5 → ratio = 0.5
        assert np.isclose(r2["dd_corr"][0] / r4["dd_corr"][0],
                          (2 / 8 * r2["dd_auto"][0] + 2 * 7 / (8 * 1) * r2["dd_cross"][0]) /
                          (4 / 8 * r4["dd_auto"][0] + 4 * 7 / (8 * 3) * r4["dd_cross"][0]),
                          rtol=1e-9)

    def test_uniform_random_xi_near_zero(self, uniform_cat_1d):
        """ξ ≈ 0 for a Poisson field in well-populated bins (CLAUDE.md criterion)."""
        coords, sv_ids, box, r_bins, k = uniform_cat_1d
        result = compute_xi(coords, sv_ids, r_bins, box, k, k)
        populated = result["rr"] > 100
        assert populated.any(), "no populated bins — increase N or widen bins"
        assert np.all(np.abs(result["xi"][populated]) < 0.05)


# ─── compute_2pcf ─────────────────────────────────────────────────────────────

class TestCompute2PCF:
    def test_output_keys(self, uniform_cat_2d):
        coords, sv_ids, box, rp_bins, pi_bins, k = uniform_cat_2d
        result = compute_2pcf(coords, sv_ids, rp_bins, pi_bins, box, k, k)
        assert {"xi", "wp", "dd_auto", "dd_cross", "dd_corr", "rr"} <= set(result)

    def test_output_shapes(self, uniform_cat_2d):
        coords, sv_ids, box, rp_bins, pi_bins, k = uniform_cat_2d
        n_rp, n_pi = len(rp_bins) - 1, len(pi_bins) - 1
        result = compute_2pcf(coords, sv_ids, rp_bins, pi_bins, box, k, k)
        assert result["xi"].shape == (n_rp, n_pi)
        assert result["wp"].shape == (n_rp,)
        assert result["dd_auto"].shape == (n_rp, n_pi)
        assert result["dd_cross"].shape == (n_rp, n_pi)
        assert result["dd_corr"].shape == (n_rp, n_pi)
        assert result["rr"].shape == (n_rp, n_pi)

    def test_wp_equals_twice_pi_integral_of_xi(self, uniform_cat_2d):
        """w_p(r_p) = 2 ∫ ξ(r_p, π) dπ — the defining relation."""
        coords, sv_ids, box, rp_bins, pi_bins, k = uniform_cat_2d
        result = compute_2pcf(coords, sv_ids, rp_bins, pi_bins, box, k, k)
        delta_pi = np.diff(pi_bins)
        expected_wp = 2.0 * np.sum(result["xi"] * delta_pi[np.newaxis, :], axis=1)
        assert np.allclose(result["wp"], expected_wp)

    def test_xi_equals_dd_corr_over_rr_minus_one(self, uniform_cat_2d):
        coords, sv_ids, box, rp_bins, pi_bins, k = uniform_cat_2d
        result = compute_2pcf(coords, sv_ids, rp_bins, pi_bins, box, k, k)
        assert np.allclose(result["xi"], result["dd_corr"] / result["rr"] - 1.0)

    def test_dd_corr_formula(self, uniform_cat_2d):
        """dd_corr = α · dd_auto + β · dd_cross with α = k/m, β = k(k-1)/[m(m-1)]."""
        coords, sv_ids, box, rp_bins, pi_bins, k = uniform_cat_2d
        m = 3
        result = compute_2pcf(coords, sv_ids, rp_bins, pi_bins, box, k, m)
        alpha = k / m
        beta = k * (k - 1) / (m * (m - 1))
        expected = alpha * result["dd_auto"] + beta * result["dd_cross"]
        assert np.allclose(result["dd_corr"], expected)

    def test_m_equals_k_alpha_beta_one(self, uniform_cat_2d):
        """For m = k: α = β = 1, dd_corr = dd_auto + dd_cross."""
        coords, sv_ids, box, rp_bins, pi_bins, k = uniform_cat_2d
        result = compute_2pcf(coords, sv_ids, rp_bins, pi_bins, box, k, k)
        assert np.allclose(result["dd_corr"], result["dd_auto"] + result["dd_cross"])

    def test_invalid_m_exceeds_k(self, uniform_cat_2d):
        coords, sv_ids, box, rp_bins, pi_bins, k = uniform_cat_2d
        with pytest.raises(ValueError):
            compute_2pcf(coords, sv_ids, rp_bins, pi_bins, box, k, k + 1)

    def test_invalid_m_is_zero(self, uniform_cat_2d):
        coords, sv_ids, box, rp_bins, pi_bins, k = uniform_cat_2d
        with pytest.raises(ValueError):
            compute_2pcf(coords, sv_ids, rp_bins, pi_bins, box, k, 0)

    def test_n_total_default_inferred(self, uniform_cat_2d):
        """Without n_total, it defaults to n_selected * k/m."""
        coords, sv_ids, box, rp_bins, pi_bins, k = uniform_cat_2d
        m = 2
        result = compute_2pcf(coords, sv_ids, rp_bins, pi_bins, box, k, m)
        n_sel = len(coords)
        n_total_inferred = int(round(n_sel * k / m))
        expected_rr = analytic_rr(rp_bins, pi_bins, box, n_total_inferred)
        assert np.allclose(result["rr"], expected_rr)

    def test_n_total_explicit_overrides_default(self, uniform_cat_2d):
        """Supplying n_total explicitly overrides the default inference."""
        coords, sv_ids, box, rp_bins, pi_bins, k = uniform_cat_2d
        m = 2
        n_explicit = 50_000
        result = compute_2pcf(coords, sv_ids, rp_bins, pi_bins, box, k, m,
                              n_total=n_explicit)
        expected_rr = analytic_rr(rp_bins, pi_bins, box, n_explicit)
        assert np.allclose(result["rr"], expected_rr)

    def test_uniform_random_xi_near_zero(self, uniform_cat_2d):
        """ξ(r_p, π) ≈ 0 for a Poisson field in well-populated bins."""
        coords, sv_ids, box, rp_bins, pi_bins, k = uniform_cat_2d
        result = compute_2pcf(coords, sv_ids, rp_bins, pi_bins, box, k, k)
        populated = result["rr"] > 100
        assert populated.any(), "no populated 2D bins — increase N or widen bins"
        assert np.all(np.abs(result["xi"][populated]) < 0.10)
