"""
Unit tests for the legacy 2-D (r_p, π) estimators.

Coverage
--------
count_pairs_2d   pair counting in (r_p, π) bins, auto/cross routing, periodic BCs
analytic_rr      annular-cylinder RR formula
compute_2pcf     full 2D pipeline: weights, w_p integral, physics

Physics check (ξ ≈ 0)
---------------------
compute_2pcf is tested on a uniform Poisson field where the true correlation
is zero. We demand |ξ| < 0.10 in every well-populated bin (RR > 100).
"""

import numpy as np
import pytest

from scope._scope import count_pairs_2d
from scope import analytic_rr, compute_2pcf


# ─── Brute-force reference ────────────────────────────────────────────────────

def _bf_2d(coords, sv_ids, rp_bins, pi_bins, box):
    """O(N²) 2D pair counter — correct by construction, used to validate Rust code."""
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


# ─── Fixtures ─────────────────────────────────────────────────────────────────

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
def uniform_cat_2d():
    """3 000-particle uniform field for ξ ≈ 0 physics check (2D)."""
    rng = np.random.default_rng(3)
    n, box, k = 3000, 200.0, 4
    coords = rng.uniform(0, box, (n, 3)).astype(np.float64)
    sv_ids = rng.integers(0, k, n).astype(np.int32)
    rp_bins = np.array([1.0, 5.0, 15.0, 40.0])
    pi_bins = np.array([0.0, 10.0, 30.0, 60.0])
    return coords, sv_ids, box, rp_bins, pi_bins, k


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
