"""
Unit tests for the 1-D real-space estimators.

Coverage
--------
count_pairs_1d   pair counting, auto/cross routing, periodic BCs, edge cases
analytic_rr_1d   shell-volume RR formula
compute_xi       full 1D pipeline: weights, output contract, physics

Physics check (ξ ≈ 0)
---------------------
compute_xi is tested on a uniform Poisson field where the true correlation
is zero. With enough particles the Poisson noise on ξ is small; we demand
|ξ| < 0.05 in every well-populated bin (RR > 100), consistent with the
CLAUDE.md acceptance criterion.
"""

import numpy as np
import pytest
from sugc._sugc import count_pairs_1d

from sugc import analytic_rr_1d, compute_xi

# ─── Brute-force reference ────────────────────────────────────────────────────


def _bf_1d(coords, part_ids, r_bins, box):
    """O(N²) pair counter — correct by construction, used to validate Rust code."""
    n = len(coords)
    r_sq_bins = r_bins**2
    n_bins = len(r_bins) - 1
    auto = np.zeros(n_bins)
    cross = np.zeros(n_bins)
    for i in range(n):
        d = coords[i + 1 :] - coords[i]
        d -= box * np.round(d / box)  # minimum-image
        r_sq = (d**2).sum(axis=1)
        same = part_ids[i + 1 :] == part_ids[i]
        for k in range(n_bins):
            in_bin = (r_sq >= r_sq_bins[k]) & (r_sq < r_sq_bins[k + 1])
            auto[k] += np.count_nonzero(in_bin & same)
            cross[k] += np.count_nonzero(in_bin & ~same)
    return auto, cross


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def small_cat():
    """60-particle catalogue: small enough for exact brute-force comparison."""
    rng = np.random.default_rng(1)
    n, box = 60, 80.0
    coords = rng.uniform(0, box, (n, 3)).astype(np.float64)
    part_ids = rng.integers(0, 4, n).astype(np.int32)
    r_bins = np.array([1.0, 5.0, 12.0, 28.0, 55.0])
    return coords, part_ids, box, r_bins


@pytest.fixture(scope="module")
def uniform_cat_1d():
    """4 000-particle uniform field for ξ ≈ 0 physics check."""
    rng = np.random.default_rng(42)
    n, box, k = 4000, 300.0, 4
    coords = rng.uniform(0, box, (n, 3)).astype(np.float64)
    part_ids = rng.integers(0, k, n).astype(np.int32)
    r_bins = np.array([5.0, 15.0, 40.0, 100.0])
    return coords, part_ids, box, r_bins, k


# ─── count_pairs_1d ───────────────────────────────────────────────────────────


class TestCountPairs1D:
    def test_matches_brute_force_auto(self, small_cat):
        coords, part_ids, box, r_bins = small_cat
        auto, _ = count_pairs_1d(coords, part_ids, r_bins, box)
        bf_auto, _ = _bf_1d(coords, part_ids, r_bins, box)
        assert np.allclose(auto, bf_auto)

    def test_matches_brute_force_cross(self, small_cat):
        coords, part_ids, box, r_bins = small_cat
        _, cross = count_pairs_1d(coords, part_ids, r_bins, box)
        _, bf_cross = _bf_1d(coords, part_ids, r_bins, box)
        assert np.allclose(cross, bf_cross)

    def test_auto_plus_cross_equals_total(self, small_cat):
        """auto + cross must equal every pair in range, regardless of sv routing."""
        coords, part_ids, box, r_bins = small_cat
        auto, cross = count_pairs_1d(coords, part_ids, r_bins, box)
        sv_ones = np.zeros(len(coords), dtype=np.int32)
        total, _ = count_pairs_1d(coords, sv_ones, r_bins, box)
        assert np.allclose(auto + cross, total)

    def test_all_same_partition_zero_cross(self, small_cat):
        """With a single partition label, every pair is auto; cross must be zero."""
        coords, _, box, r_bins = small_cat
        part_ids = np.zeros(len(coords), dtype=np.int32)
        auto, cross = count_pairs_1d(coords, part_ids, r_bins, box)
        assert np.all(cross == 0.0)

    def test_all_distinct_partitions_zero_auto(self, small_cat):
        """With every particle in its own partition, auto must be zero."""
        coords, _, box, r_bins = small_cat
        part_ids = np.arange(len(coords), dtype=np.int32)
        auto, cross = count_pairs_1d(coords, part_ids, r_bins, box)
        assert np.all(auto == 0.0)

    def test_periodic_boundary(self):
        """A pair straddling the periodic boundary is counted in the correct bin."""
        box = 100.0
        # True minimum-image separation = 2 Mpc/h (across the boundary)
        coords = np.array([[1.0, 50.0, 50.0], [99.0, 50.0, 50.0]], dtype=np.float64)
        part_ids = np.array([0, 0], dtype=np.int32)
        r_bins = np.array([1.0, 3.0, 10.0])
        auto, cross = count_pairs_1d(coords, part_ids, r_bins, box)
        assert auto[0] == 1.0  # bin [1, 3) contains the pair
        assert auto[1] == 0.0

    def test_pair_beyond_rmax_excluded(self):
        box = 200.0
        coords = np.array([[0.0, 0.0, 0.0], [60.0, 0.0, 0.0]], dtype=np.float64)
        part_ids = np.array([0, 1], dtype=np.int32)
        r_bins = np.array([1.0, 50.0])  # r_max = 50 < 60
        auto, cross = count_pairs_1d(coords, part_ids, r_bins, box)
        assert auto[0] == 0.0 and cross[0] == 0.0

    def test_pair_below_rmin_excluded(self):
        box = 200.0
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        part_ids = np.array([0, 0], dtype=np.int32)
        r_bins = np.array([5.0, 50.0])  # r_min = 5 > 1
        auto, cross = count_pairs_1d(coords, part_ids, r_bins, box)
        assert auto[0] == 0.0

    def test_two_particles_lands_in_correct_bin(self):
        """Pair at known r contributes exactly 1 count to the expected bin."""
        box = 200.0
        coords = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float64)
        part_ids = np.array([0, 1], dtype=np.int32)
        r_bins = np.array([1.0, 5.0, 15.0, 50.0])
        auto, cross = count_pairs_1d(coords, part_ids, r_bins, box)
        # r = 10 → bin index 1, which is the [5, 15) bin
        assert cross[1] == 1.0
        assert auto[1] == 0.0
        assert np.sum(auto) + np.sum(cross) == 1.0

    def test_ordering_invariant(self, small_cat):
        """Shuffling particle order must not change pair counts."""
        coords, part_ids, box, r_bins = small_cat
        rng = np.random.default_rng(7)
        perm = rng.permutation(len(coords))
        auto1, cross1 = count_pairs_1d(coords, part_ids, r_bins, box)
        auto2, cross2 = count_pairs_1d(coords[perm], part_ids[perm], r_bins, box)
        assert np.allclose(auto1, auto2)
        assert np.allclose(cross1, cross2)

    def test_output_shape(self, small_cat):
        coords, part_ids, box, r_bins = small_cat
        auto, cross = count_pairs_1d(coords, part_ids, r_bins, box)
        n_bins = len(r_bins) - 1
        assert auto.shape == (n_bins,)
        assert cross.shape == (n_bins,)

    def test_counts_nonnegative(self, small_cat):
        coords, part_ids, box, r_bins = small_cat
        auto, cross = count_pairs_1d(coords, part_ids, r_bins, box)
        assert np.all(auto >= 0) and np.all(cross >= 0)

    def test_counts_are_integer_valued(self, small_cat):
        """Pair counts are whole numbers stored as float64."""
        coords, part_ids, box, r_bins = small_cat
        auto, cross = count_pairs_1d(coords, part_ids, r_bins, box)
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
        expected = n * (n - 1) / 2 * v_shell / box**3
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
        """Wider bin must contain more RR than a narrower one at the same inner edge."""
        rr_narrow = analytic_rr_1d(np.array([5.0, 10.0]), 100.0, 500)
        rr_wide = analytic_rr_1d(np.array([5.0, 15.0]), 100.0, 500)
        assert rr_wide[0] > rr_narrow[0]

    def test_monotone_bins_give_more_rr_at_larger_r(self):
        """Outer shells have more volume, hence more RR for equal-width bins."""
        r_bins = np.array([5.0, 10.0, 15.0])
        rr = analytic_rr_1d(r_bins, 200.0, 500)
        assert rr[1] > rr[0]


# ─── compute_xi ───────────────────────────────────────────────────────────────


class TestComputeXi:
    def test_output_keys(self, small_cat):
        coords, part_ids, box, r_bins = small_cat
        result = compute_xi(coords, part_ids, r_bins, box, 4, 4)
        assert {"xi", "dd_auto", "dd_cross", "dd_corr", "rr", "r_mid"} <= set(result)

    def test_output_shapes(self, small_cat):
        coords, part_ids, box, r_bins = small_cat
        n_bins = len(r_bins) - 1
        result = compute_xi(coords, part_ids, r_bins, box, 4, 4)
        for key in ("xi", "dd_auto", "dd_cross", "dd_corr", "rr", "r_mid"):
            assert result[key].shape == (n_bins,), f"wrong shape for '{key}'"

    def test_r_mid_is_geometric_mean(self, small_cat):
        coords, part_ids, box, r_bins = small_cat
        result = compute_xi(coords, part_ids, r_bins, box, 4, 4)
        assert np.allclose(result["r_mid"], np.sqrt(r_bins[:-1] * r_bins[1:]))

    def test_dd_corr_formula(self, small_cat):
        """dd_corr = α · dd_auto + β · dd_cross, with α = m/k, β = m(k-1)/[k(m-1)]."""
        coords, part_ids, box, r_bins = small_cat
        k, m = 4, 3
        result = compute_xi(coords, part_ids, r_bins, box, k, m)
        alpha = m / k
        beta = m * (k - 1) / (k * (m - 1))
        expected = alpha * result["dd_auto"] + beta * result["dd_cross"]
        assert np.allclose(result["dd_corr"], expected)

    def test_xi_equals_dd_corr_over_rr_minus_one(self, small_cat):
        """ξ = DD_corr / RR − 1."""
        coords, part_ids, box, r_bins = small_cat
        result = compute_xi(coords, part_ids, r_bins, box, 4, 4)
        assert np.allclose(result["xi"], result["dd_corr"] / result["rr"] - 1.0)

    def test_m_equals_k_gives_alpha_beta_one(self, small_cat):
        """For m = k: α = β = 1, so dd_corr = dd_auto + dd_cross."""
        coords, part_ids, box, r_bins = small_cat
        k = 4
        result = compute_xi(coords, part_ids, r_bins, box, k, k)
        assert np.allclose(result["dd_corr"], result["dd_auto"] + result["dd_cross"])

    def test_m_equals_1_zero_cross_contribution(self):
        """For m = 1: β = 0, all pairs are auto; dd_corr = (1/k) · dd_auto."""
        rng = np.random.default_rng(5)
        n, box, k = 80, 100.0, 5
        coords = rng.uniform(0, box, (n, 3)).astype(np.float64)
        part_ids = np.zeros(n, dtype=np.int32)  # single partition
        r_bins = np.array([1.0, 10.0, 40.0])
        result = compute_xi(coords, part_ids, r_bins, box, k, 1)
        assert np.all(result["dd_cross"] == 0.0)
        assert np.allclose(result["dd_corr"], (1 / k) * result["dd_auto"])

    def test_rr_is_positive(self, small_cat):
        coords, part_ids, box, r_bins = small_cat
        result = compute_xi(coords, part_ids, r_bins, box, 4, 4)
        assert np.all(result["rr"] > 0)

    def test_rr_uses_selected_n(self):
        """RR is computed from n_selected = len(coords), not the full catalogue size."""
        rng = np.random.default_rng(9)
        n, box, k = 100, 200.0, 10
        coords = rng.uniform(0, box, (n, 3)).astype(np.float64)
        part_ids = rng.integers(0, 2, n).astype(np.int32)  # m = 2 partitions selected
        r_bins = np.array([1.0, 20.0])
        result = compute_xi(coords, part_ids, r_bins, box, k, 2)
        expected_rr = analytic_rr_1d(r_bins, box, n)
        assert np.allclose(result["rr"], expected_rr)

    def test_invalid_m_exceeds_k(self, small_cat):
        coords, part_ids, box, r_bins = small_cat
        with pytest.raises(ValueError):
            compute_xi(
                coords, part_ids, r_bins, box, n_partitions=4, n_partitions_selected=5
            )

    def test_invalid_m_is_zero(self, small_cat):
        coords, part_ids, box, r_bins = small_cat
        with pytest.raises(ValueError):
            compute_xi(
                coords, part_ids, r_bins, box, n_partitions=4, n_partitions_selected=0
            )

    def test_alpha_varies_with_m(self):
        """Halving m halves α and therefore dd_corr_auto-part."""
        rng = np.random.default_rng(11)
        n, box = 80, 100.0
        coords = rng.uniform(0, box, (n, 3)).astype(np.float64)
        part_ids = rng.integers(0, 4, n).astype(np.int32)
        r_bins = np.array([1.0, 30.0])
        r2 = compute_xi(coords, part_ids, r_bins, box, 8, 2)
        r4 = compute_xi(coords, part_ids, r_bins, box, 8, 4)
        num = 2 / 8 * r2["dd_auto"][0] + 2 * 7 / (8 * 1) * r2["dd_cross"][0]
        den = 4 / 8 * r4["dd_auto"][0] + 4 * 7 / (8 * 3) * r4["dd_cross"][0]
        assert np.isclose(r2["dd_corr"][0] / r4["dd_corr"][0], num / den, rtol=1e-9)

    def test_uniform_random_xi_near_zero(self, uniform_cat_1d):
        """ξ ≈ 0 for a Poisson field in well-populated bins (CLAUDE.md criterion)."""
        coords, part_ids, box, r_bins, k = uniform_cat_1d
        result = compute_xi(coords, part_ids, r_bins, box, k, k)
        populated = result["rr"] > 100
        assert populated.any(), "no populated bins — increase N or widen bins"
        assert np.all(np.abs(result["xi"][populated]) < 0.05)
