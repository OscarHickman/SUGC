"""
Unit tests for the redshift-space (s, μ) estimators.

Coverage
--------
count_pairs_smu   pair counting in (s, μ) bins, auto/cross routing, periodic BCs
analytic_rr_smu   spherical-shell × uniform-μ RR formula
compute_xi_smu    full RSD pipeline: weights, Legendre projections, physics

Physics check (ξ ≈ 0)
---------------------
compute_xi_smu is tested on a uniform Poisson field with no peculiar velocities,
where the true correlation is zero. We demand |ξ(s,μ)| < 0.05 in well-populated
bins (RR > 500 with n_mu_bins=10), which excludes the noisy inner s-shell.
"""

import numpy as np
import pytest

from scope._scope import count_pairs_smu
from scope import analytic_rr_smu, compute_xi_smu


# ─── Brute-force reference ────────────────────────────────────────────────────

def _bf_smu(coords, sv_ids, s_bins, n_mu_bins, mu_max, box):
    """O(N²) (s, μ) pair counter — correct by construction, used to validate Rust code."""
    n = len(coords)
    s_sq_bins = s_bins ** 2
    n_s = len(s_bins) - 1
    auto = np.zeros((n_s, n_mu_bins))
    cross = np.zeros((n_s, n_mu_bins))
    for i in range(n):
        d = coords[i + 1:] - coords[i]
        d -= box * np.round(d / box)
        s_sq = (d ** 2).sum(axis=1)
        dz_abs = np.abs(d[:, 2])
        same = sv_ids[i + 1:] == sv_ids[i]
        for k_s in range(n_s):
            in_s = (s_sq >= s_sq_bins[k_s]) & (s_sq < s_sq_bins[k_s + 1])
            for idx in np.where(in_s)[0]:
                s = np.sqrt(s_sq[idx])
                mu = dz_abs[idx] / s
                if mu >= mu_max:
                    continue
                imu = min(int(mu / mu_max * n_mu_bins), n_mu_bins - 1)
                if same[idx]:
                    auto[k_s, imu] += 1
                else:
                    cross[k_s, imu] += 1
    return auto, cross


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_cat_smu():
    """50-particle catalogue for exact (s, μ) brute-force comparison."""
    rng = np.random.default_rng(4)
    n, box = 50, 60.0
    coords = rng.uniform(0, box, (n, 3)).astype(np.float64)
    sv_ids = rng.integers(0, 3, n).astype(np.int32)
    s_bins = np.array([1.0, 8.0, 20.0, 45.0])
    n_mu_bins, mu_max = 5, 1.0
    return coords, sv_ids, box, s_bins, n_mu_bins, mu_max


@pytest.fixture(scope="module")
def uniform_cat_smu():
    """4 000-particle uniform field for ξ ≈ 0 physics check (s, μ)."""
    rng = np.random.default_rng(6)
    n, box, k = 4000, 300.0, 4
    coords = rng.uniform(0, box, (n, 3)).astype(np.float64)
    sv_ids = rng.integers(0, k, n).astype(np.int32)
    s_bins = np.array([5.0, 15.0, 40.0, 100.0])
    return coords, sv_ids, box, s_bins, k


# ─── count_pairs_smu ──────────────────────────────────────────────────────────

class TestCountPairsSmu:
    def test_matches_brute_force_auto(self, small_cat_smu):
        coords, sv_ids, box, s_bins, n_mu_bins, mu_max = small_cat_smu
        auto, _ = count_pairs_smu(coords, sv_ids, s_bins, n_mu_bins, mu_max, box)
        bf_auto, _ = _bf_smu(coords, sv_ids, s_bins, n_mu_bins, mu_max, box)
        assert np.allclose(auto, bf_auto)

    def test_matches_brute_force_cross(self, small_cat_smu):
        coords, sv_ids, box, s_bins, n_mu_bins, mu_max = small_cat_smu
        _, cross = count_pairs_smu(coords, sv_ids, s_bins, n_mu_bins, mu_max, box)
        _, bf_cross = _bf_smu(coords, sv_ids, s_bins, n_mu_bins, mu_max, box)
        assert np.allclose(cross, bf_cross)

    def test_auto_plus_cross_equals_total(self, small_cat_smu):
        """auto + cross must equal every pair in range, regardless of sv routing."""
        coords, sv_ids, box, s_bins, n_mu_bins, mu_max = small_cat_smu
        auto, cross = count_pairs_smu(coords, sv_ids, s_bins, n_mu_bins, mu_max, box)
        sv_ones = np.zeros(len(coords), dtype=np.int32)
        total, _ = count_pairs_smu(coords, sv_ones, s_bins, n_mu_bins, mu_max, box)
        assert np.allclose(auto + cross, total)

    def test_all_same_subvol_zero_cross(self, small_cat_smu):
        coords, _, box, s_bins, n_mu_bins, mu_max = small_cat_smu
        sv_ids = np.zeros(len(coords), dtype=np.int32)
        auto, cross = count_pairs_smu(coords, sv_ids, s_bins, n_mu_bins, mu_max, box)
        assert np.all(cross == 0.0)

    def test_all_distinct_subvols_zero_auto(self, small_cat_smu):
        coords, _, box, s_bins, n_mu_bins, mu_max = small_cat_smu
        sv_ids = np.arange(len(coords), dtype=np.int32)
        auto, cross = count_pairs_smu(coords, sv_ids, s_bins, n_mu_bins, mu_max, box)
        assert np.all(auto == 0.0)

    def test_output_shape(self, small_cat_smu):
        coords, sv_ids, box, s_bins, n_mu_bins, mu_max = small_cat_smu
        auto, cross = count_pairs_smu(coords, sv_ids, s_bins, n_mu_bins, mu_max, box)
        n_s = len(s_bins) - 1
        assert auto.shape == (n_s, n_mu_bins)
        assert cross.shape == (n_s, n_mu_bins)

    def test_counts_nonnegative(self, small_cat_smu):
        coords, sv_ids, box, s_bins, n_mu_bins, mu_max = small_cat_smu
        auto, cross = count_pairs_smu(coords, sv_ids, s_bins, n_mu_bins, mu_max, box)
        assert np.all(auto >= 0) and np.all(cross >= 0)

    def test_counts_are_integer_valued(self, small_cat_smu):
        coords, sv_ids, box, s_bins, n_mu_bins, mu_max = small_cat_smu
        auto, cross = count_pairs_smu(coords, sv_ids, s_bins, n_mu_bins, mu_max, box)
        assert np.all(auto == np.floor(auto))
        assert np.all(cross == np.floor(cross))

    def test_pair_at_known_s_mu_lands_in_correct_bin(self):
        """Pair at (s=5, μ=0.8) contributes exactly 1 count to the expected bin."""
        box = 200.0
        # dx=3, dy=0, dz=4 → s=5, μ=4/5=0.8
        coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 4.0]], dtype=np.float64)
        sv_ids = np.array([0, 1], dtype=np.int32)
        s_bins = np.array([1.0, 10.0])
        n_mu_bins, mu_max = 5, 1.0
        auto, cross = count_pairs_smu(coords, sv_ids, s_bins, n_mu_bins, mu_max, box)
        # s=5 → s-bin 0; μ=0.8 → imu = floor(0.8*5) = 4
        assert cross[0, 4] == 1.0
        assert np.sum(auto) + np.sum(cross) == 1.0

    def test_periodic_boundary_los(self):
        """Pair straddling the box boundary along z is counted with the wrapped separation."""
        box = 100.0
        # Unwrapped dz=95, wrapped: 95-100=-5 → |dz|=5; dy=3 → s²=9+25=34, s≈5.83, μ≈0.857
        coords = np.array([[50.0, 50.0, 2.0], [50.0, 53.0, 97.0]], dtype=np.float64)
        sv_ids = np.array([0, 0], dtype=np.int32)
        s_bins = np.array([1.0, 10.0])
        n_mu_bins, mu_max = 5, 1.0
        auto, cross = count_pairs_smu(coords, sv_ids, s_bins, n_mu_bins, mu_max, box)
        assert np.sum(auto) == 1.0   # pair found via periodic wrapping
        assert np.sum(cross) == 0.0

    def test_mu_at_max_excluded(self):
        """A pure LOS pair (μ = 1.0 = mu_max) is not counted."""
        box = 200.0
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]], dtype=np.float64)
        sv_ids = np.array([0, 0], dtype=np.int32)
        s_bins = np.array([1.0, 10.0])
        n_mu_bins, mu_max = 5, 1.0
        auto, cross = count_pairs_smu(coords, sv_ids, s_bins, n_mu_bins, mu_max, box)
        assert np.sum(auto) == 0.0
        assert np.sum(cross) == 0.0

    def test_ordering_invariant(self, small_cat_smu):
        """Shuffling particle order must not change pair counts."""
        coords, sv_ids, box, s_bins, n_mu_bins, mu_max = small_cat_smu
        rng = np.random.default_rng(13)
        perm = rng.permutation(len(coords))
        auto1, cross1 = count_pairs_smu(coords, sv_ids, s_bins, n_mu_bins, mu_max, box)
        auto2, cross2 = count_pairs_smu(coords[perm], sv_ids[perm], s_bins, n_mu_bins, mu_max, box)
        assert np.allclose(auto1, auto2)
        assert np.allclose(cross1, cross2)


# ─── analytic_rr_smu ──────────────────────────────────────────────────────────

class TestAnalyticRRSmu:
    def test_matches_formula(self):
        """RR(s, μ) = N(N-1)/2 · V_shell(s) · Δμ / V_box."""
        n, box = 1000, 200.0
        s_bins = np.array([2.0, 8.0, 25.0])
        n_mu_bins, mu_max = 4, 1.0
        rr = analytic_rr_smu(s_bins, mu_max, n_mu_bins, box, n)
        v_shell = (4 * np.pi / 3) * (s_bins[1:] ** 3 - s_bins[:-1] ** 3)
        dmu = mu_max / n_mu_bins
        prefactor = n * (n - 1) / (2 * box ** 3)
        for i_s in range(len(s_bins) - 1):
            expected = prefactor * v_shell[i_s] * dmu
            assert np.allclose(rr[i_s], expected)

    def test_output_shape(self):
        s_bins = np.array([1.0, 5.0, 15.0, 40.0])
        rr = analytic_rr_smu(s_bins, 1.0, 10, 200.0, 500)
        assert rr.shape == (3, 10)

    def test_nonnegative(self):
        s_bins = np.array([1.0, 5.0, 15.0, 40.0])
        rr = analytic_rr_smu(s_bins, 1.0, 10, 200.0, 500)
        assert np.all(rr >= 0)

    def test_uniform_mu_bins(self):
        """All μ bins within a given s-shell have equal RR for uniform random positions."""
        s_bins = np.array([5.0, 15.0])
        rr = analytic_rr_smu(s_bins, 1.0, 8, 200.0, 500)
        assert np.allclose(rr[0], rr[0, 0])

    def test_scales_quadratically_with_n(self):
        s_bins = np.array([1.0, 10.0])
        rr100 = analytic_rr_smu(s_bins, 1.0, 5, 100.0, 100)
        rr200 = analytic_rr_smu(s_bins, 1.0, 5, 100.0, 200)
        assert np.isclose(rr200[0, 0] / rr100[0, 0], 200 * 199 / (100 * 99))


# ─── compute_xi_smu ───────────────────────────────────────────────────────────

class TestComputeXiSmu:
    def test_output_keys(self, small_cat_smu):
        coords, sv_ids, box, s_bins, _, _ = small_cat_smu
        result = compute_xi_smu(coords, sv_ids, s_bins, box, 3, 3)
        expected = {"xi_smu", "xi0", "xi2", "dd_auto", "dd_cross", "dd_corr",
                    "rr", "s_mid", "mu_mid"}
        assert expected <= set(result)

    def test_output_shapes(self, small_cat_smu):
        coords, sv_ids, box, s_bins, _, _ = small_cat_smu
        n_s = len(s_bins) - 1
        n_mu = 20
        result = compute_xi_smu(coords, sv_ids, s_bins, box, 3, 3, n_mu_bins=n_mu)
        assert result["xi_smu"].shape   == (n_s, n_mu)
        assert result["xi0"].shape      == (n_s,)
        assert result["xi2"].shape      == (n_s,)
        assert result["dd_auto"].shape  == (n_s, n_mu)
        assert result["dd_cross"].shape == (n_s, n_mu)
        assert result["dd_corr"].shape  == (n_s, n_mu)
        assert result["rr"].shape       == (n_s, n_mu)
        assert result["s_mid"].shape    == (n_s,)
        assert result["mu_mid"].shape   == (n_mu,)

    def test_s_mid_is_geometric_mean(self, small_cat_smu):
        coords, sv_ids, box, s_bins, _, _ = small_cat_smu
        result = compute_xi_smu(coords, sv_ids, s_bins, box, 3, 3)
        assert np.allclose(result["s_mid"], np.sqrt(s_bins[:-1] * s_bins[1:]))

    def test_mu_mid_bin_centres(self, small_cat_smu):
        coords, sv_ids, box, s_bins, _, _ = small_cat_smu
        n_mu, mu_max = 10, 1.0
        result = compute_xi_smu(coords, sv_ids, s_bins, box, 3, 3,
                                n_mu_bins=n_mu, mu_max=mu_max)
        dmu = mu_max / n_mu
        expected = np.linspace(dmu / 2, mu_max - dmu / 2, n_mu)
        assert np.allclose(result["mu_mid"], expected)

    def test_dd_corr_formula(self, small_cat_smu):
        """dd_corr = α · dd_auto + β · dd_cross, with α = m/k, β = m(k-1)/[k(m-1)]."""
        coords, sv_ids, box, s_bins, _, _ = small_cat_smu
        k, m = 3, 2
        result = compute_xi_smu(coords, sv_ids, s_bins, box, k, m)
        alpha = m / k
        beta = m * (k - 1) / (k * (m - 1))
        expected = alpha * result["dd_auto"] + beta * result["dd_cross"]
        assert np.allclose(result["dd_corr"], expected)

    def test_xi_smu_natural_estimator(self, small_cat_smu):
        """ξ(s, μ) = DD_corr / RR − 1 in populated bins."""
        coords, sv_ids, box, s_bins, _, _ = small_cat_smu
        result = compute_xi_smu(coords, sv_ids, s_bins, box, 3, 3)
        populated = result["rr"] > 0
        assert np.allclose(
            result["xi_smu"][populated],
            (result["dd_corr"] / result["rr"] - 1.0)[populated],
        )

    def test_m_equals_k_gives_alpha_beta_one(self, small_cat_smu):
        """For m = k: α = β = 1, so dd_corr = dd_auto + dd_cross."""
        coords, sv_ids, box, s_bins, _, _ = small_cat_smu
        k = 3
        result = compute_xi_smu(coords, sv_ids, s_bins, box, k, k)
        assert np.allclose(result["dd_corr"], result["dd_auto"] + result["dd_cross"])

    def test_m_equals_1_zero_cross(self):
        """For m = 1: β = 0; dd_cross = 0 and dd_corr = (1/k) · dd_auto."""
        rng = np.random.default_rng(7)
        n, box, k = 80, 100.0, 5
        coords = rng.uniform(0, box, (n, 3)).astype(np.float64)
        sv_ids = np.zeros(n, dtype=np.int32)
        s_bins = np.array([1.0, 10.0, 40.0])
        result = compute_xi_smu(coords, sv_ids, s_bins, box, k, 1)
        assert np.all(result["dd_cross"] == 0.0)
        assert np.allclose(result["dd_corr"], (1 / k) * result["dd_auto"])

    def test_invalid_m_exceeds_k(self, small_cat_smu):
        coords, sv_ids, box, s_bins, _, _ = small_cat_smu
        with pytest.raises(ValueError):
            compute_xi_smu(coords, sv_ids, s_bins, box, n_subvols=3, n_subvols_selected=4)

    def test_invalid_m_is_zero(self, small_cat_smu):
        coords, sv_ids, box, s_bins, _, _ = small_cat_smu
        with pytest.raises(ValueError):
            compute_xi_smu(coords, sv_ids, s_bins, box, n_subvols=3, n_subvols_selected=0)

    def test_xi0_matches_legendre_projection(self, small_cat_smu):
        """ξ₀(s) = ∫ ξ(s, μ) dμ — monopole projection formula."""
        coords, sv_ids, box, s_bins, _, _ = small_cat_smu
        n_mu, mu_max = 20, 1.0
        result = compute_xi_smu(coords, sv_ids, s_bins, box, 3, 3,
                                n_mu_bins=n_mu, mu_max=mu_max)
        dmu = mu_max / n_mu
        expected = np.nansum(result["xi_smu"] * dmu, axis=1)
        assert np.allclose(result["xi0"], expected, equal_nan=True)

    def test_xi2_matches_legendre_projection(self, small_cat_smu):
        """ξ₂(s) = 5 ∫ ξ(s, μ) L₂(μ) dμ — quadrupole projection formula."""
        coords, sv_ids, box, s_bins, _, _ = small_cat_smu
        n_mu, mu_max = 20, 1.0
        result = compute_xi_smu(coords, sv_ids, s_bins, box, 3, 3,
                                n_mu_bins=n_mu, mu_max=mu_max)
        dmu = mu_max / n_mu
        mu_mid = result["mu_mid"]
        L2 = 0.5 * (3 * mu_mid ** 2 - 1)
        expected = 5 * np.nansum(result["xi_smu"] * L2[np.newaxis, :] * dmu, axis=1)
        assert np.allclose(result["xi2"], expected, equal_nan=True)

    def test_uniform_random_xi_near_zero(self, uniform_cat_smu):
        """ξ(s, μ) ≈ 0 for a Poisson field in well-populated bins (CLAUDE.md criterion).

        Uses n_mu_bins=10 and rr > 500 so that only bins with σ_ξ < 0.02
        are tested (the inner s-shell has too few pairs per μ-bin at n_mu=20).
        """
        coords, sv_ids, box, s_bins, k = uniform_cat_smu
        result = compute_xi_smu(coords, sv_ids, s_bins, box, k, k, n_mu_bins=10)
        populated = result["rr"] > 500
        assert populated.any(), "no populated (s, μ) bins — increase N or widen bins"
        assert np.all(np.abs(result["xi_smu"][populated]) < 0.05)
