import numpy as np
import pytest

from sugc import compute_xi_smu, count_pairs_smu


def _bf_smu(coords, part_ids, s_bins, n_mu, mu_max, box):
    n = len(coords)
    n_s = len(s_bins) - 1
    auto = np.zeros((n_s, n_mu))
    cross = np.zeros((n_s, n_mu))
    s_sq_bins = s_bins**2
    half_box = box * 0.5

    for i in range(n):
        xi, yi, zi = coords[i]
        parti = part_ids[i]
        for j in range(i + 1, n):
            xj, yj, zj = coords[j]
            dx = xj - xi
            dy = yj - yi
            dz = zj - zi

            # Minimum-image convention matching Rust implementation
            if dx > half_box:
                dx -= box
            elif dx < -half_box:
                dx += box
            if dy > half_box:
                dy -= box
            elif dy < -half_box:
                dy += box
            if dz > half_box:
                dz -= box
            elif dz < -half_box:
                dz += box

            s_sq = dx * dx + dy * dy + dz * dz
            if s_sq < s_sq_bins[-1]:
                # Rust uses find_bin_squared (partition_point <= val)
                # s_sq_bins = [1, 100, 625, 2500]
                idx_s = np.searchsorted(s_sq_bins, s_sq, side="right") - 1
                if idx_s >= 0:
                    s = np.sqrt(s_sq)
                    mu = np.abs(dz) / s
                    if mu < mu_max:
                        # Rust uses ((mu / mu_max) * n_mu) as usize
                        idx_mu = int((mu / mu_max) * n_mu)
                        idx_mu = min(idx_mu, n_mu - 1)
                        if parti == part_ids[j]:
                            auto[idx_s, idx_mu] += 1
                        else:
                            cross[idx_s, idx_mu] += 1
    return auto, cross


@pytest.fixture
def small_cat_smu():
    np.random.seed(42)
    # Use fewer galaxies and smaller s_max to avoid cell list edge cases in dev
    n = 50
    box = 100.0
    coords = np.random.uniform(0, box, (n, 3))
    part_ids = np.random.randint(0, 3, n, dtype=np.int32)
    s_bins = np.array([1.0, 5.0, 10.0, 20.0])
    n_mu = 5
    mu_max = 1.0
    return coords, part_ids, box, s_bins, n_mu, mu_max


class TestCountPairsSmu:
    def test_matches_brute_force_auto(self, small_cat_smu):
        coords, part_ids, box, s_bins, n_mu, mu_max = small_cat_smu
        auto, _ = count_pairs_smu(coords, part_ids, s_bins, n_mu, mu_max, box)
        bf_auto, _ = _bf_smu(coords, part_ids, s_bins, n_mu, mu_max, box)
        np.testing.assert_allclose(auto, bf_auto)

    def test_matches_brute_force_cross(self, small_cat_smu):
        coords, part_ids, box, s_bins, n_mu, mu_max = small_cat_smu
        _, cross = count_pairs_smu(coords, part_ids, s_bins, n_mu, mu_max, box)
        _, bf_cross = _bf_smu(coords, part_ids, s_bins, n_mu, mu_max, box)
        np.testing.assert_allclose(cross, bf_cross)

    def test_invalid_m_exceeds_k(self, small_cat_smu):
        coords, part_ids, box, s_bins, n_mu, _ = small_cat_smu
        with pytest.raises(ValueError):
            compute_xi_smu(
                coords,
                part_ids,
                s_bins,
                box,
                n_partitions=3,
                n_partitions_selected=4,
                n_mu_bins=n_mu,
            )

    def test_invalid_m_is_zero(self, small_cat_smu):
        coords, part_ids, box, s_bins, n_mu, _ = small_cat_smu
        with pytest.raises(ValueError):
            compute_xi_smu(
                coords,
                part_ids,
                s_bins,
                box,
                n_partitions=3,
                n_partitions_selected=0,
                n_mu_bins=n_mu,
            )

    def test_output_shape(self, small_cat_smu):
        coords, part_ids, box, s_bins, n_mu, mu_max = small_cat_smu
        auto, cross = count_pairs_smu(coords, part_ids, s_bins, n_mu, mu_max, box)
        assert auto.shape == (len(s_bins) - 1, n_mu)
        assert cross.shape == (len(s_bins) - 1, n_mu)


class TestComputeXiSmu:
    def test_compute_xi_smu_returns_correct_keys(self, small_cat_smu):
        coords, part_ids, box, s_bins, n_mu, mu_max = small_cat_smu
        res = compute_xi_smu(
            coords,
            part_ids,
            s_bins,
            box,
            n_partitions=3,
            n_partitions_selected=2,
            n_mu_bins=n_mu,
            mu_max=mu_max,
        )
        expected_keys = {
            "dd_auto",
            "dd_cross",
            "dd_corr",
            "rr",
            "xi_grid",
            "xi_smu",
            "xi0",
            "xi2",
            "s_mid",
            "mu_mid",
        }
        assert expected_keys.issubset(res.keys())

        n_s = len(s_bins) - 1
        assert res["dd_auto"].shape == (n_s, n_mu)
        assert res["dd_cross"].shape == (n_s, n_mu)
        assert res["dd_corr"].shape == (n_s, n_mu)
        assert res["rr"].shape == (n_s, n_mu)
        assert res["xi_grid"].shape == (n_s, n_mu)
        assert res["xi_smu"].shape == (n_s, n_mu)
        assert res["xi0"].shape == (n_s,)
        assert res["xi2"].shape == (n_s,)
        assert res["s_mid"].shape == (n_s,)
        assert res["mu_mid"].shape == (n_mu,)

    def test_compute_xi_smu_values(self, small_cat_smu):
        # Verify that for m=k, dd_corr = dd_auto + dd_cross
        coords, part_ids, box, s_bins, n_mu, mu_max = small_cat_smu
        res = compute_xi_smu(
            coords,
            part_ids,
            s_bins,
            box,
            n_partitions=3,
            n_partitions_selected=3,
            n_mu_bins=n_mu,
            mu_max=mu_max,
        )
        assert np.allclose(res["dd_corr"], res["dd_auto"] + res["dd_cross"])

        # Verify relation between xi_grid, dd_corr, rr
        assert np.allclose(res["xi_grid"], res["dd_corr"] / res["rr"] - 1.0)
        assert np.allclose(res["xi_smu"], res["dd_corr"] / res["rr"] - 1.0)


