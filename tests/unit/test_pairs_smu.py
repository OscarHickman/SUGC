import numpy as np
import pytest
from sugc import count_pairs_smu, analytic_rr_smu, compute_xi_smu

def _bf_smu(coords, sv_ids, s_bins, n_mu, mu_max, box):
    n = len(coords)
    n_s = len(s_bins) - 1
    auto = np.zeros((n_s, n_mu))
    cross = np.zeros((n_s, n_mu))
    s_sq_bins = s_bins**2
    half_box = box * 0.5
    
    for i in range(n):
        xi, yi, zi = coords[i]
        svi = sv_ids[i]
        for j in range(i + 1, n):
            xj, yj, zj = coords[j]
            dx = xj - xi
            dy = yj - yi
            dz = zj - zi
            
            # Use same exact logic as Rust: if d > half_box { d -= box } else if d < -half_box { d += box }
            if dx >  half_box: dx -= box
            elif dx < -half_box: dx += box
            if dy >  half_box: dy -= box
            elif dy < -half_box: dy += box
            if dz >  half_box: dz -= box
            elif dz < -half_box: dz += box
            
            s_sq = dx*dx + dy*dy + dz*dz
            if s_sq < s_sq_bins[-1]:
                # Rust uses find_bin_squared (partition_point <= val)
                # s_sq_bins = [1, 100, 625, 2500]
                idx_s = np.searchsorted(s_sq_bins, s_sq, side='right') - 1
                if idx_s >= 0:
                    s = np.sqrt(s_sq)
                    mu = np.abs(dz) / s
                    if mu < mu_max:
                        # Rust uses ((mu / mu_max) * n_mu) as usize
                        idx_mu = int((mu / mu_max) * n_mu)
                        idx_mu = min(idx_mu, n_mu - 1)
                        if svi == sv_ids[j]:
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
    sv_ids = np.random.randint(0, 3, n, dtype=np.int32)
    s_bins = np.array([1.0, 5.0, 10.0, 20.0])
    n_mu = 5
    mu_max = 1.0
    return coords, sv_ids, box, s_bins, n_mu, mu_max

class TestCountPairsSmu:
    def test_matches_brute_force_auto(self, small_cat_smu):
        coords, sv_ids, box, s_bins, n_mu, mu_max = small_cat_smu
        auto, _ = count_pairs_smu(coords, sv_ids, s_bins, n_mu, mu_max, box)
        bf_auto, _ = _bf_smu(coords, sv_ids, s_bins, n_mu, mu_max, box)
        np.testing.assert_allclose(auto, bf_auto)

    def test_matches_brute_force_cross(self, small_cat_smu):
        coords, sv_ids, box, s_bins, n_mu, mu_max = small_cat_smu
        _, cross = count_pairs_smu(coords, sv_ids, s_bins, n_mu, mu_max, box)
        _, bf_cross = _bf_smu(coords, sv_ids, s_bins, n_mu, mu_max, box)
        np.testing.assert_allclose(cross, bf_cross)

    def test_invalid_m_exceeds_k(self, small_cat_smu):
        coords, sv_ids, box, s_bins, n_mu, _ = small_cat_smu
        with pytest.raises(ValueError):
            compute_xi_smu(coords, sv_ids, s_bins, box, n_subvols=3, n_subvols_selected=4, n_mu_bins=n_mu)

    def test_invalid_m_is_zero(self, small_cat_smu):
        coords, sv_ids, box, s_bins, n_mu, _ = small_cat_smu
        with pytest.raises(ValueError):
            compute_xi_smu(coords, sv_ids, s_bins, box, n_subvols=3, n_subvols_selected=0, n_mu_bins=n_mu)

    def test_output_shape(self, small_cat_smu):
        coords, sv_ids, box, s_bins, n_mu, mu_max = small_cat_smu
        auto, cross = count_pairs_smu(coords, sv_ids, s_bins, n_mu, mu_max, box)
        assert auto.shape == (len(s_bins)-1, n_mu)
        assert cross.shape == (len(s_bins)-1, n_mu)
