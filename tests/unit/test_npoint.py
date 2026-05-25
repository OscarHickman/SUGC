import numpy as np
import pytest
from itertools import combinations
from scope import count_pairs_1d, count_npoint, compute_npcf

def brute_force_npoint(coords, subvol_ids, r_bins, box_size, n_order):
    """
    Python brute-force N-point counter for validation.
    O(N_gal^n_order * n_order^2)
    """
    n_gal = len(coords)
    n_r = len(r_bins) - 1
    t_by_s = np.zeros((n_order, n_r))
    t_total = np.zeros(n_r)
    half_box = box_size * 0.5
    
    for indices in combinations(range(n_gal), n_order):
        # Compute all pairwise distances in the tuple
        max_dist = 0.0
        for i, j in combinations(indices, 2):
            dx = coords[j] - coords[i]
            # Minimum image convention
            dx = dx - box_size * np.round(dx / box_size)
            dist = np.sqrt(np.sum(dx**2))
            if dist > max_dist:
                max_dist = dist
        
        # Binning
        if r_bins[0] <= max_dist < r_bins[-1]:
            ir = np.searchsorted(r_bins, max_dist, side='right') - 1
            
            # Count distinct subvolumes
            s = len(set(subvol_ids[list(indices)]))
            
            t_by_s[s-1, ir] += 1
            t_total[ir] += 1
            
    return t_by_s, t_total

def test_npoint_vs_pairs_1d():
    """For N=2, count_npoint must match count_pairs_1d DD_total."""
    n_gal = 100
    box_size = 100.0
    coords = np.random.uniform(0, box_size, (n_gal, 3)).astype(np.float64)
    subvol_ids = np.random.randint(0, 8, n_gal).astype(np.int32)
    r_bins = np.linspace(0.1, 10.0, 11).astype(np.float64)

    # count_pairs_1d
    dd_auto, dd_cross = count_pairs_1d(coords, subvol_ids, r_bins, box_size)
    dd_total_ref = dd_auto + dd_cross

    # count_npoint (N=2)
    t_by_s, t_total = count_npoint(coords, subvol_ids, r_bins, box_size, 2)

    np.testing.assert_allclose(t_total, dd_total_ref, rtol=1e-10)
    np.testing.assert_allclose(t_by_s[0], dd_auto, rtol=1e-10)
    np.testing.assert_allclose(t_by_s[1], dd_cross, rtol=1e-10)

def test_npoint_vs_brute_force_n3():
    """Compare Rust count_npoint with Python brute-force for N=3."""
    n_gal = 40
    box_size = 20.0
    coords = np.random.uniform(0, box_size, (n_gal, 3)).astype(np.float64)
    subvol_ids = np.random.randint(0, 3, n_gal).astype(np.int32)
    r_bins = np.linspace(0.5, 5.0, 5).astype(np.float64)
    
    t_by_s_rust, t_total_rust = count_npoint(coords, subvol_ids, r_bins, box_size, 3)
    t_by_s_brute, t_total_brute = brute_force_npoint(coords, subvol_ids, r_bins, box_size, 3)
    
    np.testing.assert_allclose(t_total_rust, t_total_brute, rtol=1e-10)
    np.testing.assert_allclose(t_by_s_rust, t_by_s_brute, rtol=1e-10)

def test_npoint_vs_brute_force_n4():
    """Compare Rust count_npoint with Python brute-force for N=4."""
    n_gal = 20
    box_size = 15.0
    coords = np.random.uniform(0, box_size, (n_gal, 3)).astype(np.float64)
    subvol_ids = np.random.randint(0, 4, n_gal).astype(np.int32)
    r_bins = np.linspace(0.5, 4.0, 4).astype(np.float64)
    
    t_by_s_rust, t_total_rust = count_npoint(coords, subvol_ids, r_bins, box_size, 4)
    t_by_s_brute, t_total_brute = brute_force_npoint(coords, subvol_ids, r_bins, box_size, 4)
    
    np.testing.assert_allclose(t_total_rust, t_total_brute, rtol=1e-10)
    np.testing.assert_allclose(t_by_s_rust, t_by_s_brute, rtol=1e-10)

def test_compute_npcf_full_box():
    """When m=k, t_corr must equal t_total."""
    n_gal = 200
    box_size = 50.0
    coords = np.random.uniform(0, box_size, (n_gal, 3)).astype(np.float64)
    subvol_ids = np.random.randint(0, 4, n_gal).astype(np.int32)
    r_bins = np.linspace(0.1, 5.0, 6).astype(np.float64)
    
    # m=k=4, N=3
    res = compute_npcf(coords, subvol_ids, r_bins, box_size, 4, 4, 3)
    
    np.testing.assert_allclose(res["t_corr"], res["t_total"], rtol=1e-12)
    # weights should all be 1.0 when m=k
    np.testing.assert_allclose(res["weights"], [1.0, 1.0, 1.0], rtol=1e-12)

def test_npoint_early_pruning():
    """Verify that tuples with max distance > r_max are correctly ignored."""
    box_size = 100.0
    # Two galaxies far apart, others close to the first one
    coords = np.array([
        [1.0, 1.0, 1.0],
        [1.5, 1.0, 1.0],
        [2.0, 1.0, 1.0],
        [50.0, 50.0, 50.0],
    ], dtype=np.float64)
    subvol_ids = np.zeros(4, dtype=np.int32)
    r_bins = np.array([0.1, 5.0], dtype=np.float64) # r_max = 5.0
    
    # N=3 Triplets:
    # (0, 1, 2): max=1.0 < 5.0 -> Count!
    # (0, 1, 3): max > 5.0 -> Pruned
    # (0, 2, 3): max > 5.0 -> Pruned
    # (1, 2, 3): max > 5.0 -> Pruned
    
    t_by_s, t_total = count_npoint(coords, subvol_ids, r_bins, box_size, 3)
    np.testing.assert_array_equal(t_total, [1])
