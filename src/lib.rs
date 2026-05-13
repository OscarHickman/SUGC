use numpy::{ndarray::{Array1, Array2}, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Find which bin `value` belongs to in a sorted ascending edge array.
/// Bin k is the half-open interval [edges[k], edges[k+1]).
/// Returns None if value is outside [edges[0], edges[last]).
#[inline]
fn find_bin(value: f64, edges: &[f64]) -> Option<usize> {
    if value < edges[0] || value >= *edges.last().unwrap() {
        return None;
    }
    // Binary search: partition_point returns the first index where edge > value,
    // so the bin index is one less. O(log n) — faster than linear for many bins.
    Some(edges.partition_point(|&e| e <= value) - 1)
}

/// Find which bin `value_sq` belongs to in a sorted ascending squared edge array.
/// Avoids sqrt() call in hot loops. Pre-computed squared edges via e → e².
/// Bin k is the half-open interval [edges_sq[k], edges_sq[k+1]).
#[inline]
fn find_bin_squared(value_sq: f64, edges_sq: &[f64]) -> Option<usize> {
    if value_sq < edges_sq[0] || value_sq >= *edges_sq.last().unwrap() {
        return None;
    }
    Some(edges_sq.partition_point(|&e| e <= value_sq) - 1)
}

/// Build an anisotropic cell list.
///
/// Cell count per dimension is capped so the list stays within ~50 MB regardless
/// of how small r_p_max or pi_max is (important when sweeping from kpc/h scales).
///
/// Returns (cells, n_cells_xy, n_cells_z, cell_size_xy, cell_size_z)
/// where cells[ix * n_cells_xy * n_cells_z + iy * n_cells_z + iz] holds
/// the indices of particles in that cell.
fn build_cell_list(
    coords: &[[f64; 3]],
    box_size: f64,
    r_p_max: f64,
    pi_max: f64,
) -> (Vec<Vec<usize>>, usize, usize, f64, f64) {
    // Minimum 3 cells: with <3 cells in a periodic dimension, rem_euclid maps
    // both dix=-1 and dix=+1 to the same neighbour — pairs would be double-counted.
    // Maximum 128: bounds memory to ≤128³ × 24 bytes ≈ 50 MB for the cell headers,
    // independent of r_max. Cells larger than r_max add cheap distance comparisons
    // but never miss pairs.
    const N_CELLS_MAX: usize = 128;
    let n_cells_xy = ((box_size / r_p_max) as usize).clamp(3, N_CELLS_MAX);
    let n_cells_z  = ((box_size / pi_max)  as usize).clamp(3, N_CELLS_MAX);
    let cell_size_xy = box_size / n_cells_xy as f64;
    let cell_size_z  = box_size / n_cells_z  as f64;

    let mut cells: Vec<Vec<usize>> = vec![Vec::new(); n_cells_xy * n_cells_xy * n_cells_z];
    for (i, &[x, y, z]) in coords.iter().enumerate() {
        let ix = ((x / cell_size_xy) as usize).min(n_cells_xy - 1);
        let iy = ((y / cell_size_xy) as usize).min(n_cells_xy - 1);
        let iz = ((z / cell_size_z)  as usize).min(n_cells_z  - 1);
        cells[ix * n_cells_xy * n_cells_z + iy * n_cells_z + iz].push(i);
    }

    (cells, n_cells_xy, n_cells_z, cell_size_xy, cell_size_z)
}

/// Count galaxy pairs in 3D real-space radial bins, split by sub-volume membership.
///
/// Uses an isotropic cell-list for O(N) scaling. Periodic boundary conditions applied
/// via minimum-image convention. Each pair counted once (i < j).
///
/// Parameters
/// ----------
/// coords : (N, 3) float64 C-contiguous array
///     Galaxy positions [x, y, z] in Mpc/h.
/// subvol_ids : (N,) int32 array
///     Sub-volume index for each galaxy.
/// r_bins : (n_r+1,) float64 array
///     Radial separation bin edges in Mpc/h. May span from sub-Mpc to box/2.
/// box_size : float
///     Side length of the periodic simulation box.
///
/// Returns
/// -------
/// (dd_auto, dd_cross) : two (n_r,) float64 arrays
///     Pair counts from same-subvol and cross-subvol pairs respectively.
#[pyfunction]
#[pyo3(signature = (coords, subvol_ids, r_bins, box_size))]
fn count_pairs_1d<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    subvol_ids: PyReadonlyArray1<'py, i32>,
    r_bins: PyReadonlyArray1<'py, f64>,
    box_size: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let coords_arr = coords.as_array();
    let sv_arr = subvol_ids.as_array();
    let r_arr = r_bins.as_array();

    let n = coords_arr.shape()[0];
    let n_r = r_arr.len() - 1;
    let r_max = r_arr[n_r];
    let r_sq_max = r_max * r_max;

    // Pre-compute squared bin edges to avoid sqrt() in hot loop
    let r_bins_sq: Vec<f64> = r_arr.iter().map(|&x| x * x).collect();

    let coords_flat: Vec<[f64; 3]> = (0..n)
        .map(|i| [coords_arr[[i, 0]], coords_arr[[i, 1]], coords_arr[[i, 2]]])
        .collect();
    let sv_flat: Vec<i32> = sv_arr.to_vec();

    // Isotropic cell list: pass r_max for both transverse and LOS arguments.
    let (cells, n_cells, _, cell_size, _) =
        build_cell_list(&coords_flat, box_size, r_max, r_max);
    let half_box = box_size * 0.5;

    let (auto_flat, cross_flat) = (0..n)
        .into_par_iter()
        .fold(
            || (vec![0.0f64; n_r], vec![0.0f64; n_r]),
            |mut acc, i| {
                let [xi, yi, zi] = coords_flat[i];
                let svi = sv_flat[i];

                let ix = ((xi / cell_size) as usize).min(n_cells - 1);
                let iy = ((yi / cell_size) as usize).min(n_cells - 1);
                let iz = ((zi / cell_size) as usize).min(n_cells - 1);

                for dix in -1i64..=1 {
                    for diy in -1i64..=1 {
                        for diz in -1i64..=1 {
                            let nx = (ix as i64 + dix).rem_euclid(n_cells as i64) as usize;
                            let ny = (iy as i64 + diy).rem_euclid(n_cells as i64) as usize;
                            let nz = (iz as i64 + diz).rem_euclid(n_cells as i64) as usize;
                            let ncell = nx * n_cells * n_cells + ny * n_cells + nz;

                            for &j in &cells[ncell] {
                                if j <= i {
                                    continue;
                                }

                                let [xj, yj, zj] = coords_flat[j];

                                let mut dx = xj - xi;
                                let mut dy = yj - yi;
                                let mut dz = zj - zi;
                                if dx >  half_box { dx -= box_size; } else if dx < -half_box { dx += box_size; }
                                if dy >  half_box { dy -= box_size; } else if dy < -half_box { dy += box_size; }
                                if dz >  half_box { dz -= box_size; } else if dz < -half_box { dz += box_size; }

                                let r_sq = dx * dx + dy * dy + dz * dz;
                                if r_sq >= r_sq_max {
                                    continue;
                                }

                                // Use squared bin edges to avoid sqrt
                                if let Some(ir) = find_bin_squared(r_sq, &r_bins_sq) {
                                    if sv_flat[j] == svi {
                                        acc.0[ir] += 1.0;
                                    } else {
                                        acc.1[ir] += 1.0;
                                    }
                                }
                            }
                        }
                    }
                }
                acc
            },
        )
        .reduce(
            || (vec![0.0f64; n_r], vec![0.0f64; n_r]),
            |mut a, b| {
                a.0.iter_mut().zip(&b.0).for_each(|(x, y)| *x += y);
                a.1.iter_mut().zip(&b.1).for_each(|(x, y)| *x += y);
                a
            },
        );

    Ok((
        Array1::from_vec(auto_flat).into_pyarray(py),
        Array1::from_vec(cross_flat).into_pyarray(py),
    ))
}

/// Count galaxy pairs in 2D (r_p, π) bins, split by sub-volume membership.
///
/// Uses an anisotropic cell-list for O(N) scaling. The line-of-sight axis is z.
/// Periodic boundary conditions are applied.
///
/// Parameters
/// ----------
/// coords : (N, 3) float64 C-contiguous array
///     Galaxy positions [x, y, z]. z is the line-of-sight axis.
/// subvol_ids : (N,) int32 array
///     Sub-volume index for each galaxy.
/// r_p_bins : (n_rp+1,) float64 array
///     Transverse separation bin edges in Mpc/h.
/// pi_bins : (n_pi+1,) float64 array
///     Line-of-sight separation bin edges in the same unit.
/// box_size : float
///     Side length of the periodic simulation box.
///
/// Returns
/// -------
/// (dd_auto, dd_cross) : two (n_rp, n_pi) float64 arrays
///     Pair counts from same-subvol and cross-subvol pairs respectively.
///     Each pair is counted exactly once (i < j convention).
#[pyfunction]
#[pyo3(signature = (coords, subvol_ids, r_p_bins, pi_bins, box_size))]
fn count_pairs_2d<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    subvol_ids: PyReadonlyArray1<'py, i32>,
    r_p_bins: PyReadonlyArray1<'py, f64>,
    pi_bins: PyReadonlyArray1<'py, f64>,
    box_size: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let coords_arr = coords.as_array();
    let sv_arr = subvol_ids.as_array();
    let rp_arr = r_p_bins.as_array();
    let pi_arr = pi_bins.as_array();

    let n = coords_arr.shape()[0];
    let n_rp = rp_arr.len() - 1;
    let n_pi = pi_arr.len() - 1;

    let r_p_max = rp_arr[n_rp];
    let pi_max = pi_arr[n_pi];
    let r_p_sq_max = r_p_max * r_p_max;

    // Pre-compute squared r_p bin edges to avoid sqrt() in hot loop
    let rp_bins_sq: Vec<f64> = rp_arr.iter().map(|&x| x * x).collect();

    let coords_flat: Vec<[f64; 3]> = (0..n)
        .map(|i| [coords_arr[[i, 0]], coords_arr[[i, 1]], coords_arr[[i, 2]]])
        .collect();
    let sv_flat: Vec<i32> = sv_arr.to_vec();
    let pi_bins_vec: Vec<f64> = pi_arr.to_vec();

    let (cells, n_cells_xy, n_cells_z, cell_size_xy, cell_size_z) =
        build_cell_list(&coords_flat, box_size, r_p_max, pi_max);

    let half_box = box_size * 0.5;

    let (auto_flat, cross_flat) = (0..n)
        .into_par_iter()
        .fold(
            || (vec![0.0f64; n_rp * n_pi], vec![0.0f64; n_rp * n_pi]),
            |mut acc, i| {
                let [xi, yi, zi] = coords_flat[i];
                let svi = sv_flat[i];

                let ix = ((xi / cell_size_xy) as usize).min(n_cells_xy - 1);
                let iy = ((yi / cell_size_xy) as usize).min(n_cells_xy - 1);
                let iz = ((zi / cell_size_z)  as usize).min(n_cells_z  - 1);

                for dix in -1i64..=1 {
                    for diy in -1i64..=1 {
                        for diz in -1i64..=1 {
                            let nx = (ix as i64 + dix).rem_euclid(n_cells_xy as i64) as usize;
                            let ny = (iy as i64 + diy).rem_euclid(n_cells_xy as i64) as usize;
                            let nz = (iz as i64 + diz).rem_euclid(n_cells_z  as i64) as usize;
                            let ncell = nx * n_cells_xy * n_cells_z + ny * n_cells_z + nz;

                            for &j in &cells[ncell] {
                                if j <= i {
                                    continue;
                                }

                                let [xj, yj, zj] = coords_flat[j];

                                let mut dx = xj - xi;
                                let mut dy = yj - yi;
                                let mut dz = zj - zi;
                                if dx >  half_box { dx -= box_size; } else if dx < -half_box { dx += box_size; }
                                if dy >  half_box { dy -= box_size; } else if dy < -half_box { dy += box_size; }
                                if dz >  half_box { dz -= box_size; } else if dz < -half_box { dz += box_size; }

                                let r_p_sq = dx * dx + dy * dy;
                                let pi = dz.abs();

                                if r_p_sq >= r_p_sq_max || pi >= pi_max {
                                    continue;
                                }

                                // Use squared bin edges for r_p to avoid sqrt
                                if let (Some(irp), Some(ipi)) = (
                                    find_bin_squared(r_p_sq, &rp_bins_sq),
                                    find_bin(pi, &pi_bins_vec),
                                ) {
                                    let flat = irp * n_pi + ipi;
                                    if sv_flat[j] == svi {
                                        acc.0[flat] += 1.0;
                                    } else {
                                        acc.1[flat] += 1.0;
                                    }
                                }
                            }
                        }
                    }
                }
                acc
            },
        )
        .reduce(
            || (vec![0.0f64; n_rp * n_pi], vec![0.0f64; n_rp * n_pi]),
            |mut a, b| {
                a.0.iter_mut().zip(&b.0).for_each(|(x, y)| *x += y);
                a.1.iter_mut().zip(&b.1).for_each(|(x, y)| *x += y);
                a
            },
        );

    let dd_auto =
        Array2::from_shape_vec((n_rp, n_pi), auto_flat).expect("shape mismatch in dd_auto");
    let dd_cross =
        Array2::from_shape_vec((n_rp, n_pi), cross_flat).expect("shape mismatch in dd_cross");

    Ok((dd_auto.into_pyarray(py), dd_cross.into_pyarray(py)))
}

#[pymodule]
fn _scope(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(count_pairs_1d, m)?)?;
    m.add_function(wrap_pyfunction!(count_pairs_2d, m)?)?;
    Ok(())
}
