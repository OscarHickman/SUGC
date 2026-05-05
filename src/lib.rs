use numpy::{ndarray::Array2, IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
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
    // Linear scan is fastest for the small number of bins typical in 2pcf.
    for k in 0..edges.len() - 1 {
        if value < edges[k + 1] {
            return Some(k);
        }
    }
    None
}

/// Build an anisotropic cell list.
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
    // Choose cell sizes so that one shell of neighbours is sufficient.
    let n_cells_xy = ((box_size / r_p_max) as usize).max(1);
    let n_cells_z = ((box_size / pi_max) as usize).max(1);
    let cell_size_xy = box_size / n_cells_xy as f64;
    let cell_size_z = box_size / n_cells_z as f64;

    let mut cells: Vec<Vec<usize>> = vec![Vec::new(); n_cells_xy * n_cells_xy * n_cells_z];
    for (i, &[x, y, z]) in coords.iter().enumerate() {
        let ix = ((x / cell_size_xy) as usize).min(n_cells_xy - 1);
        let iy = ((y / cell_size_xy) as usize).min(n_cells_xy - 1);
        let iz = ((z / cell_size_z) as usize).min(n_cells_z - 1);
        cells[ix * n_cells_xy * n_cells_z + iy * n_cells_z + iz].push(i);
    }

    (cells, n_cells_xy, n_cells_z, cell_size_xy, cell_size_z)
}

/// Count galaxy pairs split by sub-volume membership into (DD_auto, DD_cross).
///
/// Uses an anisotropic cell-list for O(N) scaling in the search phase.
/// The line-of-sight direction is z. Periodic boundary conditions are applied.
///
/// Parameters
/// ----------
/// coords : (N, 3) float64 C-contiguous array
///     Galaxy positions [x, y, z]. z is the line-of-sight axis.
/// subvol_ids : (N,) int32 array
///     Sub-volume index for each galaxy.
/// r_p_bins : (n_rp+1,) float64 array
///     Transverse separation bin edges in Mpc/h (or any consistent unit).
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

    // Flatten input into contiguous Vecs for cache-friendly access and thread safety.
    let coords_flat: Vec<[f64; 3]> = (0..n)
        .map(|i| [coords_arr[[i, 0]], coords_arr[[i, 1]], coords_arr[[i, 2]]])
        .collect();
    let sv_flat: Vec<i32> = sv_arr.iter().copied().collect();
    let rp_bins_vec: Vec<f64> = rp_arr.iter().copied().collect();
    let pi_bins_vec: Vec<f64> = pi_arr.iter().copied().collect();

    let (cells, n_cells_xy, n_cells_z, cell_size_xy, cell_size_z) =
        build_cell_list(&coords_flat, box_size, r_p_max, pi_max);

    // Parallel fold: each thread accumulates into a thread-local flat buffer,
    // then the buffers are summed in the reduce step.
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
                let iz = ((zi / cell_size_z) as usize).min(n_cells_z - 1);

                // Search the 3x3x3 neighbourhood of neighbouring cells.
                for dix in -1i64..=1 {
                    for diy in -1i64..=1 {
                        for diz in -1i64..=1 {
                            let nx = (ix as i64 + dix).rem_euclid(n_cells_xy as i64) as usize;
                            let ny = (iy as i64 + diy).rem_euclid(n_cells_xy as i64) as usize;
                            let nz = (iz as i64 + diz).rem_euclid(n_cells_z as i64) as usize;
                            let ncell = nx * n_cells_xy * n_cells_z + ny * n_cells_z + nz;

                            for &j in &cells[ncell] {
                                // Each unordered pair counted once (i < j).
                                if j <= i {
                                    continue;
                                }

                                let [xj, yj, zj] = coords_flat[j];

                                // Minimum-image periodic separations.
                                let mut dx = xj - xi;
                                let mut dy = yj - yi;
                                let mut dz = zj - zi;
                                if dx > half_box { dx -= box_size; } else if dx < -half_box { dx += box_size; }
                                if dy > half_box { dy -= box_size; } else if dy < -half_box { dy += box_size; }
                                if dz > half_box { dz -= box_size; } else if dz < -half_box { dz += box_size; }

                                let r_p_sq = dx * dx + dy * dy;
                                let pi = dz.abs();

                                if r_p_sq >= r_p_sq_max || pi >= pi_max {
                                    continue;
                                }

                                let r_p = r_p_sq.sqrt();

                                if let (Some(irp), Some(ipi)) = (
                                    find_bin(r_p, &rp_bins_vec),
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

    Ok((
        dd_auto.into_pyarray(py),
        dd_cross.into_pyarray(py),
    ))
}

#[pymodule]
fn _scope(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(count_pairs_2d, m)?)?;
    Ok(())
}
