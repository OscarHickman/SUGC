use numpy::{ndarray::Array2, IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::cell_list::{CellList, HALF_SHELL, find_bin, find_bin_squared};

/// Count galaxy pairs in 2D (r_p, π) bins, split by sub-volume membership.
///
/// Uses an anisotropic flat cell-list and half-shell traversal. LOS axis is z.
/// Periodic boundary conditions applied via minimum-image.
///
/// Parameters
/// ----------
/// coords : (N, 3) float64 C-contiguous array — galaxy positions [x, y, z].
/// subvol_ids : (N,) int32 array — sub-volume index.
/// r_p_bins : (n_rp+1,) float64 array — transverse bin edges in Mpc/h.
/// pi_bins : (n_pi+1,) float64 array — LOS bin edges in the same unit.
/// box_size : float — periodic box side length.
///
/// Returns
/// -------
/// (dd_auto, dd_cross) : two (n_rp, n_pi) float64 arrays
#[pyfunction]
#[pyo3(signature = (coords, subvol_ids, r_p_bins, pi_bins, box_size))]
pub fn count_pairs_2d<'py>(
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

    let rp_bins_sq: Vec<f64> = rp_arr.iter().map(|&x| x * x).collect();
    let pi_bins_vec: Vec<f64> = pi_arr.to_vec();

    let coords_flat: Vec<[f64; 3]> = (0..n)
        .map(|i| [coords_arr[[i, 0]], coords_arr[[i, 1]], coords_arr[[i, 2]]])
        .collect();
    let sv_flat: Vec<i32> = sv_arr.to_vec();

    let cl = CellList::build(&coords_flat, box_size, r_p_max, pi_max);
    let half_box = box_size * 0.5;
    let n_xy = cl.n_xy;
    let n_z = cl.n_z;

    let (auto_flat, cross_flat) = (0..n)
        .into_par_iter()
        .fold(
            || (vec![0.0f64; n_rp * n_pi], vec![0.0f64; n_rp * n_pi]),
            |mut acc, i| {
                let [xi, yi, zi] = coords_flat[i];
                let svi = sv_flat[i];

                let ix = ((xi / cl.size_xy) as usize).min(n_xy - 1);
                let iy = ((yi / cl.size_xy) as usize).min(n_xy - 1);
                let iz = ((zi / cl.size_z)  as usize).min(n_z  - 1);

                {
                    let mut count = |j: usize| {
                        let [xj, yj, zj] = coords_flat[j];
                        let mut dx = xj - xi;
                        let mut dy = yj - yi;
                        let mut dz = zj - zi;
                        if dx >  half_box { dx -= box_size; } else if dx < -half_box { dx += box_size; }
                        if dy >  half_box { dy -= box_size; } else if dy < -half_box { dy += box_size; }
                        if dz >  half_box { dz -= box_size; } else if dz < -half_box { dz += box_size; }

                        let r_p_sq = dx*dx + dy*dy;
                        let pi = dz.abs();

                        if r_p_sq >= r_p_sq_max || pi >= pi_max { return; }

                        if let (Some(irp), Some(ipi)) = (
                            find_bin_squared(r_p_sq, &rp_bins_sq),
                            find_bin(pi, &pi_bins_vec),
                        ) {
                            let flat = irp * n_pi + ipi;
                            if sv_flat[j] == svi { acc.0[flat] += 1.0; } else { acc.1[flat] += 1.0; }
                        }
                    };

                    // Self cell: j > i only
                    let c0 = cl.idx(ix, iy, iz);
                    for &j in cl.particles(c0) {
                        if j > i { count(j); }
                    }

                    // 13 forward half-shell cells: all j
                    for &(dix, diy, diz) in &HALF_SHELL {
                        let nx = (ix as i32 + dix).rem_euclid(n_xy as i32) as usize;
                        let ny = (iy as i32 + diy).rem_euclid(n_xy as i32) as usize;
                        let nz = (iz as i32 + diz).rem_euclid(n_z  as i32) as usize;
                        let nc = cl.idx(nx, ny, nz);
                        for &j in cl.particles(nc) { count(j); }
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

    let dd_auto  = Array2::from_shape_vec((n_rp, n_pi), auto_flat).expect("shape mismatch");
    let dd_cross = Array2::from_shape_vec((n_rp, n_pi), cross_flat).expect("shape mismatch");

    Ok((dd_auto.into_pyarray(py), dd_cross.into_pyarray(py)))
}
