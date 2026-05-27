use numpy::{ndarray::Array1, IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::cell_list::{CellList, HALF_SHELL, find_bin_squared};

/// Count galaxy pairs in 3D real-space radial bins, split by sub-volume membership.
///
/// Uses an isotropic flat cell-list and half-shell traversal for O(N) scaling.
/// Periodic boundary conditions via minimum-image. Each pair counted once (i < j).
///
/// Parameters
/// ----------
/// coords : (N, 3) float64 C-contiguous array — galaxy positions [x, y, z] in Mpc/h.
/// subvol_ids : (N,) int32 array — sub-volume index for each galaxy.
/// r_bins : (n_r+1,) float64 array — radial bin edges in Mpc/h.
/// box_size : float — periodic box side length.
///
/// Returns
/// -------
/// (dd_auto, dd_cross) : two (n_r,) float64 arrays
#[pyfunction]
#[pyo3(signature = (coords, subvol_ids, r_bins, box_size))]
pub fn count_pairs_1d<'py>(
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
    let r_bins_sq: Vec<f64> = r_arr.iter().map(|&x| x * x).collect();

    let coords_flat: Vec<[f64; 3]> = (0..n)
        .map(|i| [coords_arr[[i, 0]], coords_arr[[i, 1]], coords_arr[[i, 2]]])
        .collect();
    let sv_flat: Vec<i32> = sv_arr.to_vec();

    let cl = CellList::build(&coords_flat, box_size, r_max, r_max);
    let half_box = box_size * 0.5;
    let n_xy = cl.n_xy;
    let n_z = cl.n_z;

    let (auto_flat, cross_flat) = (0..n)
        .into_par_iter()
        .fold(
            || (vec![0.0f64; n_r], vec![0.0f64; n_r]),
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
                        let r_sq = dx*dx + dy*dy + dz*dz;
                        if r_sq >= r_sq_max { return; }
                        if let Some(ir) = find_bin_squared(r_sq, &r_bins_sq) {
                            if sv_flat[j] == svi { acc.0[ir] += 1.0; } else { acc.1[ir] += 1.0; }
                        }
                    };

                    // Self cell: j > i only (half of self-cell pairs)
                    let c0 = cl.idx(ix, iy, iz);
                    for &j in cl.particles(c0) {
                        if j > i { count(j); }
                    }

                    // 13 forward half-shell cells: all j, no i<j guard needed
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
