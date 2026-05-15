use numpy::{ndarray::Array2, IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::cell_list::{CellList, HALF_SHELL, find_bin_squared};

/// Count galaxy pairs in 2D (s, μ) redshift-space bins, split by sub-volume membership.
///
/// Uses an isotropic flat cell-list and half-shell traversal. LOS axis is z.
/// Caller must pre-apply the RSD shift and wrap z into [0, box_size):
///     z_rsd = (z + v_pec_z / H(z)) % box_size   (all in Mpc/h)
///
/// Parameters
/// ----------
/// coords : (N, 3) float64 array — redshift-space positions [x, y, z_rsd].
/// subvol_ids : (N,) int32 array — sub-volume index.
/// s_bins : (n_s+1,) float64 array — separation bin edges in Mpc/h.
/// n_mu_bins : int — number of uniform μ = |Δz|/s bins in [0, mu_max].
/// mu_max : float — maximum μ (typically 1.0).
/// box_size : float — periodic box side length.
///
/// Returns
/// -------
/// (dd_auto, dd_cross) : two (n_s, n_mu_bins) float64 arrays
#[pyfunction]
#[pyo3(signature = (coords, subvol_ids, s_bins, n_mu_bins, mu_max, box_size))]
pub fn count_pairs_smu<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    subvol_ids: PyReadonlyArray1<'py, i32>,
    s_bins: PyReadonlyArray1<'py, f64>,
    n_mu_bins: usize,
    mu_max: f64,
    box_size: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let coords_arr = coords.as_array();
    let sv_arr     = subvol_ids.as_array();
    let s_arr      = s_bins.as_array();

    let n   = coords_arr.shape()[0];
    let n_s = s_arr.len() - 1;
    let s_max = s_arr[n_s];
    let s_sq_bins: Vec<f64> = s_arr.iter().map(|&x| x * x).collect();

    let coords_flat: Vec<[f64; 3]> = (0..n)
        .map(|i| [coords_arr[[i, 0]], coords_arr[[i, 1]], coords_arr[[i, 2]]])
        .collect();
    let sv_flat: Vec<i32> = sv_arr.to_vec();

    // Isotropic cell list: search radius s_max in all three dimensions.
    let cl = CellList::build(&coords_flat, box_size, s_max, s_max);
    let half_box = box_size * 0.5;
    let n_xy = cl.n_xy;
    let n_z  = cl.n_z;
    let n_mu = n_mu_bins;

    let (auto_flat, cross_flat) = (0..n)
        .into_par_iter()
        .fold(
            || (vec![0.0f64; n_s * n_mu], vec![0.0f64; n_s * n_mu]),
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

                        let s_sq = dx*dx + dy*dy + dz*dz;
                        if let Some(is) = find_bin_squared(s_sq, &s_sq_bins) {
                            let s = s_sq.sqrt();
                            let mu = dz.abs() / s;
                            if mu >= mu_max { return; }
                            let imu = ((mu / mu_max) * n_mu as f64) as usize;
                            let imu = imu.min(n_mu - 1);
                            let flat = is * n_mu + imu;
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
            || (vec![0.0f64; n_s * n_mu], vec![0.0f64; n_s * n_mu]),
            |mut a, b| {
                a.0.iter_mut().zip(&b.0).for_each(|(x, y)| *x += y);
                a.1.iter_mut().zip(&b.1).for_each(|(x, y)| *x += y);
                a
            },
        );

    let dd_auto  = Array2::from_shape_vec((n_s, n_mu), auto_flat).expect("shape mismatch");
    let dd_cross = Array2::from_shape_vec((n_s, n_mu), cross_flat).expect("shape mismatch");

    Ok((dd_auto.into_pyarray(py), dd_cross.into_pyarray(py)))
}
