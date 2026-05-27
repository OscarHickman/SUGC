use numpy::{ndarray::Array2, IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::cell_list::{CellList, HALF_SHELL, find_bin_squared};

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

    let mut cl = CellList::build(&coords_flat, box_size, s_max, s_max);

    // Z-sorting inside cells
    for c in 0..(cl.offsets.len() - 1) {
        let start = cl.offsets[c];
        let end = cl.offsets[c+1];
        cl.indices[start..end].sort_unstable_by(|&a, &b| {
            coords_flat[a][2].partial_cmp(&coords_flat[b][2]).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    let n_xy = cl.n_xy;
    let n_z  = cl.n_z;
    let n_total_cells = n_xy * n_xy * n_z;
    let n_mu = n_mu_bins;
    let half_box = box_size * 0.5;

    // Contiguous Structure-of-Arrays reordering
    let mut xs = vec![0.0; n];
    let mut ys = vec![0.0; n];
    let mut zs = vec![0.0; n];
    let mut svs = vec![0i32; n];
    let mut oidx = vec![0usize; n];
    for (k, &i) in cl.indices.iter().enumerate() {
        xs[k] = coords_flat[i][0];
        ys[k] = coords_flat[i][1];
        zs[k] = coords_flat[i][2];
        svs[k] = sv_flat[i];
        oidx[k] = i;
    }

    let needs_pbc = 2.0 * s_max > half_box;

    let (auto_flat, cross_flat) = (0..n_total_cells)
        .into_par_iter()
        .fold(
            || (vec![0.0f64; n_s * n_mu], vec![0.0f64; n_s * n_mu]),
            |mut acc, c1| {
                let start1 = cl.offsets[c1];
                let end1 = cl.offsets[c1+1];
                if start1 == end1 { return acc; }
                let iz1 = c1 % n_z;
                let iy1 = (c1 / n_z) % n_xy;
                let ix1 = c1 / (n_z * n_xy);

                let (t_auto, t_cross) = &mut acc;

                for i in start1..end1 {
                    let xi = xs[i]; let yi = ys[i]; let zi = zs[i];
                    let svi = svs[i]; let i_orig = oidx[i];

                    // Self cell: j > i only
                    for j in start1..end1 {
                        if oidx[j] > i_orig {
                            let mut dx = xs[j] - xi;
                            let mut dy = ys[j] - yi;
                            let mut dz = zs[j] - zi;
                            if needs_pbc {
                                if dx > half_box { dx -= box_size; } else if dx < -half_box { dx += box_size; }
                                if dy > half_box { dy -= box_size; } else if dy < -half_box { dy += box_size; }
                                if dz > half_box { dz -= box_size; } else if dz < -half_box { dz += box_size; }
                            }
                            let s_sq = dx*dx + dy*dy + dz*dz;
                            if let Some(is) = find_bin_squared(s_sq, &s_sq_bins) {
                                let s = s_sq.sqrt();
                                let mu = dz.abs() / s;
                                if mu < mu_max {
                                    let imu = (mu / mu_max * n_mu as f64) as usize;
                                    let imu = imu.min(n_mu - 1);
                                    let flat = is * n_mu + imu;
                                    if svs[j] == svi { t_auto[flat] += 1.0; } else { t_cross[flat] += 1.0; }
                                }
                            }
                        }
                    }

                    // 13 neighboring cells
                    for &(dix, diy, diz) in &HALF_SHELL {
                        let mut nx = ix1 as i32 + dix; let mut ox = 0.0;
                        if nx < 0 { nx += n_xy as i32; ox = -box_size; }
                        else if nx >= n_xy as i32 { nx -= n_xy as i32; ox = box_size; }

                        let mut ny = iy1 as i32 + diy; let mut oy = 0.0;
                        if ny < 0 { ny += n_xy as i32; oy = -box_size; }
                        else if ny >= n_xy as i32 { ny -= n_xy as i32; oy = box_size; }

                        let mut nz = iz1 as i32 + diz; let mut oz = 0.0;
                        if nz < 0 { nz += n_z as i32; oz = -box_size; }
                        else if nz >= n_z as i32 { nz -= n_z as i32; oz = box_size; }

                        let c2 = (nx as usize) * n_xy * n_z + (ny as usize) * n_z + (nz as usize);
                        let start2 = cl.offsets[c2];
                        let end2 = cl.offsets[c2+1];
                        if start2 == end2 { continue; }

                        if needs_pbc {
                            for j_loc in 0..(end2 - start2) {
                                let kj = start2 + j_loc;
                                let mut dx = xs[kj] - xi + ox;
                                if dx > half_box { dx -= box_size; } else if dx < -half_box { dx += box_size; }
                                let mut dy = ys[kj] - yi + oy;
                                if dy > half_box { dy -= box_size; } else if dy < -half_box { dy += box_size; }
                                let mut dz = zs[kj] - zi + oz;
                                if dz > half_box { dz -= box_size; } else if dz < -half_box { dz += box_size; }

                                let s_sq = dx*dx + dy*dy + dz*dz;
                                if let Some(is) = find_bin_squared(s_sq, &s_sq_bins) {
                                    let s = s_sq.sqrt();
                                    let mu = dz.abs() / s;
                                    if mu < mu_max {
                                        let imu = (mu / mu_max * n_mu as f64) as usize;
                                        let imu = imu.min(n_mu - 1);
                                        let flat = is * n_mu + imu;
                                        if svs[kj] == svi { t_auto[flat] += 1.0; } else { t_cross[flat] += 1.0; }
                                    }
                                }
                            }
                        } else {
                            let cell2_zs = &zs[start2..end2];
                            let z_min = zi - s_max - oz;
                            let z_max = zi + s_max - oz;
                            let j_start_loc = cell2_zs.partition_point(|&z| z < z_min);
                            for j_loc in j_start_loc..cell2_zs.len() {
                                if unsafe { *cell2_zs.get_unchecked(j_loc) } > z_max { break; }
                                let kj = start2 + j_loc;

                                let dx = xs[kj] - xi + ox;
                                let dy = ys[kj] - yi + oy;
                                let dz = zs[kj] - zi + oz;
                                let s_sq = dx*dx + dy*dy + dz*dz;
                                if let Some(is) = find_bin_squared(s_sq, &s_sq_bins) {
                                    let s = s_sq.sqrt();
                                    let mu = dz.abs() / s;
                                    if mu < mu_max {
                                        let imu = (mu / mu_max * n_mu as f64) as usize;
                                        let imu = imu.min(n_mu - 1);
                                        let flat = is * n_mu + imu;
                                        if svs[kj] == svi { t_auto[flat] += 1.0; } else { t_cross[flat] += 1.0; }
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
