use numpy::{ndarray::{Array1, Array2}, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::cell_list::{CellList, HALF_SHELL, find_bin_squared};

#[pyfunction]
#[pyo3(signature = (coords, subvol_ids, r_bins, box_size, n_order))]
pub fn count_npoint<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    subvol_ids: PyReadonlyArray1<'py, i32>,
    r_bins: PyReadonlyArray1<'py, f64>,
    box_size: f64,
    n_order: usize,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
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

    let mut cl = CellList::build(&coords_flat, box_size, r_max, r_max);
    
    for c in 0..(cl.offsets.len() - 1) {
        let start = cl.offsets[c];
        let end = cl.offsets[c+1];
        cl.indices[start..end].sort_unstable_by(|&a, &b| {
            coords_flat[a][2].partial_cmp(&coords_flat[b][2]).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    let n_xy = cl.n_xy;
    let n_z = cl.n_z;
    let n_total_cells = n_xy * n_xy * n_z;
    let half_box = box_size * 0.5;
    
    let mut xs = vec![0.0; n];
    let mut ys = vec![0.0; n];
    let mut zs = vec![0.0; n];
    let mut svs = vec![0i32; n];
    let mut oidx = vec![0usize; n];
    for (k, &i) in cl.indices.iter().enumerate() {
        xs[k] = coords_flat[i][0]; ys[k] = coords_flat[i][1]; zs[k] = coords_flat[i][2];
        svs[k] = sv_arr[i]; oidx[k] = i;
    }

    if n_order == 2 {
        let t_by_s_flat = (0..n_total_cells)
            .into_par_iter()
            .fold(
                || vec![0.0f64; 2 * n_r],
                |mut t_by_s, c1| {
                    let start1 = cl.offsets[c1]; let end1 = cl.offsets[c1+1];
                    if start1 == end1 { return t_by_s; }
                    let iz1 = c1 % n_z; let iy1 = (c1 / n_z) % n_xy; let ix1 = c1 / (n_z * n_xy);

                    for i in start1..end1 {
                        let xi = xs[i]; let yi = ys[i]; let zi = zs[i];
                        let svi = svs[i]; let i_orig = oidx[i];
                        let z_max = zi + r_max;
                        for j in start1..end1 {
                            if oidx[j] > i_orig {
                                if zs[j] > z_max { break; } // Pruning works if we start j from i+1 or check range
                                // Actually, if we want bit-perfect and fast:
                                let dx = xs[j] - xi; let dy = ys[j] - yi; let dz = zs[j] - zi;
                                let r_sq = dx*dx + dy*dy + dz*dz;
                                if let Some(bin) = find_bin_squared(r_sq, &r_bins_sq) {
                                    let s = (svi != svs[j]) as usize;
                                    unsafe { *t_by_s.get_unchecked_mut(s * n_r + bin) += 1.0; }
                                }
                            }
                        }
                    }

                    for &(dix, diy, diz) in &HALF_SHELL {
                        let mut nx = ix1 as i32 + dix; let mut ox = 0.0;
                        if nx < 0 { nx += n_xy as i32; ox = -box_size; } else if nx >= n_xy as i32 { nx -= n_xy as i32; ox = box_size; }
                        let mut ny = iy1 as i32 + diy; let mut oy = 0.0;
                        if ny < 0 { ny += n_xy as i32; oy = -box_size; } else if ny >= n_xy as i32 { ny -= n_xy as i32; oy = box_size; }
                        let mut nz = iz1 as i32 + diz; let mut oz = 0.0;
                        if nz < 0 { nz += n_z as i32; oz = -box_size; } else if nz >= n_z as i32 { nz -= n_z as i32; oz = box_size; }
                        let c2 = (nx as usize) * n_xy * n_z + (ny as usize) * n_z + (nz as usize);
                        let start2 = cl.offsets[c2]; let end2 = cl.offsets[c2+1];
                        let cell2_zs = &zs[start2..end2];
                        for i in start1..end1 {
                            let xi = xs[i]; let yi = ys[i]; let zi = zs[i];
                            let svi = svs[i];
                            let z_min = zi - r_max - oz; let z_max = zi + r_max - oz;
                            let j_start_loc = cell2_zs.partition_point(|&z| z < z_min);
                            for j_loc in j_start_loc..cell2_zs.len() {
                                if unsafe { *cell2_zs.get_unchecked(j_loc) } > z_max { break; }
                                let kj = start2 + j_loc;
                                let dx = xs[kj] - xi + ox; let dy = ys[kj] - yi + oy; let dz = zs[kj] - zi + oz;
                                let r_sq = dx*dx + dy*dy + dz*dz;
                                if let Some(bin) = find_bin_squared(r_sq, &r_bins_sq) {
                                    let s = (svi != svs[kj]) as usize;
                                    unsafe { *t_by_s.get_unchecked_mut(s * n_r + bin) += 1.0; }
                                }
                            }
                        }
                    }
                    t_by_s
                }
            )
            .reduce(
                || vec![0.0f64; 2 * n_r],
                |mut a, b| { a.iter_mut().zip(&b).for_each(|(x, y)| *x += y); a }
            );

        let t_by_s = Array2::from_shape_vec((2, n_r), t_by_s_flat).unwrap();
        let mut t_total_arr = vec![0.0f64; n_r];
        for ir in 0..n_r { t_total_arr[ir] = t_by_s[[0, ir]] + t_by_s[[1, ir]]; }
        return Ok((t_by_s.into_pyarray(py), Array1::from_vec(t_total_arr).into_pyarray(py)));
    }

    // N >= 3: Keep existing optimized paths
    let n_table = 4096;
    let table_scale = (n_table - 1) as f64 / r_sq_max;
    let bin_table: Vec<i32> = (0..n_table).map(|i| {
        let r_sq = (i as f64 / (n_table - 1) as f64) * r_sq_max;
        match r_bins_sq.partition_point(|&e| e <= r_sq) {
            0 => -1,
            idx => (idx - 1) as i32,
        }
    }).collect();

    let needs_pbc_for_neighbors = 2.0 * r_max > half_box;
    let t_by_s_flat = (0..n_total_cells)
        .into_par_iter()
        .fold(
            || (vec![0.0f64; n_order * n_r], Vec::with_capacity(512), vec![0usize; n_order], vec![[0.0; 3]; n_order]),
            |mut acc, c1| {
                let start1 = cl.offsets[c1]; let end1 = cl.offsets[c1+1];
                if start1 == end1 { return acc; }
                let iz1 = c1 % n_z; let iy1 = (c1 / n_z) % n_xy; let ix1 = c1 / (n_z * n_xy);
                for i in start1..end1 {
                    let xi = xs[i]; let yi = ys[i]; let zi = zs[i];
                    let svi = svs[i]; let i_orig = oidx[i];
                    let (t_by_s, neighbors, selected_k, selected_coords) = &mut acc;
                    neighbors.clear();
                    for dix in -1..=1 {
                        let mut nx = ix1 as i32 + dix; let mut ox = 0.0;
                        if nx < 0 { nx += n_xy as i32; ox = -box_size; } else if nx >= n_xy as i32 { nx -= n_xy as i32; ox = box_size; }
                        for diy in -1..=1 {
                            let mut ny = iy1 as i32 + diy; let mut oy = 0.0;
                            if ny < 0 { ny += n_xy as i32; oy = -box_size; } else if ny >= n_xy as i32 { ny -= n_xy as i32; oy = box_size; }
                            for diz in -1..=1 {
                                let mut nz = iz1 as i32 + diz; let mut oz = 0.0;
                                if nz < 0 { nz += n_z as i32; oz = -box_size; } else if nz >= n_z as i32 { nz -= n_z as i32; oz = box_size; }
                                let nc = (nx as usize) * n_xy * n_z + (ny as usize) * n_z + (nz as usize);
                                let start2 = cl.offsets[nc]; let end2 = cl.offsets[nc+1];
                                let z_min = zi - r_max - oz; let z_max = zi + r_max - oz;
                                let zs_sub = &zs[start2..end2];
                                let j_start_loc = zs_sub.partition_point(|&z| z < z_min);
                                for j_loc in j_start_loc..zs_sub.len() {
                                    if unsafe { *zs_sub.get_unchecked(j_loc) } > z_max { break; }
                                    let kj = start2 + j_loc;
                                    if oidx[kj] > i_orig {
                                        let dx = xs[kj] - xi + ox; let dy = ys[kj] - yi + oy; let dz = zs[kj] - zi + oz;
                                        let r_sq = dx*dx + dy*dy + dz*dz;
                                        if r_sq < r_sq_max { neighbors.push((kj, dx, dy, dz, r_sq)); }
                                    }
                                }
                            }
                        }
                    }
                    if n_order == 3 {
                        for idx_j in 0..neighbors.len() {
                            let (kj, dxj, dyj, dzj, r_ij_sq) = neighbors[idx_j];
                            let svj = svs[kj];
                            for idx_k in idx_j + 1..neighbors.len() {
                                let (kk, dxk, dyk, dzk, r_ik_sq) = neighbors[idx_k];
                                let mut dx = dxk - dxj; let mut dy = dyk - dyj; let mut dz = dzk - dzj;
                                if needs_pbc_for_neighbors {
                                    if dx > half_box { dx -= box_size; } else if dx < -half_box { dx += box_size; }
                                    if dy > half_box { dy -= box_size; } else if dy < -half_box { dy += box_size; }
                                    if dz > half_box { dz -= box_size; } else if dz < -half_box { dz += box_size; }
                                }
                                let r_sq = dx*dx + dy*dy + dz*dz;
                                if r_sq >= r_sq_max { continue; }
                                let r_tuple_max = r_ij_sq.max(r_ik_sq).max(r_sq);
                                let b_idx = (r_tuple_max * table_scale) as usize;
                                let bin = unsafe { *bin_table.get_unchecked(b_idx.min(n_table - 1)) };
                                if bin >= 0 {
                                    let mut s_vals = [svi, svj, svs[kk]]; s_vals.sort_unstable();
                                    let mut s = 1; if s_vals[1] != s_vals[0] { s += 1; } if s_vals[2] != s_vals[1] { s += 1; }
                                    unsafe { *t_by_s.get_unchecked_mut((s - 1) * n_r + bin as usize) += 1.0; }
                                }
                            }
                        }
                    } else {
                        selected_k[0] = i; selected_coords[0] = [0.0; 3];
                        #[allow(clippy::too_many_arguments)]
                        fn recurse(depth: usize, start: usize, cur_max: f64, sk: &mut [usize], sc: &mut [[f64; 3]], ns: &Vec<(usize, f64, f64, f64, f64)>, svs: &[i32], nr: usize, box_size: f64, half_box: f64, r_sq_max: f64, needs_pbc: bool, tbs: &mut Vec<f64>, bt: &[i32], ts: f64, nt: usize) {
                            if depth == sk.len() {
                                let b_idx = (cur_max * ts) as usize;
                                let bin = unsafe { *bt.get_unchecked(b_idx.min(nt - 1)) };
                                if bin >= 0 {
                                    let mut set = [0i32; 8]; let mut s = 0;
                                    for d in 0..depth {
                                        let sv = svs[sk[d]]; let mut found = false;
                                        for idx_s in 0..s { if set[idx_s] == sv { found = true; break; } }
                                        if !found { if s < 8 { set[s] = sv; s += 1; } }
                                    }
                                    unsafe { *tbs.get_unchecked_mut((s - 1) * nr + bin as usize) += 1.0; }
                                }
                                return;
                            }
                            for idx in start..ns.len() {
                                let (kn, dx_n, dy_n, dz_n, r_in_sq) = ns[idx];
                                let mut nmx = cur_max.max(r_in_sq);
                                if nmx >= r_sq_max { continue; }
                                let mut ok = true;
                                for d in 1..depth {
                                    let [dxp, dyp, dzp] = sc[d];
                                    let mut dx = dx_n - dxp; let mut dy = dy_n - dyp; let mut dz = dz_n - dzp;
                                    if needs_pbc {
                                        if dx > half_box { dx -= box_size; } else if dx < -half_box { dx += box_size; }
                                        if dy > half_box { dy -= box_size; } else if dy < -half_box { dy += box_size; }
                                        if dz > half_box { dz -= box_size; } else if dz < -half_box { dz += box_size; }
                                    }
                                    let rs = dx*dx + dy*dy + dz*dz;
                                    if rs >= r_sq_max { ok = false; break; }
                                    if rs > nmx { nmx = rs; }
                                }
                                if ok {
                                    sk[depth] = kn; sc[depth] = [dx_n, dy_n, dz_n];
                                    recurse(depth + 1, idx + 1, nmx, sk, sc, ns, svs, nr, box_size, half_box, r_sq_max, needs_pbc, tbs, bt, ts, nt);
                                }
                            }
                        }
                        recurse(1, 0, 0.0, selected_k, selected_coords, neighbors, &svs, n_r, box_size, half_box, r_sq_max, 2.0 * r_max > half_box, t_by_s, &bin_table, table_scale, n_table);
                    }
                }
                acc
            }
        )
        .reduce(
            || (vec![0.0f64; n_order * n_r], Vec::new(), Vec::new(), Vec::new()),
            |mut a, b| { a.0.iter_mut().zip(&b.0).for_each(|(x, y)| *x += y); a }
        );

    let t_by_s = Array2::from_shape_vec((n_order, n_r), t_by_s_flat.0).unwrap();
    let mut t_total_arr = vec![0.0f64; n_r];
    for s in 0..n_order { for ir in 0..n_r { t_total_arr[ir] += t_by_s[[s, ir]]; } }
    Ok((t_by_s.into_pyarray(py), Array1::from_vec(t_total_arr).into_pyarray(py)))
}
