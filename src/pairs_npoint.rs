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
    let sv_flat: Vec<i32> = sv_arr.to_vec();

    let cl = CellList::build(&coords_flat, box_size, r_max, r_max);
    let n_xy = cl.n_xy;
    let n_z = cl.n_z;
    let half_box = box_size * 0.5;
    
    // Reorder data for linear access
    let mut coords_reordered = vec![[0.0; 3]; n];
    let mut sv_reordered = vec![0i32; n];
    let mut original_indices = vec![0usize; n];
    for (k, &i) in cl.indices.iter().enumerate() {
        coords_reordered[k] = coords_flat[i];
        sv_reordered[k] = sv_flat[i];
        original_indices[k] = i;
    }

    let needs_pbc_for_neighbors = 2.0 * r_max > half_box;

    let (t_by_s_flat, t_total_flat, _, _, _) = (0..n)
        .into_par_iter()
        .fold(
            || (
                vec![0.0f64; n_order * n_r], 
                vec![0.0f64; n_r], 
                Vec::with_capacity(256), 
                vec![0usize; n_order],   
                vec![[0.0; 3]; n_order]  
            ),
            |mut acc, k| {
                let [xi, yi, zi] = coords_reordered[k];
                let svi = sv_reordered[k];
                let i_orig = original_indices[k];

                let ix = ((xi / cl.size_xy) as usize).min(n_xy - 1);
                let iy = ((yi / cl.size_xy) as usize).min(n_xy - 1);
                let iz = ((zi / cl.size_z)  as usize).min(n_z  - 1);

                if n_order == 2 {
                    let (t_by_s, t_total, ..) = (&mut acc.0, &mut acc.1);
                    let c0 = cl.idx(ix, iy, iz);
                    for &j_orig in cl.particles(c0) {
                        if j_orig > i_orig {
                            let [xj, yj, zj] = coords_flat[j_orig];
                            let dx = xj - xi;
                            let dy = yj - yi;
                            let dz = zj - zi;
                            let r_sq = dx*dx + dy*dy + dz*dz;
                            if r_sq < r_sq_max {
                                if let Some(ir) = find_bin_squared(r_sq, &r_bins_sq) {
                                    let s = if svi == sv_flat[j_orig] { 1 } else { 2 };
                                    t_by_s[(s - 1) * n_r + ir] += 1.0;
                                    t_total[ir] += 1.0;
                                }
                            }
                        }
                    }
                    for &(dix, diy, diz) in &HALF_SHELL {
                        let mut nx = ix as i32 + dix;
                        let mut ox = 0.0;
                        if nx < 0 { nx += n_xy as i32; ox = -box_size; }
                        else if nx >= n_xy as i32 { nx -= n_xy as i32; ox = box_size; }
                        let mut ny = iy as i32 + diy;
                        let mut oy = 0.0;
                        if ny < 0 { ny += n_xy as i32; oy = -box_size; }
                        else if ny >= n_xy as i32 { ny -= n_xy as i32; oy = box_size; }
                        let mut nz = iz as i32 + diz;
                        let mut oz = 0.0;
                        if nz < 0 { nz += n_z as i32; oz = -box_size; }
                        else if nz >= n_z as i32 { nz -= n_z as i32; oz = box_size; }
                        let nc = cl.idx(nx as usize, ny as usize, nz as usize);
                        for &j_orig in cl.particles(nc) {
                            let [xj, yj, zj] = coords_flat[j_orig];
                            let dx = xj - xi + ox;
                            let dy = yj - yi + oy;
                            let dz = zj - zi + oz;
                            let r_sq = dx*dx + dy*dy + dz*dz;
                            if r_sq < r_sq_max {
                                if let Some(ir) = find_bin_squared(r_sq, &r_bins_sq) {
                                    let s = if svi == sv_flat[j_orig] { 1 } else { 2 };
                                    t_by_s[(s - 1) * n_r + ir] += 1.0;
                                    t_total[ir] += 1.0;
                                }
                            }
                        }
                    }
                    return acc;
                }

                // General case (N >= 3)
                acc.2.clear();
                for dix in -1..=1 {
                    let mut nx = ix as i32 + dix;
                    let mut ox = 0.0;
                    if nx < 0 { nx += n_xy as i32; ox = -box_size; }
                    else if nx >= n_xy as i32 { nx -= n_xy as i32; ox = box_size; }
                    for diy in -1..=1 {
                        let mut ny = iy as i32 + diy;
                        let mut oy = 0.0;
                        if ny < 0 { ny += n_xy as i32; oy = -box_size; }
                        else if ny >= n_xy as i32 { ny -= n_xy as i32; oy = box_size; }
                        for diz in -1..=1 {
                            let mut nz = iz as i32 + diz;
                            let mut oz = 0.0;
                            if nz < 0 { nz += n_z as i32; oz = -box_size; }
                            else if nz >= n_z as i32 { nz -= n_z as i32; oz = box_size; }
                            let nc = cl.idx(nx as usize, ny as usize, nz as usize);
                            for &j_orig in cl.particles(nc) {
                                if j_orig > i_orig {
                                    let [xj, yj, zj] = coords_flat[j_orig];
                                    let dx = xj - xi + ox;
                                    let dy = yj - yi + oy;
                                    let dz = zj - zi + oz;
                                    let r_sq = dx*dx + dy*dy + dz*dz;
                                    if r_sq < r_sq_max {
                                        acc.2.push((j_orig, dx, dy, dz, r_sq));
                                    }
                                }
                            }
                        }
                    }
                }

                if n_order == 3 {
                    let (t_by_s, t_total, neighbors, _, _) = &mut acc;
                    for idx_j in 0..neighbors.len() {
                        let (j_orig, dxj, dyj, dzj, r_ij_sq) = neighbors[idx_j];
                        let svj = sv_flat[j_orig];
                        for idx_k in idx_j + 1..neighbors.len() {
                            let (k_orig, dxk, dyk, dzk, r_ik_sq) = neighbors[idx_k];
                            let svk = sv_flat[k_orig];
                            let mut dx = dxk - dxj;
                            let mut dy = dyk - dyj;
                            let mut dz = dzk - dzj;
                            if needs_pbc_for_neighbors {
                                if dx > half_box { dx -= box_size; } else if dx < -half_box { dx += box_size; }
                                if dy > half_box { dy -= box_size; } else if dy < -half_box { dy += box_size; }
                                if dz > half_box { dz -= box_size; } else if dz < -half_box { dz += box_size; }
                            }
                            let r_jk_sq = dx*dx + dy*dy + dz*dz;
                            if r_jk_sq >= r_sq_max { continue; }
                            let r_max_sq = r_ij_sq.max(r_ik_sq).max(r_jk_sq);
                            if let Some(ir) = find_bin_squared(r_max_sq, &r_bins_sq) {
                                let s = if svi == svj && svj == svk { 1 }
                                        else if svi != svj && svj != svk && svi != svk { 3 }
                                        else { 2 };
                                t_by_s[(s - 1) * n_r + ir] += 1.0;
                                t_total[ir] += 1.0;
                            }
                        }
                    }
                } else {
                    let (t_by_s, t_total, neighbors, selected_indices, selected_coords) = &mut acc;
                    selected_indices[0] = i_orig;
                    selected_coords[0] = [0.0; 3];

                    #[allow(clippy::too_many_arguments)]
                    fn fast_recurse(
                        depth: usize,
                        start_neighbor_idx: usize,
                        current_max_r_sq: f64,
                        selected_indices: &mut [usize],
                        selected_coords: &mut [[f64; 3]],
                        neighbors: &Vec<(usize, f64, f64, f64, f64)>,
                        sv_flat: &Vec<i32>,
                        r_bins_sq: &[f64],
                        n_r: usize,
                        n_order: usize,
                        box_size: f64,
                        half_box: f64,
                        r_sq_max: f64,
                        needs_pbc: bool,
                        t_by_s: &mut Vec<f64>,
                        t_total: &mut Vec<f64>,
                    ) {
                        if depth == n_order {
                            if let Some(ir) = find_bin_squared(current_max_r_sq, r_bins_sq) {
                                let mut distinct_svs = [0i32; 8];
                                let mut s = 0;
                                for d in 0..n_order {
                                    let sv = sv_flat[selected_indices[d]];
                                    let mut found = false;
                                    for idx_s in 0..s {
                                        if distinct_svs[idx_s] == sv { found = true; break; }
                                    }
                                    if !found {
                                        if s < 8 { distinct_svs[s] = sv; s += 1; }
                                    }
                                }
                                t_by_s[(s - 1) * n_r + ir] += 1.0;
                                t_total[ir] += 1.0;
                            }
                            return;
                        }

                        for idx in start_neighbor_idx..neighbors.len() {
                            let (next_p_orig, dx_n, dy_n, dz_n, r_in_sq) = neighbors[idx];
                            let mut new_max_r_sq = current_max_r_sq.max(r_in_sq);
                            if new_max_r_sq >= r_sq_max { continue; }

                            let mut possible = true;
                            for d in 1..depth {
                                let [dxp, dyp, dzp] = selected_coords[d];
                                let mut dx = dx_n - dxp;
                                let mut dy = dy_n - dyp;
                                let mut dz = dz_n - dzp;
                                if needs_pbc {
                                    if dx > half_box { dx -= box_size; } else if dx < -half_box { dx += box_size; }
                                    if dy > half_box { dy -= box_size; } else if dy < -half_box { dy += box_size; }
                                    if dz > half_box { dz -= box_size; } else if dz < -half_box { dz += box_size; }
                                }
                                let r_sq = dx*dx + dy*dy + dz*dz;
                                if r_sq >= r_sq_max {
                                    possible = false;
                                    break;
                                }
                                if r_sq > new_max_r_sq { new_max_r_sq = r_sq; }
                            }

                            if possible && new_max_r_sq < r_sq_max {
                                selected_indices[depth] = next_p_orig;
                                selected_coords[depth] = [dx_n, dy_n, dz_n];
                                fast_recurse(
                                    depth + 1,
                                    idx + 1,
                                    new_max_r_sq,
                                    selected_indices,
                                    selected_coords,
                                    neighbors,
                                    sv_flat,
                                    r_bins_sq,
                                    n_r,
                                    n_order,
                                    box_size,
                                    half_box,
                                    r_sq_max,
                                    needs_pbc,
                                    t_by_s,
                                    t_total,
                                );
                            }
                        }
                    }

                    fast_recurse(
                        1,
                        0,
                        0.0,
                        selected_indices,
                        selected_coords,
                        neighbors,
                        &sv_flat,
                        &r_bins_sq,
                        n_r,
                        n_order,
                        box_size,
                        half_box,
                        r_sq_max,
                        needs_pbc_for_neighbors,
                        t_by_s,
                        t_total,
                    );
                }

                acc
            },
        )
        .reduce(
            || (vec![0.0f64; n_order * n_r], vec![0.0f64; n_r], Vec::new(), Vec::new(), Vec::new()),
            |mut a, b| {
                a.0.iter_mut().zip(&b.0).for_each(|(x, y)| *x += y);
                a.1.iter_mut().zip(&b.1).for_each(|(x, y)| *x += y);
                a
            },
        );

    let t_by_s = Array2::from_shape_vec((n_order, n_r), t_by_s_flat).unwrap();
    let t_total = Array1::from_vec(t_total_flat);

    Ok((
        t_by_s.into_pyarray(py),
        t_total.into_pyarray(py),
    ))
}
