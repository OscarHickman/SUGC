use numpy::{ndarray::{Array1, Array2}, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

#[inline]
fn find_bin(value: f64, edges: &[f64]) -> Option<usize> {
    if value < edges[0] || value >= *edges.last().unwrap() {
        return None;
    }
    Some(edges.partition_point(|&e| e <= value) - 1)
}

#[inline]
fn find_bin_squared(value_sq: f64, edges_sq: &[f64]) -> Option<usize> {
    if value_sq < edges_sq[0] || value_sq >= *edges_sq.last().unwrap() {
        return None;
    }
    Some(edges_sq.partition_point(|&e| e <= value_sq) - 1)
}

/// 13 forward half-shell offsets (dix, diy, diz), excluding self (0,0,0).
/// For particle i, we count all j in these 13 forward cells (no i<j check needed)
/// plus j > i in the self cell. Together these cover every unordered pair exactly once.
const HALF_SHELL: [(i32, i32, i32); 13] = [
    (-1,-1,1), (0,-1,1), (1,-1,1),
    (-1, 0,1), (0, 0,1), (1, 0,1),
    (-1, 1,1), (0, 1,1), (1, 1,1),
    (-1, 1,0), (0, 1,0), (1, 1,0),
    ( 1, 0,0),
];

/// Flat cell list: a single contiguous `indices` array (particles sorted by cell)
/// plus an `offsets` array for O(1) cell slicing. Avoids the Vec<Vec<>> pointer
/// indirection that breaks CPU prefetching in the inner pair loop.
struct CellList {
    indices: Vec<usize>,
    offsets: Vec<usize>,
    n_xy: usize,
    n_z: usize,
    size_xy: f64,
    size_z: f64,
}

impl CellList {
    fn build(coords: &[[f64; 3]], box_size: f64, r_xy: f64, r_z: f64) -> Self {
        const N_MAX: usize = 128;
        let n_xy = ((box_size / r_xy) as usize).clamp(3, N_MAX);
        let n_z  = ((box_size / r_z)  as usize).clamp(3, N_MAX);
        let size_xy = box_size / n_xy as f64;
        let size_z  = box_size / n_z  as f64;
        let n_total = n_xy * n_xy * n_z;

        let cell_of: Vec<usize> = coords.iter().map(|&[x, y, z]| {
            let ix = ((x / size_xy) as usize).min(n_xy - 1);
            let iy = ((y / size_xy) as usize).min(n_xy - 1);
            let iz = ((z / size_z)  as usize).min(n_z  - 1);
            ix * n_xy * n_z + iy * n_z + iz
        }).collect();

        let mut counts = vec![0usize; n_total];
        for &c in &cell_of { counts[c] += 1; }

        let mut offsets = vec![0usize; n_total + 1];
        for c in 0..n_total { offsets[c + 1] = offsets[c] + counts[c]; }

        let mut fill = offsets[..n_total].to_vec();
        let mut indices = vec![0usize; coords.len()];
        for (i, &c) in cell_of.iter().enumerate() {
            indices[fill[c]] = i;
            fill[c] += 1;
        }

        CellList { indices, offsets, n_xy, n_z, size_xy, size_z }
    }

    #[inline(always)]
    fn particles(&self, c: usize) -> &[usize] {
        &self.indices[self.offsets[c]..self.offsets[c + 1]]
    }

    #[inline(always)]
    fn idx(&self, ix: usize, iy: usize, iz: usize) -> usize {
        ix * self.n_xy * self.n_z + iy * self.n_z + iz
    }
}

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

#[pymodule]
fn _scope(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(count_pairs_1d, m)?)?;
    m.add_function(wrap_pyfunction!(count_pairs_2d, m)?)?;
    Ok(())
}
