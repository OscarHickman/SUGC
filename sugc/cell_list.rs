/// 13 forward half-shell offsets (dix, diy, diz), excluding self (0,0,0).
/// For particle i, we count all j in these 13 forward cells (no i<j check needed)
/// plus j > i in the self cell. Together these cover every unordered pair exactly once.
pub const HALF_SHELL: [(i32, i32, i32); 13] = [
    (-1,-1,1), (0,-1,1), (1,-1,1),
    (-1, 0,1), (0, 0,1), (1, 0,1),
    (-1, 1,1), (0, 1,1), (1, 1,1),
    (-1, 1,0), (0, 1,0), (1, 1,0),
    ( 1, 0,0),
];

/// Flat cell list: a single contiguous `indices` array (particles sorted by cell)
/// plus an `offsets` array for O(1) cell slicing. Avoids the Vec<Vec<>> pointer
/// indirection that breaks CPU prefetching in the inner pair loop.
pub struct CellList {
    pub indices: Vec<usize>,
    pub offsets: Vec<usize>,
    pub n_xy: usize,
    pub n_z: usize,
    pub size_xy: f64,
    pub size_z: f64,
}

impl CellList {
    pub fn build(coords: &[[f64; 3]], box_size: f64, r_xy: f64, r_z: f64) -> Self {
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
    pub fn particles(&self, c: usize) -> &[usize] {
        &self.indices[self.offsets[c]..self.offsets[c + 1]]
    }

    #[inline(always)]
    pub fn idx(&self, ix: usize, iy: usize, iz: usize) -> usize {
        ix * self.n_xy * self.n_z + iy * self.n_z + iz
    }
}

#[inline]
pub fn find_bin(value: f64, edges: &[f64]) -> Option<usize> {
    if value < edges[0] || value >= *edges.last().unwrap() {
        return None;
    }
    Some(edges.partition_point(|&e| e <= value) - 1)
}

#[inline]
pub fn find_bin_squared(value_sq: f64, edges_sq: &[f64]) -> Option<usize> {
    if value_sq < edges_sq[0] || value_sq >= *edges_sq.last().unwrap() {
        return None;
    }
    Some(edges_sq.partition_point(|&e| e <= value_sq) - 1)
}
