use numpy::{ndarray::{Array1, Array2}, IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyfunction]
pub fn has_cuda() -> bool {
    // This would ideally use a crate like 'cudarc' or 'nvml-wrapper'
    // For now, we provide a placeholder that returns false.
    false
}

#[pyfunction]
#[pyo3(signature = (coords, subvol_ids, r_bins, box_size, n_order))]
pub fn count_npoint_gpu<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'py, f64>,
    subvol_ids: PyReadonlyArray1<'py, i32>,
    r_bins: PyReadonlyArray1<'py, f64>,
    box_size: f64,
    n_order: usize,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
    crate::pairs_npoint::count_npoint(py, coords, subvol_ids, r_bins, box_size, n_order)
}
