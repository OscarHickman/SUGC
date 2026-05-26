use pyo3::prelude::*;

mod cell_list;
mod pairs_1d;
mod pairs_2d;
mod pairs_npoint;
mod pairs_npoint_gpu;
mod pairs_smu;

use pairs_1d::count_pairs_1d;
use pairs_2d::count_pairs_2d;
use pairs_npoint::count_npoint;
use pairs_npoint_gpu::{count_npoint_gpu, has_cuda};
use pairs_smu::count_pairs_smu;

#[pymodule]
fn _scope(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(count_pairs_1d, m)?)?;
    m.add_function(wrap_pyfunction!(count_pairs_2d, m)?)?;
    m.add_function(wrap_pyfunction!(count_npoint, m)?)?;
    m.add_function(wrap_pyfunction!(count_npoint_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(has_cuda, m)?)?;
    m.add_function(wrap_pyfunction!(count_pairs_smu, m)?)?;
    Ok(())
}
