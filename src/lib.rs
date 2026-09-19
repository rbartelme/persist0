pub mod batch;
pub mod h0;

use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3};
use pyo3::prelude::*;

/// h0_batched(D: f32[B, N, N]) -> u32[B, N-1, 2]
#[pyfunction]
fn h0_batched<'py>(
    py: Python<'py>,
    d: PyReadonlyArray3<'py, f32>,
) -> PyResult<Bound<'py, PyArray3<u32>>> {
    let arr = d.as_array();
    let (b, n, n2) = arr.dim();
    if n != n2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "D must be [B, N, N]",
        ));
    }
    let flat = arr
        .as_slice()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("D must be C-contiguous"))?;

    let out = py.detach(|| batch::h0_batched(flat, b, n));

    Ok(
        numpy::ndarray::Array3::from_shape_vec((b, n.saturating_sub(1), 2), out)
            .expect("shape")
            .into_pyarray(py),
    )
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(h0_batched, m)?)?;
    Ok(())
}
