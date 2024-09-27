use pyo3::prelude::*;

pub mod enums;
pub mod layer;
pub mod multilayer;
pub mod scattering_matrix;
pub mod transfer_matrix;

use enums::*;
use layer::*;
use multilayer::*;

#[pymodule]
fn multilayer_solver(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BackEnd>()?;
    m.add_class::<Polarization>()?;
    m.add_class::<Layer>()?;
    m.add_class::<MultiLayer>()?;
    Ok(())
}
