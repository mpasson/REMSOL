use pyo3::prelude::*;

#[pyclass]
#[derive(Copy, Clone)]
pub enum BackEnd {
    Scattering,
    Transfer,
}

#[pyclass]
#[derive(Copy, Clone)]
pub enum Polarization {
    TE,
    TM,
}
