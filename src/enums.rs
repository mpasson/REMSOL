//! This modules implements the enums used in the library.
use pyo3::prelude::*;

/// Enum for schosing the back end of the simulation.
#[pyclass]
#[derive(Copy, Clone)]
pub enum BackEnd {
    /// Implements the scattering matrix method.
    /// Only supports index finding and not the field plotting.
    Scattering,
    /// Implements the transfer matrix method.
    /// Supports index finding and field plotting.
    Transfer,
}

/// Enum for choosing the polarization of the light.
#[pyclass]
#[derive(Copy, Clone)]
pub enum Polarization {
    /// TE polarization.
    TE,
    /// TM polarization.
    TM,
}
