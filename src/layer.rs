//! Implementation of the Layer and related structs
extern crate num_complex;
extern crate serde;

use num_complex::Complex;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

/// Struct representing the modal coefficient inside a layer.
#[derive(Debug, Copy, Clone)]
pub struct LayerCoefficientVector {
    /// Forward propagating wave.
    pub a: Complex<f64>,
    /// Backward propagating wave.
    pub b: Complex<f64>,
}

impl LayerCoefficientVector {
    /// Create a new LayerCoefficientVector.
    pub fn new(a: Complex<f64>, b: Complex<f64>) -> LayerCoefficientVector {
        LayerCoefficientVector { a, b }
    }
}

/// Struct representing a layer in the stack.
/// This class is also available in the Python API.
#[pyclass]
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct Layer {
    /// Refractive index of the layer.
    pub n: f64,
    /// Thickness of the layer.
    pub d: f64,
}

/// Implementation of the Python API for the Layer struct.
#[pymethods]
impl Layer {
    /// Create a new Layer.
    #[new]
    pub fn new(n: f64, d: f64) -> Layer {
        Layer { n, d }
    }

    /// Define how a layer is printed in Python.
    fn __str__(&self) -> PyResult<String> {
        self.__repr__()
    }

    /// Define how a layer is printed in Python.
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Layer(n={}, d={})", self.n, self.d))
    }
}
