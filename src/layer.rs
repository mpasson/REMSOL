extern crate num_complex;

use num_complex::Complex;
use pyo3::prelude::*;

#[derive(Debug, Copy, Clone)]
pub struct LayerCoefficientVector {
    pub a: Complex<f64>,
    pub b: Complex<f64>,
}

impl LayerCoefficientVector {
    pub fn new(a: Complex<f64>, b: Complex<f64>) -> LayerCoefficientVector {
        LayerCoefficientVector { a, b }
    }
}

#[pyclass]
#[derive(Debug, Copy, Clone)]
pub struct Layer {
    pub n: Complex<f64>,
    pub d: f64,
}

#[pymethods]
impl Layer {
    #[new]
    pub fn new(n: Complex<f64>, d: f64) -> Layer {
        Layer { n, d }
    }

    fn __str__(&self) -> PyResult<String> {
        self.__repr__()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Layer(n={}, d={})", self.n, self.d))
    }
}
