use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Copy, Clone)]
pub struct Layer {
    pub n: f64,
    pub d: f64,
}

#[pymethods]
impl Layer {
    #[new]
    pub fn new(n: f64, d: f64) -> Layer {
        Layer { n, d }
    }

    fn __str__(&self) -> PyResult<String> {
        self.__repr__()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("Layer(n={}, d={})", self.n, self.d))
    }
}
