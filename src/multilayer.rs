extern crate cumsum;
extern crate find_peaks;
extern crate itertools;

use itertools::izip;
use num_complex::Complex64;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyComplex;
use std::cmp::Ordering;
use std::iter::zip;

use crate::enums::BackEnd;
use crate::enums::Polarization;
use crate::layer::{Layer, LayerCoefficientVector};
use crate::scattering_matrix::calculate_s_matrix;
use crate::transfer_matrix::{calculate_t_matrix, get_propagation_coefficients_transfer};
use cumsum::cumsum;
use find_peaks::PeakFinder;
use num_complex::Complex;

const Z0: Complex<f64> = Complex {
    re: 376.73031346177066,
    im: 0.0,
};

#[derive(Debug, Clone)]
struct ComplexWrapper {
    complex: Complex<f64>,
}

impl IntoPy<PyObject> for ComplexWrapper {
    fn into_py(self, py: Python<'_>) -> Py<PyAny> {
        PyComplex::from_doubles_bound(py, self.complex.re, self.complex.im).to_object(py)
    }
}

pub struct GridData {
    pub xplot: Vec<f64>,
    pub xstarts: Vec<f64>,
    pub ixstarts: Vec<usize>,
}

pub struct FieldData {
    pub x: Vec<f64>,
    pub Ex: Vec<Complex<f64>>,
    pub Ey: Vec<Complex<f64>>,
    pub Ez: Vec<Complex<f64>>,
    pub Hx: Vec<Complex<f64>>,
    pub Hy: Vec<Complex<f64>>,
    pub Hz: Vec<Complex<f64>>,
}

#[pyclass]
pub struct PythonFieldData {
    #[pyo3(get)]
    pub x: Vec<f64>,
    #[pyo3(get)]
    pub Ex: Vec<ComplexWrapper>,
    #[pyo3(get)]
    pub Ey: Vec<ComplexWrapper>,
    #[pyo3(get)]
    pub Ez: Vec<ComplexWrapper>,
    #[pyo3(get)]
    pub Hx: Vec<ComplexWrapper>,
    #[pyo3(get)]
    pub Hy: Vec<ComplexWrapper>,
    #[pyo3(get)]
    pub Hz: Vec<ComplexWrapper>,
}

#[pyclass]
pub struct IndexData {
    #[pyo3(get)]
    pub x: Vec<f64>,
    #[pyo3(get)]
    pub n: Vec<f64>,
}

fn get_field_slice(
    a: Complex<f64>,
    b: Complex<f64>,
    om: f64,
    k: f64,
    n: f64,
    x: Vec<f64>,
) -> Vec<Complex<f64>> {
    let om = num_complex::Complex::new(om, 0.0);
    let k = num_complex::Complex::new(k, 0.0);
    let n = num_complex::Complex::new(n, 0.0);
    let beta = ((om * n).powi(2) - k.powi(2)).sqrt();
    x.iter()
        .map(|x| {
            let z = num_complex::Complex::new(0.0, *x);
            let phase_p = z * beta;
            let phase_n = -z * beta;
            a * phase_p.exp() + b * phase_n.exp()
        })
        .collect()
}

#[pyclass]
pub struct MultiLayer {
    layers: Vec<Layer>,
    iteration: usize,
    backend: BackEnd,
    required_accuracy: i32,
    #[pyo3(get, set)]
    pub plot_step: f64,
}

#[pymethods]
impl MultiLayer {
    #[new]
    pub fn new(layers: Vec<Layer>) -> MultiLayer {
        let mut multilayer = MultiLayer {
            layers,
            iteration: 8,
            backend: BackEnd::Transfer,
            required_accuracy: 10,
            plot_step: 1e-3,
        };
        multilayer.set_backend(BackEnd::Transfer);
        multilayer
    }

    #[pyo3(name = "neff")]
    #[pyo3(signature = (omega, polarization=None, mode=None))]
    pub fn python_neff(
        &self,
        omega: f64,
        polarization: Option<Polarization>,
        mode: Option<usize>,
    ) -> PyResult<f64> {
        let polarization = polarization.unwrap_or(Polarization::TE);
        let mode = mode.unwrap_or(0);
        match self.neff(omega, polarization, mode) {
            Ok(neff) => Ok(neff),
            Err(err) => Err(PyException::new_err(err)),
        }
    }

    #[pyo3(name = "index")]
    pub fn python_index(&self) -> IndexData {
        let grid_data = self.get_grid_data();
        let index = self.get_index(&grid_data);
        IndexData {
            x: grid_data.xplot,
            n: index,
        }
    }

    #[pyo3(name = "field")]
    #[pyo3(signature = (omega, polarization=None, mode=None))]
    pub fn python_field(
        &self,
        omega: f64,
        polarization: Option<Polarization>,
        mode: Option<usize>,
    ) -> PyResult<PythonFieldData> {
        let polarization = polarization.unwrap_or(Polarization::TE);
        let mode = mode.unwrap_or(0);
        match self.field(omega, polarization, mode) {
            Ok(field_data) => Ok(PythonFieldData {
                x: field_data.x,
                Ex: field_data
                    .Ex
                    .iter()
                    .map(|c| ComplexWrapper { complex: *c })
                    .collect(),
                Ey: field_data
                    .Ey
                    .iter()
                    .map(|c| ComplexWrapper { complex: *c })
                    .collect(),
                Ez: field_data
                    .Ez
                    .iter()
                    .map(|c| ComplexWrapper { complex: *c })
                    .collect(),
                Hx: field_data
                    .Hx
                    .iter()
                    .map(|c| ComplexWrapper { complex: *c })
                    .collect(),
                Hy: field_data
                    .Hy
                    .iter()
                    .map(|c| ComplexWrapper { complex: *c })
                    .collect(),
                Hz: field_data
                    .Hz
                    .iter()
                    .map(|c| ComplexWrapper { complex: *c })
                    .collect(),
            }),
            Err(err) => Err(PyException::new_err(err)),
        }
    }
}

impl MultiLayer {
    pub fn set_backend(&mut self, backend: BackEnd) {
        self.backend = backend;
    }

    pub fn set_iteration(&mut self, iteration: usize) {
        self.iteration = iteration;
    }

    fn get_threshold(accuracy: i32) -> f64 {
        if accuracy < 3 {
            return -2.0;
        }
        if accuracy < 6 {
            return 0.0;
        }
        if accuracy < 9 {
            return 3.0;
        }
        if accuracy < 12 {
            return 6.0;
        }
        return 9.0;
    }

    fn characteristic_function(&self, om: f64, k: f64, polarization: Polarization) -> Complex<f64> {
        match self.backend {
            BackEnd::Scattering => {
                calculate_s_matrix(&self.layers, om, k, polarization).determinant()
            }
            BackEnd::Transfer => 1.0 / calculate_t_matrix(&self.layers, om, k, polarization).t22,
        }
    }

    fn find_minmax_n(&self) -> (f64, f64) {
        find_minmax_n(&self.layers)
    }

    fn solve_step(
        &self,
        om: f64,
        k_min: f64,
        k_max: f64,
        step: f64,
        treshold: f64,
        polarization: Polarization,
    ) -> Vec<f64> {
        let kv: Vec<f64> = iter_num_tools::arange(k_min..k_max, step).collect();
        let det: Vec<f64> = kv
            .clone()
            .into_iter()
            .map(|k| {
                self.characteristic_function(om, k, polarization)
                    .norm()
                    .log10()
            })
            .collect();

        let mut peak_finder = PeakFinder::new(&det);
        let peaks = peak_finder.with_min_height(treshold).find_peaks();

        let ksolutions = {
            peaks
                .into_iter()
                .map(|p| kv.get(p.middle_position()).unwrap_or_else(|| &0.0))
                // .flatten()
                .collect::<Vec<_>>()
        };

        ksolutions.into_iter().map(|k| *k).collect::<Vec<_>>()
    }

    pub fn solve(&self, om: f64, polarization: Polarization) -> Vec<f64> {
        let (min_n, max_n) = self.find_minmax_n();
        let k_min = om * min_n + 1e-9;
        let k_max = om * max_n - 1e-9;

        let mut solution_backets = vec![(k_min, k_max)];

        let mut ksolutions = Vec::new();

        for accuracy in 2..self.required_accuracy {
            let step = 10.0_f64.powi(-accuracy);
            let threshold = Self::get_threshold(accuracy);
            ksolutions.clear();
            for (_kmin, _kmax) in solution_backets {
                let _solutions = self.solve_step(om, _kmin, _kmax, step, threshold, polarization);
                ksolutions.extend(_solutions);
            }
            solution_backets = ksolutions
                .iter()
                .map(|k| (k - step, k + step))
                .collect::<Vec<_>>();
        }

        let mut n_solutions = ksolutions.into_iter().map(|k| k / om).collect::<Vec<_>>();

        n_solutions.sort_by(|a, b| b.partial_cmp(a).unwrap_or_else(|| Ordering::Equal));
        n_solutions
    }

    pub fn neff(&self, om: f64, polarization: Polarization, mode: usize) -> Result<f64, String> {
        let n_solutions = self.solve(om, polarization);
        match n_solutions.get(mode) {
            Some(n) => Ok(*n),
            None => Err(format!(
                "Mode {} not found. Only {} modes (0->{}) available.",
                mode,
                n_solutions.len(),
                n_solutions.len() - 1
            )),
        }
    }

    pub fn get_propagation_coefficients(
        &self,
        om: f64,
        k: f64,
        polarization: Polarization,
        a: Complex<f64>,
        b: Complex<f64>,
    ) -> Vec<LayerCoefficientVector> {
        match self.backend {
            BackEnd::Transfer => {
                get_propagation_coefficients_transfer(&self.layers, om, k, polarization, a, b)
            }
            BackEnd::Scattering => {
                panic!("Not implemented yet")
            }
        }
    }

    pub fn get_grid_data(&self) -> GridData {
        let xstart = -self.layers[0].d;
        let xend = self.layers.iter().map(|l| l.d).sum::<f64>() + xstart;
        let xgrid: Vec<f64> = iter_num_tools::arange(xstart..xend, self.plot_step).collect();
        let grid_starts: Vec<f64> = self.layers.iter().map(|l| l.d).collect();
        let grid_starts: Vec<f64> = [vec![0.0_f64], grid_starts].concat();
        let mut grid_starts: Vec<f64> = cumsum(&grid_starts).iter().map(|x| x + xstart).collect();
        let mut grid_istarts: Vec<usize> = vec![0];
        let mut slice_iter = grid_starts.iter();
        let _ = slice_iter.next();
        let mut start = slice_iter.next().unwrap();
        for (i, x) in xgrid.iter().enumerate() {
            if x >= start {
                grid_istarts.push(i);
                start = slice_iter.next().unwrap();
            }
        }
        grid_istarts.push(xgrid.len());

        grid_starts[0] = 0.0;

        GridData {
            xplot: xgrid,
            xstarts: grid_starts,
            ixstarts: grid_istarts,
        }
    }

    fn get_field_componet(
        &self,
        coefficient_vector: &Vec<LayerCoefficientVector>,
        grid_data: &GridData,
        om: f64,
        k: f64,
    ) -> Vec<Complex<f64>> {
        let x = grid_data.xplot.clone();

        let mut field_vectors: Vec<Complex64> = Vec::new();
        for (i, (istart, iend)) in zip(
            grid_data.ixstarts.clone(),
            grid_data.ixstarts.clone().iter().skip(1),
        )
        .enumerate()
        {
            let xstart = grid_data.xstarts[i];
            let coefficients = coefficient_vector[i];
            let layer = &self.layers[i];
            let xslice: Vec<f64> = x[istart..*iend]
                .iter()
                .map(|x| x - xstart)
                .collect::<Vec<_>>();
            field_vectors.extend(get_field_slice(
                coefficients.a,
                coefficients.b,
                om,
                k,
                layer.n,
                xslice,
            ));
        }
        field_vectors
    }

    pub fn get_coefficient_all_components(
        &self,
        om: f64,
        k: f64,
        main_coefficients: Vec<LayerCoefficientVector>,
    ) -> (
        Vec<LayerCoefficientVector>,
        Vec<LayerCoefficientVector>,
        Vec<LayerCoefficientVector>,
        Vec<LayerCoefficientVector>,
        Vec<LayerCoefficientVector>,
        Vec<LayerCoefficientVector>,
    ) {
        let main1 = main_coefficients;
        let om = Complex::new(om, 0.0);
        let k = Complex::new(k, 0.0);
        let mut main2 = Vec::new();
        let mut main3 = Vec::new();
        let mut maink = Vec::new();
        let mut mainb = Vec::new();
        let zeros = vec![
            LayerCoefficientVector::new(Complex::new(0.0, 0.0), Complex::new(0.0, 0.0));
            self.layers.len()
        ];
        for (layer, coefficients) in zip(self.layers.iter(), main1.iter()) {
            let kpar = ((layer.n * om).powi(2) - k.powi(2)).sqrt();
            let n = Complex::new(layer.n, 0.0);
            main2.push(LayerCoefficientVector::new(
                -coefficients.a * k / kpar,
                coefficients.b * k / kpar,
            ));
            main3.push(LayerCoefficientVector::new(
                -coefficients.a * om * n.powi(2) / kpar,
                coefficients.b * om * n.powi(2) / kpar,
            ));
            maink.push(LayerCoefficientVector::new(
                coefficients.a * kpar / om,
                -coefficients.b * kpar / om,
            ));
            mainb.push(LayerCoefficientVector::new(
                -coefficients.a * k / om,
                -coefficients.b * k / om,
            ));
        }
        (main1, main2, main3, maink, mainb, zeros)
    }

    pub fn field(
        &self,
        om: f64,
        polarization: Polarization,
        mode: usize,
    ) -> Result<FieldData, String> {
        let neff = match self.neff(om, polarization, mode) {
            Ok(n) => n,
            Err(e) => return Err(e),
        };

        let coefficient_vector = self.get_propagation_coefficients(
            om,
            om * neff,
            polarization,
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
        );
        let grid_data = self.get_grid_data();

        let main_component =
            self.get_field_componet(&coefficient_vector, &grid_data, om, om * neff);
        let coefficients = self.get_coefficient_all_components(om, om * neff, coefficient_vector);

        let (main1, main2, main3, maink, mainb, zeros) = coefficients;
        let field1 = self.get_field_componet(&main1, &grid_data, om, om * neff);
        let fieldzeros = self.get_field_componet(&zeros, &grid_data, om, om * neff);

        let field_data = match polarization {
            Polarization::TE => {
                let fieldk = self.get_field_componet(&maink, &grid_data, om, om * neff);
                let fieldb = self.get_field_componet(&mainb, &grid_data, om, om * neff);

                FieldData {
                    x: grid_data.xplot.clone(),
                    Ex: fieldzeros.clone(),
                    Ey: field1,
                    Ez: fieldzeros.clone(),
                    Hx: fieldb.iter().map(|x| x / Z0).collect(),
                    Hy: fieldzeros.clone(),
                    Hz: fieldk.iter().map(|x| x / Z0).collect(),
                }
            }
            Polarization::TM => {
                let field2 = self.get_field_componet(&main2, &grid_data, om, om * neff);
                let field3 = self.get_field_componet(&main3, &grid_data, om, om * neff);
                FieldData {
                    x: grid_data.xplot.clone(),
                    Ex: field2,
                    Ey: fieldzeros.clone(),
                    Ez: field1,
                    Hx: fieldzeros.clone(),
                    Hy: field3.iter().map(|x| x / Z0).collect(),
                    Hz: fieldzeros.clone(),
                }
            }
        };

        Ok(field_data)
    }

    pub fn get_index(&self, grid_data: &GridData) -> Vec<f64> {
        let xgrid = grid_data.xplot.clone();
        let mut n = vec![self.layers[0].n; xgrid.len()];
        for (i, layer) in self.layers.iter().enumerate() {
            let start = grid_data.ixstarts[i];
            let end = grid_data.ixstarts[i + 1];
            n[start..end].iter_mut().for_each(|x| *x = layer.n);
        }
        n
    }
}

pub fn find_minmax_n(layers: &Vec<Layer>) -> (f64, f64) {
    let mut min_n = layers[0].n;
    let mut max_n = layers[0].n;
    for layer in layers.iter() {
        if layer.n < min_n {
            min_n = layer.n;
        }
        if layer.n > max_n {
            max_n = layer.n;
        }
    }
    (min_n, max_n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{f64::consts::PI, fmt::Debug};

    trait ApproxEqual {
        fn approx_eq(&self, other: &Self, epsilon: f64) -> bool;
    }

    impl ApproxEqual for f64 {
        fn approx_eq(&self, other: &Self, epsilon: f64) -> bool {
            (self - other).abs() < epsilon
        }
    }

    impl ApproxEqual for Complex<f64> {
        fn approx_eq(&self, other: &Self, epsilon: f64) -> bool {
            self.re.approx_eq(&other.re, epsilon) && self.im.approx_eq(&other.im, epsilon)
        }
    }

    impl ApproxEqual for LayerCoefficientVector {
        fn approx_eq(&self, other: &Self, epsilon: f64) -> bool {
            self.a.approx_eq(&other.a, epsilon) && self.b.approx_eq(&other.b, epsilon)
        }
    }

    fn assert_vec_approx_equal<T>(vec1: &[T], vec2: &[T], epsilon: f64)
    where
        T: ApproxEqual,
        T: Debug,
    {
        assert_eq!(vec1.len(), vec2.len(), "Vectors have different lengths");
        for (i, (a, b)) in vec1.iter().zip(vec2.iter()).enumerate() {
            assert!(
                a.approx_eq(b, epsilon),
                "Values at index {} are not approximately equal: {:?} != {:?}",
                i,
                a,
                b
            );
        }
    }

    fn create_slab_multilayer() -> MultiLayer {
        let layers: Vec<Layer> = vec![
            Layer::new(1.0, 1.0),
            Layer::new(2.0, 0.6),
            Layer::new(1.0, 1.0),
        ];
        MultiLayer::new(layers)
    }

    fn create_coupled_slab_multilayer() -> MultiLayer {
        let layers: Vec<Layer> = vec![
            Layer::new(1.0, 1.0),
            Layer::new(2.0, 0.6),
            Layer::new(1.0, 2.0),
            Layer::new(2.0, 0.6),
            Layer::new(1.0, 1.0),
        ];
        MultiLayer::new(layers)
    }

    #[test]
    fn test_scattering_slab_te() {
        let mut multi_layer = create_slab_multilayer();
        let om = 2.0 * PI / 1.55;
        multi_layer.set_backend(BackEnd::Scattering);

        let neff = multi_layer.solve(om, Polarization::TE);
        let expected_neff = vec![1.804297363, 1.191174978];
        assert_vec_approx_equal(&neff, &expected_neff, 1e-9);
    }

    #[test]
    fn test_scattering_slab_tm() {
        let mut multi_layer = create_slab_multilayer();
        let om = 2.0 * PI / 1.55;
        multi_layer.set_backend(BackEnd::Scattering);

        let neff = multi_layer.solve(om, Polarization::TM);
        let expected_neff = vec![1.657017474, 1.028990635];
        assert_vec_approx_equal(&neff, &expected_neff, 1e-9);
    }

    #[test]
    fn test_scattering_coupled_slab_te() {
        let mut multi_layer = create_coupled_slab_multilayer();
        let om = 2.0 * PI / 1.55;
        multi_layer.set_backend(BackEnd::Scattering);

        let neff = multi_layer.solve(om, Polarization::TE);
        let expected_neff = vec![1.804297929, 1.804296798, 1.192052932, 1.190270579];
        assert_vec_approx_equal(&neff, &expected_neff, 1e-9);
    }

    #[test]
    fn test_scattering_coupled_slab_tm() {
        let mut multi_layer = create_coupled_slab_multilayer();
        let om = 2.0 * PI / 1.55;
        multi_layer.set_backend(BackEnd::Scattering);

        let neff = multi_layer.solve(om, Polarization::TM);
        let expected_neff = vec![1.657019473, 1.657015474, 1.035192425, 1.019866805];
        assert_vec_approx_equal(&neff, &expected_neff, 1e-9);
    }

    #[test]
    fn test_transfer_slab_te() {
        let mut multi_layer = create_slab_multilayer();
        let om = 2.0 * PI / 1.55;
        multi_layer.set_backend(BackEnd::Transfer);

        let neff = multi_layer.solve(om, Polarization::TE);
        let expected_neff = vec![1.804297363, 1.191174978];
        assert_vec_approx_equal(&neff, &expected_neff, 1e-9);
    }

    #[test]
    fn test_transfer_slab_tm() {
        let mut multi_layer = create_slab_multilayer();
        let om = 2.0 * PI / 1.55;
        multi_layer.set_backend(BackEnd::Transfer);

        let neff = multi_layer.solve(om, Polarization::TM);
        let expected_neff = vec![1.657017474, 1.028990635];
        assert_vec_approx_equal(&neff, &expected_neff, 1e-9);
    }

    #[test]
    fn test_transfer_coupled_slab_te() {
        let mut multi_layer = create_coupled_slab_multilayer();
        let om = 2.0 * PI / 1.55;
        multi_layer.set_backend(BackEnd::Transfer);

        let neff = multi_layer.solve(om, Polarization::TE);
        let expected_neff = vec![1.804297929, 1.804296798, 1.192052932, 1.190270579];
        assert_vec_approx_equal(&neff, &expected_neff, 1e-9);
    }

    #[test]
    fn test_transfer_coupled_slab_tm() {
        let mut multi_layer = create_coupled_slab_multilayer();
        let om = 2.0 * PI / 1.55;
        multi_layer.set_backend(BackEnd::Transfer);

        let neff = multi_layer.solve(om, Polarization::TM);
        let expected_neff = vec![1.657019473, 1.657015474, 1.035192425, 1.019866805];
        assert_vec_approx_equal(&neff, &expected_neff, 1e-9);
    }

    #[test]
    fn test_transfer_field_slab() {
        let mut multi_layer = create_slab_multilayer();
        let om = 2.0 * PI / 1.55;
        multi_layer.set_backend(BackEnd::Transfer);

        let n = multi_layer.neff(om, Polarization::TE, 0).unwrap_or(0.0);
        let ref_coefficients = vec![
            LayerCoefficientVector::new(Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)),
            LayerCoefficientVector::new(
                Complex::new(0.5, -0.870271563),
                Complex::new(0.5, 0.870271563),
            ),
            LayerCoefficientVector::new(Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)),
        ];
        let coefficients = multi_layer.get_propagation_coefficients(
            om,
            om * n,
            Polarization::TE,
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
        );
        assert_vec_approx_equal(&coefficients, &ref_coefficients, 1e-9);

        let n = multi_layer.neff(om, Polarization::TM, 0).unwrap_or(0.0);
        let ref_coefficients = vec![
            LayerCoefficientVector::new(Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)),
            LayerCoefficientVector::new(
                Complex::new(0.5, 0.105955587),
                Complex::new(0.5, -0.105955587),
            ),
            LayerCoefficientVector::new(Complex::new(-1.0, 0.0), Complex::new(0.0, 0.0)),
        ];
        let coefficients = multi_layer.get_propagation_coefficients(
            om,
            om * n,
            Polarization::TM,
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
        );
        assert_vec_approx_equal(&coefficients, &ref_coefficients, 2e-9);
    }
}
