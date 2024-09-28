extern crate find_peaks;
extern crate itertools;

use pyo3::prelude::*;
use std::cmp::Ordering;

use crate::enums::BackEnd;
use crate::enums::Polarization;
use crate::layer::Layer;
use crate::scattering_matrix::calculate_s_matrix;
use crate::transfer_matrix::calculate_t_matrix;
use find_peaks::PeakFinder;
use num_complex::Complex;

#[pyclass]
pub struct MultiLayer {
    layers: Vec<Layer>,
    iteration: usize,
    backend: BackEnd,
    required_accuracy: i32,
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
        mode: Option<u8>,
    ) -> f64 {
        let polarization = polarization.unwrap_or(Polarization::TE);
        let mode = mode.unwrap_or(0);
        self.neff(omega, polarization, mode)
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

    pub fn neff(&self, om: f64, polarization: Polarization, mode: u8) -> f64 {
        let n_solutions = self.solve(om, polarization);
        let n_eff = n_solutions.get(mode as usize).unwrap_or_else(|| &0.0);
        *n_eff
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
    use std::f64::consts::PI;

    fn approx_equal(a: f64, b: f64, epsilon: f64) -> bool {
        (a - b).abs() < epsilon
    }

    fn assert_vec_approx_equal(vec1: &[f64], vec2: &[f64], epsilon: f64) {
        assert_eq!(vec1.len(), vec2.len(), "Vectors have different lengths");
        for (i, (a, b)) in vec1.iter().zip(vec2.iter()).enumerate() {
            assert!(
                approx_equal(*a, *b, epsilon),
                "Values at index {} are not approximately equal: {} != {}",
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
}
