extern crate find_peaks;
extern crate itertools;

use std::f64::consts::PI;

use crate::enums::BackEnd;
use crate::enums::Polarization;
use crate::layer::Layer;
use crate::scattering_matrix::calculate_s_matrix;
// use crate::scattering_matrix::ScatteringMatrix;
use find_peaks::PeakFinder;
use num_complex::Complex;

pub struct MultiLayer {
    layers: Vec<Layer>,
    iteration: usize,
    solution_threshold: f64,
    backend: BackEnd,
}

impl MultiLayer {
    pub fn new(layers: Vec<Layer>) -> MultiLayer {
        MultiLayer {
            layers: layers,
            iteration: 3,
            solution_threshold: 2.0,
            backend: BackEnd::Scattering,
        }
    }

    pub fn set_backend(&mut self, backend: BackEnd) {
        self.backend = backend;
    }

    pub fn set_iteration(&mut self, iteration: usize) {
        self.iteration = iteration;
    }

    pub fn set_solution_threshold(&mut self, solution_threshold: f64) {
        self.solution_threshold = solution_threshold;
    }

    // pub fn calculate_s_matrix(&self, om: f64, k: f64) -> ScatteringMatrix {
    //     calculate_s_matrix(&self.layers, om, k)
    // }

    // pub fn calculate_structure_determinant(&self, om: f64, k: f64) -> Complex<f64> {
    //     self.calculate_s_matrix(om, k).determinant()
    // }

    fn characteristic_function(&self, om: f64, k: f64, polarization: Polarization) -> Complex<f64> {
        match self.backend {
            BackEnd::Scattering => {
                calculate_s_matrix(&self.layers, om, k, polarization).determinant()
            }
            BackEnd::Transfer => {
                panic!("Transfer matrix is not implemented yet")
            }
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
        let peaks = peak_finder
            .with_min_height(self.solution_threshold)
            .with_min_difference(1e-3)
            .find_peaks();

        let ksolutions = {
            peaks
                .into_iter()
                .map(|p| kv.get(p.middle_position()))
                .flatten()
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

        for step in [1e-3, 1e-6, 1e-9, 1e-12].into_iter().take(self.iteration) {
            ksolutions.clear();
            for (_kmin, _kmax) in solution_backets {
                let _solutions = self.solve_step(om, _kmin, _kmax, step, polarization);
                ksolutions.extend(_solutions);
            }
            solution_backets = ksolutions
                .iter()
                .map(|k| (k - step, k + step))
                .collect::<Vec<_>>();
        }

        let mut n_solutions = ksolutions.into_iter().map(|k| k / om).collect::<Vec<_>>();

        n_solutions.sort_by(|a, b| b.partial_cmp(a).unwrap());

        for p in n_solutions.iter() {
            println!("{:?}", p);
        }
        n_solutions
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

// pub fn calculate_structure_determinant(layers: &Vec<Layer>, om: f64, k: f64) -> Complex<f64> {
//     let s_matrix = calculate_s_matrix(layers, om, k);
//     s_matrix.determinant()
// }

#[cfg(test)]
mod tests {
    use super::*;

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
}
