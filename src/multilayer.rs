extern crate find_peaks;
extern crate itertools;

use std::cmp::Ordering;
use std::iter::zip;

use crate::enums::BackEnd;
use crate::enums::Polarization;
use crate::layer::Layer;
use crate::scattering_matrix::calculate_s_matrix;
use crate::transfer_matrix::calculate_t_matrix;
// use crate::scattering_matrix::ScatteringMatrix;
use find_peaks::PeakFinder;
use num_complex::Complex;

#[derive(Clone)]
pub struct StepSettings {
    pub step: f64,
    pub threshold: f64,
}

pub struct MultiLayer {
    layers: Vec<Layer>,
    iteration: usize,
    backend: BackEnd,
    settings: Vec<StepSettings>,
}

impl MultiLayer {
    pub fn new(layers: Vec<Layer>) -> MultiLayer {
        let mut multilayer = MultiLayer {
            layers: layers,
            iteration: 3,
            backend: BackEnd::Transfer,
            settings: vec![],
        };
        multilayer.set_backend(BackEnd::Transfer);
        multilayer
    }

    pub fn set_backend(&mut self, backend: BackEnd) {
        self.backend = backend;
        match backend {
            BackEnd::Scattering => {
                self.settings = vec![
                    StepSettings {
                        step: 1e-3,
                        threshold: 2.0,
                    },
                    StepSettings {
                        step: 1e-6,
                        threshold: 5.0,
                    },
                    StepSettings {
                        step: 1e-9,
                        threshold: 7.0,
                    },
                    StepSettings {
                        step: 1e-12,
                        threshold: 9.0,
                    },
                ];
            }
            BackEnd::Transfer => {
                self.settings = vec![
                    StepSettings {
                        step: 1e-3,
                        threshold: 0.0,
                    },
                    StepSettings {
                        step: 1e-6,
                        threshold: 3.0,
                    },
                    StepSettings {
                        step: 1e-9,
                        threshold: 6.0,
                    },
                    StepSettings {
                        step: 1e-12,
                        threshold: 6.0,
                    },
                ];
            }
        }
    }

    pub fn set_iteration(&mut self, iteration: usize) {
        self.iteration = iteration;
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

        for step in self.settings.clone().into_iter().take(self.iteration) {
            ksolutions.clear();
            for (_kmin, _kmax) in solution_backets {
                let _solutions =
                    self.solve_step(om, _kmin, _kmax, step.step, step.threshold, polarization);
                ksolutions.extend(_solutions);
            }
            solution_backets = ksolutions
                .iter()
                .map(|k| (k - step.step, k + step.step))
                .collect::<Vec<_>>();
        }

        for _k in ksolutions.iter() {
            println!("k = {}", _k);
        }

        let mut n_solutions = ksolutions.into_iter().map(|k| k / om).collect::<Vec<_>>();

        n_solutions.sort_by(|a, b| b.partial_cmp(a).unwrap_or_else(|| Ordering::Equal));
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
