extern crate find_peaks;
extern crate itertools;
extern crate num_complex;

use std::fs::File;
use std::io::Write;
use std::{iter::zip, ops::Deref};

use find_peaks::PeakFinder;
use itertools::izip;
use num_complex::Complex;
use plotters::data;

pub struct Layer {
    n: f64,
    d: f64,
}

impl Layer {
    pub fn new(n: f64, d: f64) -> Layer {
        Layer { n, d }
    }
}

#[derive(Debug)]
pub struct ScatteringMatrix {
    s11: Complex<f64>,
    s12: Complex<f64>,
    s21: Complex<f64>,
    s22: Complex<f64>,
}

impl ScatteringMatrix {
    pub fn compose(self, other: ScatteringMatrix) -> ScatteringMatrix {
        let denominator = 1.0 - self.s12 * other.s21;
        ScatteringMatrix {
            s11: self.s11 * other.s11 / denominator,
            s12: other.s12 + other.s11 * self.s12 * other.s22 / denominator,
            s21: self.s21 + self.s22 * other.s21 * self.s11 / denominator,
            s22: self.s22 * other.s22 / denominator,
        }
    }

    pub fn determinant(&self) -> Complex<f64> {
        self.s11 * self.s22 - self.s12 * self.s21
    }

    pub fn matrix_start() -> ScatteringMatrix {
        ScatteringMatrix {
            s11: Complex::new(1.0, 0.0),
            s12: Complex::new(0.0, 0.0),
            s21: Complex::new(0.0, 0.0),
            s22: Complex::new(1.0, 0.0),
        }
    }

    pub fn matrix_propagation(n: f64, d: f64, om: f64, k: f64) -> ScatteringMatrix {
        let om = Complex::new(om, 0.0);
        let k = Complex::new(k, 0.0);
        let d = Complex::new(d, 0.0);
        let a = ((om * n).powi(2) - k.powi(2)).sqrt();
        let phase = Complex::new(0.0, 1.0) * a * d;
        ScatteringMatrix {
            s11: phase.exp(),
            s12: Complex::new(0.0, 0.0),
            s21: Complex::new(0.0, 0.0),
            s22: phase.exp(),
        }
    }

    pub fn matrix_interface_te(n1: f64, n2: f64, om: f64, k: f64) -> ScatteringMatrix {
        let n1 = Complex::new(n1, 0.0);
        let n2 = Complex::new(n2, 0.0);
        let om = Complex::new(om, 0.0);
        let k = Complex::new(k, 0.0);

        let k1 = ((om * n1).powi(2) - k.powi(2)).sqrt();
        let k2 = ((om * n2).powi(2) - k.powi(2)).sqrt();
        ScatteringMatrix {
            s11: 2.0 * k2 / (k1 + k2),
            s12: (k2 - k1) / (k1 + k2),
            s21: (k1 - k2) / (k1 + k2),
            s22: 2.0 * k1 / (k1 + k2),
        }
    }
}

pub struct MultiLayer {
    layers: Vec<Layer>,
    iteration: usize,
    solution_threshold: f64,
    plot_index: i32,
}

impl MultiLayer {
    pub fn new(layers: Vec<Layer>) -> MultiLayer {
        MultiLayer {
            layers: layers,
            iteration: 3,
            solution_threshold: 1.0,
            plot_index: 0,
        }
    }

    pub fn set_iteration(&mut self, iteration: usize) {
        self.iteration = iteration;
    }

    pub fn set_solution_threshold(&mut self, solution_threshold: f64) {
        self.solution_threshold = solution_threshold;
    }

    pub fn calculate_s_matrix(&self, om: f64, k: f64) -> ScatteringMatrix {
        calculate_s_matrix(&self.layers, om, k)
    }

    pub fn calculate_structure_determinant(&self, om: f64, k: f64) -> Complex<f64> {
        self.calculate_s_matrix(om, k).determinant()
    }

    fn find_minmax_n(&self) -> (f64, f64) {
        find_minmax_n(&self.layers)
    }

    fn solve_step(&mut self, om: f64, k_min: f64, k_max: f64, step: f64) -> Vec<f64> {
        let kv: Vec<f64> = iter_num_tools::arange(k_min..k_max, step).collect();
        let det: Vec<f64> = kv
            .clone()
            .into_iter()
            .map(|k| self.calculate_structure_determinant(om, k).norm().log10())
            .collect();

        {
            let mut file = File::create(format!("det_{:}.txt", self.plot_index)).unwrap();
            self.plot_index += 1;
            for (k, a) in zip(&kv, &det) {
                // println!("{:?} {:?}", k, a);
                file.write_all(format!("{:?} {:?}\n", k, a).as_bytes())
                    .unwrap();
            }
        }
        let mut peak_finder = PeakFinder::new(&det);
        let peaks = peak_finder
            .with_min_height(self.solution_threshold)
            .with_min_difference(1e-3)
            .find_peaks();

        println!("{:?}", peaks);

        let ksolutions = {
            peaks
                .into_iter()
                .map(|p| kv.get(p.middle_position()))
                .flatten()
                .collect::<Vec<_>>()
        };

        println!("{:?}", ksolutions);

        ksolutions.into_iter().map(|k| *k).collect::<Vec<_>>()
    }

    pub fn solve(&mut self, om: f64) -> Vec<f64> {
        let (min_n, max_n) = self.find_minmax_n();
        let k_min = om * min_n + 1e-9;
        let k_max = om * max_n - 1e-9;

        let mut solution_backets = vec![(k_min, k_max)];

        let mut ksolutions = Vec::new();

        for step in [1e-3, 1e-6, 1e-9, 1e-12].into_iter().take(self.iteration) {
            ksolutions.clear();
            for (_kmin, _kmax) in solution_backets {
                let _solutions = self.solve_step(om, _kmin, _kmax, step);
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

pub fn calculate_s_matrix(layers: &Vec<Layer>, om: f64, k: f64) -> ScatteringMatrix {
    let mut result = ScatteringMatrix::matrix_interface_te(layers[0].n, layers[1].n, om, k);
    // println!("Interface_matrix: {:?}", result);
    for (layer1, layer2) in izip!(layers.iter().skip(1), layers.iter().skip(2)) {
        let matrix = ScatteringMatrix::matrix_propagation(layer1.n, layer1.d, om, k);
        // println!("Propagation_matrix layer n {}: {:?}", layer1.n, matrix);
        result = result.compose(matrix);
        // println!("Intermediate_matrix: {:?}", result);
        let matrix = ScatteringMatrix::matrix_interface_te(layer1.n, layer2.n, om, k);
        // println!("Interface_matrix: {:?}", matrix);
        result = result.compose(matrix);
        // println!("Intermediate_matrix: {:?}", result);
    }
    // println!("Final matrix: {:?}", result);
    // println!("");
    result
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

pub fn calculate_structure_determinant(layers: &Vec<Layer>, om: f64, k: f64) -> Complex<f64> {
    let s_matrix = calculate_s_matrix(layers, om, k);
    s_matrix.determinant()
}

fn find_maxima_xy(x_data: Vec<f64>, y_data: Vec<f64>) -> Vec<f64> {
    let mut going_up = false;
    let mut maxima: Vec<f64> = Vec::new();
    let mut current = y_data.get(0).expect("Empty array").log10();
    let next = y_data.get(1).expect("Empty array").log10();

    if next > current {
        going_up = true;
    }

    current = next;

    for (x, y) in izip!(x_data.into_iter().skip(1), y_data.iter().skip(2)) {
        let next = *y;
        if going_up {
            if next < current && current > 1.0 {
                going_up = false;
                maxima.push(x);
            }
        } else {
            if next > current {
                going_up = true;
            }
        }
        current = next;
    }
    maxima
}
