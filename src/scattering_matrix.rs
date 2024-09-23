extern crate itertools;
extern crate num_complex;

use itertools::izip;
use num_complex::Complex;

use crate::enums::Polarization;
use crate::layer::Layer;

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
    pub fn matrix_interface_tm(n1: f64, n2: f64, om: f64, k: f64) -> ScatteringMatrix {
        let n1 = Complex::new(n1, 0.0);
        let n2 = Complex::new(n2, 0.0);
        let om = Complex::new(om, 0.0);
        let k = Complex::new(k, 0.0);

        let k1 = n2.powi(2) * ((om * n1).powi(2) - k.powi(2)).sqrt();
        let k2 = n1.powi(2) * ((om * n2).powi(2) - k.powi(2)).sqrt();
        ScatteringMatrix {
            s11: 2.0 * k2 / (k1 + k2),
            s12: (k2 - k1) / (k1 + k2),
            s21: (k1 - k2) / (k1 + k2),
            s22: 2.0 * k1 / (k1 + k2),
        }
    }

    pub fn matrix_interface(
        n1: f64,
        n2: f64,
        om: f64,
        k: f64,
        polarization: Polarization,
    ) -> ScatteringMatrix {
        match polarization {
            Polarization::TE => ScatteringMatrix::matrix_interface_te(n1, n2, om, k),
            Polarization::TM => ScatteringMatrix::matrix_interface_tm(n1, n2, om, k),
        }
    }
}

pub fn calculate_s_matrix(
    layers: &Vec<Layer>,
    om: f64,
    k: f64,
    polarization: Polarization,
) -> ScatteringMatrix {
    let mut result =
        ScatteringMatrix::matrix_interface(layers[0].n, layers[1].n, om, k, polarization);
    // println!("Interface_matrix: {:?}", result);
    for (layer1, layer2) in izip!(layers.iter().skip(1), layers.iter().skip(2)) {
        let matrix = ScatteringMatrix::matrix_propagation(layer1.n, layer1.d, om, k);
        result = result.compose(matrix);
        let matrix = ScatteringMatrix::matrix_interface(layer1.n, layer2.n, om, k, polarization);
        result = result.compose(matrix);
    }
    result
}
