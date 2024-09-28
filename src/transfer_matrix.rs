extern crate itertools;
extern crate num_complex;

use crate::enums::Polarization;
use crate::layer::{Layer, LayerCoefficientVector};
use num_complex::Complex;
use std::iter::zip;

#[derive(Debug)]
pub struct TransferMatrix {
    t11: Complex<f64>,
    t12: Complex<f64>,
    t21: Complex<f64>,
    pub t22: Complex<f64>,
}

impl TransferMatrix {
    pub fn compose(self, other: TransferMatrix) -> TransferMatrix {
        TransferMatrix {
            t11: self.t11 * other.t11 + self.t12 * other.t21,
            t12: self.t11 * other.t12 + self.t12 * other.t22,
            t21: self.t21 * other.t11 + self.t22 * other.t21,
            t22: self.t21 * other.t12 + self.t22 * other.t22,
        }
    }

    pub fn matrix_start() -> TransferMatrix {
        TransferMatrix {
            t11: Complex::new(1.0, 0.0),
            t12: Complex::new(0.0, 0.0),
            t21: Complex::new(0.0, 0.0),
            t22: Complex::new(1.0, 0.0),
        }
    }

    pub fn matrix_propagation(n: f64, d: f64, om: f64, k: f64) -> TransferMatrix {
        let om = Complex::new(om, 0.0);
        let k = Complex::new(k, 0.0);
        let d = Complex::new(d, 0.0);
        let a = ((om * n).powi(2) - k.powi(2)).sqrt();
        let phase_positive = Complex::new(0.0, 1.0) * a * d;
        let phase_negative = Complex::new(0.0, -1.0) * a * d;
        TransferMatrix {
            t11: phase_positive.exp(),
            t12: Complex::new(0.0, 0.0),
            t21: Complex::new(0.0, 0.0),
            t22: phase_negative.exp(),
        }
    }

    pub fn matrix_interface_te(n1: f64, n2: f64, om: f64, k: f64) -> TransferMatrix {
        let n1 = Complex::new(n1, 0.0);
        let n2 = Complex::new(n2, 0.0);
        let om = Complex::new(om, 0.0);
        let k = Complex::new(k, 0.0);

        let k1 = ((om * n1).powi(2) - k.powi(2)).sqrt();
        let k2 = ((om * n2).powi(2) - k.powi(2)).sqrt();
        TransferMatrix {
            t11: 0.5 * (k2 + k1) / k2,
            t12: 0.5 * (k2 - k1) / k2,
            t21: 0.5 * (k2 - k1) / k2,
            t22: 0.5 * (k2 + k1) / k2,
        }
    }

    pub fn matrix_interface_tm(n1: f64, n2: f64, om: f64, k: f64) -> TransferMatrix {
        let n1 = Complex::new(n1, 0.0);
        let n2 = Complex::new(n2, 0.0);
        let om = Complex::new(om, 0.0);
        let k = Complex::new(k, 0.0);

        let k1 = n2.powi(2) * ((om * n1).powi(2) - k.powi(2)).sqrt();
        let k2 = n1.powi(2) * ((om * n2).powi(2) - k.powi(2)).sqrt();
        TransferMatrix {
            t11: 0.5 * (k2 + k1) / k1,
            t12: 0.5 * (k1 - k2) / k1,
            t21: 0.5 * (k1 - k2) / k1,
            t22: 0.5 * (k2 + k1) / k1,
        }
    }

    pub fn matrix_interface(
        n1: f64,
        n2: f64,
        om: f64,
        k: f64,
        polarization: Polarization,
    ) -> TransferMatrix {
        match polarization {
            Polarization::TE => TransferMatrix::matrix_interface_te(n1, n2, om, k),
            Polarization::TM => TransferMatrix::matrix_interface_tm(n1, n2, om, k),
        }
    }

    pub fn multiply(&self, coefficient_vector: &LayerCoefficientVector) -> LayerCoefficientVector {
        LayerCoefficientVector {
            a: self.t11 * coefficient_vector.a + self.t12 * coefficient_vector.b,
            b: self.t21 * coefficient_vector.a + self.t22 * coefficient_vector.b,
        }
    }
}

pub fn calculate_t_matrix(
    layers: &Vec<Layer>,
    om: f64,
    k: f64,
    polarization: Polarization,
) -> TransferMatrix {
    let mut result =
        TransferMatrix::matrix_interface(layers[0].n, layers[1].n, om, k, polarization);
    for (layer1, layer2) in zip(layers.iter().skip(1), layers.iter().skip(2)) {
        let matrix = TransferMatrix::matrix_propagation(layer1.n, layer1.d, om, k);
        result = result.compose(matrix);
        let matrix = TransferMatrix::matrix_interface(layer1.n, layer2.n, om, k, polarization);
        result = result.compose(matrix);
    }
    result
}

pub fn get_propagation_coefficients_transfer(
    layers: &Vec<Layer>,
    om: f64,
    k: f64,
    polarization: Polarization,
    a: Complex<f64>,
    b: Complex<f64>,
) -> Vec<LayerCoefficientVector> {
    let mut coefficients: Vec<LayerCoefficientVector> = Vec::new();
    let mut current_coefficients = LayerCoefficientVector::new(a, b);
    coefficients.push(current_coefficients);
    let mut transfer_matrix =
        TransferMatrix::matrix_interface(layers[0].n, layers[1].n, om, k, polarization);
    current_coefficients = transfer_matrix.multiply(&current_coefficients);
    coefficients.push(current_coefficients);
    for (layer1, layer2) in zip(layers.iter().skip(1), layers.iter().skip(2)) {
        let mut transfer_matrix = TransferMatrix::matrix_propagation(layer1.n, layer1.d, om, k);
        let matrix = TransferMatrix::matrix_interface(layer1.n, layer2.n, om, k, polarization);
        transfer_matrix = transfer_matrix.compose(matrix);
        current_coefficients = transfer_matrix.multiply(&current_coefficients);
        coefficients.push(current_coefficients);
    }
    coefficients
}
