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

    pub fn matrix_propagation<T, U, V, W>(n: T, d: U, om: V, k: W) -> TransferMatrix
    where
        T: Into<Complex<f64>>,
        U: Into<Complex<f64>>,
        V: Into<Complex<f64>>,
        W: Into<Complex<f64>>,
    {
        let om: Complex<f64> = om.into();
        let k: Complex<f64> = k.into();
        let d: Complex<f64> = d.into();
        let n: Complex<f64> = n.into();
        let a = ((om.clone() * n.clone()).powi(2) - k.clone().powi(2)).sqrt();
        let phase_positive = Complex::new(0.0, 1.0) * a.clone() * d.clone();
        let phase_negative = Complex::new(0.0, -1.0) * a * d;
        TransferMatrix {
            t11: phase_positive.exp(),
            t12: Complex::new(0.0, 0.0),
            t21: Complex::new(0.0, 0.0),
            t22: phase_negative.exp(),
        }
    }

    pub fn matrix_interface_te<T, U, V, W>(n1: T, n2: U, om: V, k: W) -> TransferMatrix
    where
        T: Into<Complex<f64>>,
        U: Into<Complex<f64>>,
        V: Into<Complex<f64>>,
        W: Into<Complex<f64>>,
    {
        let n1: Complex<f64> = n1.into();
        let n2: Complex<f64> = n2.into();
        let om: Complex<f64> = om.into();
        let k: Complex<f64> = k.into();

        let k1 = ((om * n1).powi(2) - k.powi(2)).sqrt();
        let k2 = ((om * n2).powi(2) - k.powi(2)).sqrt();
        TransferMatrix {
            t11: 0.5 * (k2 + k1) / k2,
            t12: 0.5 * (k2 - k1) / k2,
            t21: 0.5 * (k2 - k1) / k2,
            t22: 0.5 * (k2 + k1) / k2,
        }
    }

    pub fn matrix_interface_tm<T, U, V, W>(n1: T, n2: U, om: V, k: W) -> TransferMatrix
    where
        T: Into<Complex<f64>>,
        U: Into<Complex<f64>>,
        V: Into<Complex<f64>>,
        W: Into<Complex<f64>>,
    {
        let n1: Complex<f64> = n1.into();
        let n2: Complex<f64> = n2.into();
        let om: Complex<f64> = om.into();
        let k: Complex<f64> = k.into();

        let k1 = n2.powi(2) * ((om * n1).powi(2) - k.powi(2)).sqrt();
        let k2 = n1.powi(2) * ((om * n2).powi(2) - k.powi(2)).sqrt();
        TransferMatrix {
            t11: 0.5 * (k2 + k1) / k1,
            t12: 0.5 * (k1 - k2) / k1,
            t21: 0.5 * (k1 - k2) / k1,
            t22: 0.5 * (k2 + k1) / k1,
        }
    }

    pub fn matrix_interface<T, U, V, W>(
        n1: T,
        n2: U,
        om: V,
        k: W,
        polarization: Polarization,
    ) -> TransferMatrix
    where
        T: Into<Complex<f64>>,
        U: Into<Complex<f64>>,
        V: Into<Complex<f64>>,
        W: Into<Complex<f64>>,
    {
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

pub fn calculate_t_matrix<T, U>(
    layers: &Vec<Layer>,
    om: T,
    k: U,
    polarization: Polarization,
) -> TransferMatrix
where
    T: Into<Complex<f64>> + Copy,
    U: Into<Complex<f64>> + Copy,
{
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
    let transfer_matrix =
        TransferMatrix::matrix_interface(layers[0].n, layers[1].n, om, k, polarization);
    current_coefficients = transfer_matrix.multiply(&current_coefficients);
    coefficients.push(current_coefficients);
    for (layer1, layer2) in zip(layers.iter().skip(1), layers.iter().skip(2)) {
        let propagation_matrix = TransferMatrix::matrix_propagation(layer1.n, layer1.d, om, k);
        current_coefficients = propagation_matrix.multiply(&current_coefficients);
        let interface_matrix =
            TransferMatrix::matrix_interface(layer1.n, layer2.n, om, k, polarization);
        current_coefficients = interface_matrix.multiply(&current_coefficients);
        coefficients.push(current_coefficients);
    }
    coefficients
}
