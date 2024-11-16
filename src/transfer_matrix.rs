//! Implementation of the transfer matrix method for multilayer structures.
extern crate itertools;
extern crate num_complex;

use crate::enums::Polarization;
use crate::layer::{Layer, LayerCoefficientVector};
use num_complex::Complex;
use std::iter::zip;

/// Struct representing the transfer matrix.
#[derive(Debug)]
pub struct TransferMatrix {
    /// t11 element of the matrix.
    t11: Complex<f64>,
    /// t12 element of the matrix.
    t12: Complex<f64>,
    /// t21 element of the matrix.
    t21: Complex<f64>,
    /// t22 element of the matrix.
    pub t22: Complex<f64>,
}

impl TransferMatrix {
    /// Returns a new TransferMatrix which is the composition of two matrices..
    /// # Arguments:
    /// * `other` - The other matrix to compose with.
    /// # Returns:
    /// The composition of the two matrices.
    pub fn compose(self, other: TransferMatrix) -> TransferMatrix {
        TransferMatrix {
            t11: self.t11 * other.t11 + self.t12 * other.t21,
            t12: self.t11 * other.t12 + self.t12 * other.t22,
            t21: self.t21 * other.t11 + self.t22 * other.t21,
            t22: self.t21 * other.t12 + self.t22 * other.t22,
        }
    }

    /// Creates the idendity transfer matrix. Useful to start the recursion..
    /// # Returns:
    /// The identity transfer matrix.
    pub fn matrix_start() -> TransferMatrix {
        TransferMatrix {
            t11: Complex::new(1.0, 0.0),
            t12: Complex::new(0.0, 0.0),
            t21: Complex::new(0.0, 0.0),
            t22: Complex::new(1.0, 0.0),
        }
    }

    /// Creates the transfer matrix for a homogeneous layer.
    /// # Arguments:
    /// * `n` - The refractive index of the layer.
    /// * `d` - The thickness of the layer.
    /// * `k0` - The vacuum wavevector.
    /// * `k` - The parallel component of the wavevector.
    /// # Returns:
    /// The propagation transfer matrix for the layer.
    pub fn matrix_propagation(n: f64, d: f64, k0: f64, k: f64) -> TransferMatrix {
        let k0 = Complex::new(k0, 0.0);
        let k = Complex::new(k, 0.0);
        let d = Complex::new(d, 0.0);
        let a = ((k0 * n).powi(2) - k.powi(2)).sqrt();
        let phase_positive = Complex::new(0.0, 1.0) * a * d;
        let phase_negative = Complex::new(0.0, -1.0) * a * d;
        TransferMatrix {
            t11: phase_positive.exp(),
            t12: Complex::new(0.0, 0.0),
            t21: Complex::new(0.0, 0.0),
            t22: phase_negative.exp(),
        }
    }

    /// Creates the transfer matrix representing the interafce between two layers for TE polarization.
    /// # Arguments:
    /// * `n1` - The refractive index of the first layer.
    /// * `n2` - The refractive index of the second layer.
    /// * `k0` - The vacuum wavevector.
    /// * `k` - The parallel component of the wavevector.
    /// # Returns:
    /// The interface transfer matrix for TE polarization.
    pub fn matrix_interface_te(n1: f64, n2: f64, k0: f64, k: f64) -> TransferMatrix {
        let n1 = Complex::new(n1, 0.0);
        let n2 = Complex::new(n2, 0.0);
        let k0 = Complex::new(k0, 0.0);
        let k = Complex::new(k, 0.0);

        let k1 = ((k0 * n1).powi(2) - k.powi(2)).sqrt();
        let k2 = ((k0 * n2).powi(2) - k.powi(2)).sqrt();
        TransferMatrix {
            t11: 0.5 * (k2 + k1) / k2,
            t12: 0.5 * (k2 - k1) / k2,
            t21: 0.5 * (k2 - k1) / k2,
            t22: 0.5 * (k2 + k1) / k2,
        }
    }

    /// Creates the transfer matrix representing the interafce between two layers for TM polarization./
    /// # Arguments:
    /// * `n1` - The refractive index of the first layer.
    /// * `n2` - The refractive index of the second layer.
    /// * `k0` - The vacuum wavevector.
    /// * `k` - The parallel component of the wavevector.
    /// # Returns:
    /// The interface transfer matrix for TM polarization.
    pub fn matrix_interface_tm(n1: f64, n2: f64, k0: f64, k: f64) -> TransferMatrix {
        let n1 = Complex::new(n1, 0.0);
        let n2 = Complex::new(n2, 0.0);
        let k0 = Complex::new(k0, 0.0);
        let k = Complex::new(k, 0.0);

        let k1 = n2.powi(2) * ((k0 * n1).powi(2) - k.powi(2)).sqrt();
        let k2 = n1.powi(2) * ((k0 * n2).powi(2) - k.powi(2)).sqrt();
        TransferMatrix {
            t11: 0.5 * (k2 + k1) / k1,
            t12: 0.5 * (k1 - k2) / k1,
            t21: 0.5 * (k1 - k2) / k1,
            t22: 0.5 * (k2 + k1) / k1,
        }
    }

    /// Creates the transfer matrix representing the interafce between two layers..
    /// # Arguments:
    /// * `n1` - The refractive index of the first layer.
    /// * `n2` - The refractive index of the second layer.
    /// * `k0` - The vacuum wavevector.
    /// * `k` - The parallel component of the wavevector.
    /// * `polarization` - The polarization of the light.
    /// # Returns:
    /// The interface transfer matrix.
    pub fn matrix_interface(
        n1: f64,
        n2: f64,
        k0: f64,
        k: f64,
        polarization: Polarization,
    ) -> TransferMatrix {
        match polarization {
            Polarization::TE => TransferMatrix::matrix_interface_te(n1, n2, k0, k),
            Polarization::TM => TransferMatrix::matrix_interface_tm(n1, n2, k0, k),
        }
    }

    /// Calculates the modal coefficient on the right of the matrix starting from the ones on the left.
    /// # Arguments:
    /// * `coefficient_vector` - The modal coefficient on the left of the matrix.
    /// # Returns:
    /// The modal coefficient on the right of the matrix.
    pub fn multiply(&self, coefficient_vector: &LayerCoefficientVector) -> LayerCoefficientVector {
        LayerCoefficientVector {
            a: self.t11 * coefficient_vector.a + self.t12 * coefficient_vector.b,
            b: self.t21 * coefficient_vector.a + self.t22 * coefficient_vector.b,
        }
    }
}

/// Calculates the transfer matrix of a multilayer system.
/// # Arguments:
/// * `layers` - The layers of the system.
/// * `k0` - The vacuum wavevector.
/// * `k` - The parallel component of the wavevector.
/// * `polarization` - The polarization of the light.
/// # Returns:
/// The transfer matrix of the system.
pub fn calculate_t_matrix(
    layers: &[Layer],
    k0: f64,
    k: f64,
    polarization: Polarization,
) -> TransferMatrix {
    let mut result =
        TransferMatrix::matrix_interface(layers[0].n, layers[1].n, k0, k, polarization);
    for (layer1, layer2) in zip(layers.iter().skip(1), layers.iter().skip(2)) {
        let matrix = TransferMatrix::matrix_propagation(layer1.n, layer1.d, k0, k);
        result = result.compose(matrix);
        let matrix = TransferMatrix::matrix_interface(layer1.n, layer2.n, k0, k, polarization);
        result = result.compose(matrix);
    }
    result
}

/// Calculates the modal coefficients in each layer of a multilayer system given the modal coefficients in the first layer..
/// # Arguments:
/// * `layers` - The layers of the system.
/// * `k0` - The vacuum wavevector.
/// * `k` - The parallel component of the wavevector.
/// * `polarization` - The polarization of the light.
/// * `a` - The forward modal coefficient of the first layer.
/// * `b` - The backward modal coefficient of the first layer.
/// # Returns:
/// The modal coefficients in each layer of the system.
pub fn get_propagation_coefficients_transfer(
    layers: &[Layer],
    k0: f64,
    k: f64,
    polarization: Polarization,
    a: Complex<f64>,
    b: Complex<f64>,
) -> Vec<LayerCoefficientVector> {
    let mut coefficients: Vec<LayerCoefficientVector> = Vec::new();
    let mut current_coefficients = LayerCoefficientVector::new(a, b);
    coefficients.push(current_coefficients);
    let transfer_matrix =
        TransferMatrix::matrix_interface(layers[0].n, layers[1].n, k0, k, polarization);
    current_coefficients = transfer_matrix.multiply(&current_coefficients);
    coefficients.push(current_coefficients);
    for (layer1, layer2) in zip(layers.iter().skip(1), layers.iter().skip(2)) {
        let propagation_matrix = TransferMatrix::matrix_propagation(layer1.n, layer1.d, k0, k);
        current_coefficients = propagation_matrix.multiply(&current_coefficients);
        let interface_matrix =
            TransferMatrix::matrix_interface(layer1.n, layer2.n, k0, k, polarization);
        current_coefficients = interface_matrix.multiply(&current_coefficients);
        coefficients.push(current_coefficients);
    }
    coefficients
}
