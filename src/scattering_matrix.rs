//! Implementation of the scattering matrix method for multilayer structures.
extern crate itertools;
extern crate num_complex;

use itertools::izip;
use num_complex::Complex;

use crate::enums::Polarization;
use crate::layer::Layer;

/// Struct representing the scattering matrix.
#[derive(Debug)]
pub struct ScatteringMatrix {
    /// s11 element of the scattering matrix.
    s11: Complex<f64>,
    /// s12 element of the scattering matrix.
    s12: Complex<f64>,
    /// s21 element of the scattering matrix.
    s21: Complex<f64>,
    /// s22 element of the scattering matrix.
    s22: Complex<f64>,
}

impl ScatteringMatrix {
    /// Compose two scattering matrices.
    /// # Arguments
    /// * `other` - The other scattering matrix to compose with.
    /// # Returns
    /// The composed scattering matrix.
    pub fn compose(self, other: ScatteringMatrix) -> ScatteringMatrix {
        let denominator = 1.0 - self.s12 * other.s21;
        ScatteringMatrix {
            s11: self.s11 * other.s11 / denominator,
            s12: other.s12 + other.s11 * self.s12 * other.s22 / denominator,
            s21: self.s21 + self.s22 * other.s21 * self.s11 / denominator,
            s22: self.s22 * other.s22 / denominator,
        }
    }

    /// Calculate the determinant of the scattering matrix.
    /// # Returns
    /// The determinant of the scattering matrix.
    pub fn determinant(&self) -> Complex<f64> {
        self.s11 * self.s22 - self.s12 * self.s21
    }

    /// Creates the identity scattering matrix.
    /// # Returns
    /// The identity scattering matrix.
    pub fn matrix_start() -> ScatteringMatrix {
        ScatteringMatrix {
            s11: Complex::new(1.0, 0.0),
            s12: Complex::new(0.0, 0.0),
            s21: Complex::new(0.0, 0.0),
            s22: Complex::new(1.0, 0.0),
        }
    }

    /// Creates a scattering matrix for propagation in a layer.
    /// # Arguments
    /// * `n` - The refractive index of the layer.
    /// * `d` - The thickness of the layer.
    /// * `k0` - The vacuum wavevector.
    /// * `k` - The parallel wavevector in the layer.
    /// # Returns
    /// The scattering matrix for propagation in the layer.
    pub fn matrix_propagation(n: f64, d: f64, k0: f64, k: f64) -> ScatteringMatrix {
        let k0 = Complex::new(k0, 0.0);
        let k = Complex::new(k, 0.0);
        let d = Complex::new(d, 0.0);
        let a = ((k0 * n).powi(2) - k.powi(2)).sqrt();
        let phase = Complex::new(0.0, 1.0) * a * d;
        ScatteringMatrix {
            s11: phase.exp(),
            s12: Complex::new(0.0, 0.0),
            s21: Complex::new(0.0, 0.0),
            s22: phase.exp(),
        }
    }

    /// Creates a scattering matrix for the interface between two layers for TE polarization.
    /// # Arguments
    /// * `n1` - The refractive index of the first layer.
    /// * `n2` - The refractive index of the second layer.
    /// * `k0` - The vacuum wavevector.
    /// * `k` - The parallel wavevector in the first layer.
    /// # Returns
    /// The scattering matrix for the interface between the two layers.
    pub fn matrix_interface_te(n1: f64, n2: f64, k0: f64, k: f64) -> ScatteringMatrix {
        let n1 = Complex::new(n1, 0.0);
        let n2 = Complex::new(n2, 0.0);
        let k0 = Complex::new(k0, 0.0);
        let k = Complex::new(k, 0.0);

        let k1 = ((k0 * n1).powi(2) - k.powi(2)).sqrt();
        let k2 = ((k0 * n2).powi(2) - k.powi(2)).sqrt();
        ScatteringMatrix {
            s11: 2.0 * k2 / (k1 + k2),
            s12: (k2 - k1) / (k1 + k2),
            s21: (k1 - k2) / (k1 + k2),
            s22: 2.0 * k1 / (k1 + k2),
        }
    }

    /// Creates a scattering matrix for the interface between two layers for TM polarization.
    /// # Arguments
    /// * `n1` - The refractive index of the first layer.
    /// * `n2` - The refractive index of the second layer.
    /// * `k0` - The vacuum wavevector.
    /// * `k` - The parallel wavevector in the first layer.
    /// # Returns
    /// The scattering matrix for the interface between the two layers.
    pub fn matrix_interface_tm(n1: f64, n2: f64, k0: f64, k: f64) -> ScatteringMatrix {
        let n1 = Complex::new(n1, 0.0);
        let n2 = Complex::new(n2, 0.0);
        let k0 = Complex::new(k0, 0.0);
        let k = Complex::new(k, 0.0);

        let k1 = n2.powi(2) * ((k0 * n1).powi(2) - k.powi(2)).sqrt();
        let k2 = n1.powi(2) * ((k0 * n2).powi(2) - k.powi(2)).sqrt();
        ScatteringMatrix {
            s11: 2.0 * k2 / (k1 + k2),
            s12: (k2 - k1) / (k1 + k2),
            s21: (k1 - k2) / (k1 + k2),
            s22: 2.0 * k1 / (k1 + k2),
        }
    }

    /// Creates a scattering matrix for the interface between two layers.
    /// # Arguments
    /// * `n1` - The refractive index of the first layer.
    /// * `n2` - The refractive index of the second layer.
    /// * `k0` - The vacuum wavevector.
    /// * `k` - The parallel wavevector in the first layer.
    /// * `polarization` - The polarization of the light.
    /// # Returns
    /// The scattering matrix for the interface between the two layers.
    pub fn matrix_interface(
        n1: f64,
        n2: f64,
        k0: f64,
        k: f64,
        polarization: Polarization,
    ) -> ScatteringMatrix {
        match polarization {
            Polarization::TE => ScatteringMatrix::matrix_interface_te(n1, n2, k0, k),
            Polarization::TM => ScatteringMatrix::matrix_interface_tm(n1, n2, k0, k),
        }
    }
}

/// Calculates the scattering matrix for a multilayer system.
/// # Arguments
/// * `layers` - The layers of the system.
/// * `k0` - The vacuum wavevector.
/// * `k` - The parallel wavevector in the first layer.
/// * `polarization` - The polarization of the light.
/// # Returns
/// The scattering matrix for the multilayer system.
pub fn calculate_s_matrix(
    layers: &[Layer],
    k0: f64,
    k: f64,
    polarization: Polarization,
) -> ScatteringMatrix {
    let mut result =
        ScatteringMatrix::matrix_interface(layers[0].n, layers[1].n, k0, k, polarization);
    // println!("Interface_matrix: {:?}", result);
    for (layer1, layer2) in izip!(layers.iter().skip(1), layers.iter().skip(2)) {
        let matrix = ScatteringMatrix::matrix_propagation(layer1.n, layer1.d, k0, k);
        result = result.compose(matrix);
        let matrix = ScatteringMatrix::matrix_interface(layer1.n, layer2.n, k0, k, polarization);
        result = result.compose(matrix);
    }
    result
}
