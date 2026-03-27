//! Implementation of the transfer matrix method for multilayer structures.
//!
//! # Transfer matrix convention
//!
//! All matrices in this module use the **physical forward-propagation** convention.
//! The full system transfer matrix for layers `[L0, L1, …, L_{n-1}]` is defined as
//!
//! ```text
//! T = T_int(L_{n-2}, L_{n-1}) · T_prop(L_{n-2}) · … · T_prop(L1) · T_int(L0, L1)
//! ```
//!
//! Applied to the column vector `[a_in, b_in]` at the left boundary, it produces
//! `[a_out, b_out]` at the right boundary:
//!
//! ```text
//! [a_out, b_out]^T = T · [a_in, b_in]^T
//! ```
//!
//! The standard semi-infinite mode condition (`b_out = 0` for input `[0, 1]`) therefore
//! corresponds to `T[1,1] = 0`, i.e. `t22 = 0`.
//!
//! The propagation-coefficient helpers walk left-to-right in the same physical order,
//! so the accumulated matrix and the step-by-step propagation are always consistent.
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
    /// Returns a new TransferMatrix which is the composition of two matrices.
    ///
    /// `self.compose(other)` computes `self · other` (matrix multiplication).
    /// Applied to a vector, `other` acts first, then `self`.
    ///
    /// # Arguments
    /// * `other` - The matrix to right-multiply by.
    ///
    /// # Returns
    /// The product `self · other`.
    pub fn compose(self, other: TransferMatrix) -> TransferMatrix {
        TransferMatrix {
            t11: self.t11 * other.t11 + self.t12 * other.t21,
            t12: self.t11 * other.t12 + self.t12 * other.t22,
            t21: self.t21 * other.t11 + self.t22 * other.t21,
            t22: self.t21 * other.t12 + self.t22 * other.t22,
        }
    }

    /// Creates the identity transfer matrix. Useful to start a recursion.
    ///
    /// # Returns
    /// The identity transfer matrix.
    pub fn matrix_start() -> TransferMatrix {
        TransferMatrix {
            t11: Complex::new(1.0, 0.0),
            t12: Complex::new(0.0, 0.0),
            t21: Complex::new(0.0, 0.0),
            t22: Complex::new(1.0, 0.0),
        }
    }

    /// Creates the propagation transfer matrix for a homogeneous layer.
    ///
    /// The matrix propagates the forward/backward amplitudes `(a, b)` from the
    /// left edge to the right edge of the layer:
    ///
    /// ```text
    /// a_right = exp(+i·β·d) · a_left
    /// b_right = exp(-i·β·d) · b_left
    /// ```
    ///
    /// where `β = sqrt((k0·n)² − k²)` (possibly imaginary for evanescent layers).
    ///
    /// # Arguments
    /// * `n` - The refractive index of the layer.
    /// * `d` - The thickness of the layer.
    /// * `k0` - The vacuum wavevector.
    /// * `k` - The in-plane component of the wavevector.
    ///
    /// # Returns
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

    /// Creates the TE interface transfer matrix between two layers.
    ///
    /// Relates the forward/backward amplitudes `(a, b)` on the left side of the
    /// interface to those on the right side, enforcing continuity of the tangential
    /// electric field (`Ey`) and its derivative.
    ///
    /// # Arguments
    /// * `n1` - The refractive index of the layer on the left.
    /// * `n2` - The refractive index of the layer on the right.
    /// * `k0` - The vacuum wavevector.
    /// * `k` - The in-plane component of the wavevector.
    ///
    /// # Returns
    /// The TE interface transfer matrix.
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

    /// Creates the TM interface transfer matrix between two layers.
    ///
    /// Relates the forward/backward amplitudes `(a, b)` on the left side of the
    /// interface to those on the right side, enforcing continuity of the tangential
    /// magnetic field (`Hy`) and the normal displacement field.
    ///
    /// # Arguments
    /// * `n1` - The refractive index of the layer on the left.
    /// * `n2` - The refractive index of the layer on the right.
    /// * `k0` - The vacuum wavevector.
    /// * `k` - The in-plane component of the wavevector.
    ///
    /// # Returns
    /// The TM interface transfer matrix.
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

    /// Creates the interface transfer matrix between two layers for the given polarization.
    ///
    /// Dispatches to [`matrix_interface_te`] or [`matrix_interface_tm`].
    ///
    /// # Arguments
    /// * `n1` - The refractive index of the layer on the left.
    /// * `n2` - The refractive index of the layer on the right.
    /// * `k0` - The vacuum wavevector.
    /// * `k` - The in-plane component of the wavevector.
    /// * `polarization` - The polarization of the light.
    ///
    /// # Returns
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

    /// Applies the transfer matrix to a modal coefficient vector.
    ///
    /// Returns the output `(a_out, b_out)` produced by left-multiplying the
    /// column vector `[a, b]^T` by this matrix.
    ///
    /// # Arguments
    /// * `coefficient_vector` - The input modal coefficient vector.
    ///
    /// # Returns
    /// The output modal coefficient vector.
    pub fn multiply(&self, coefficient_vector: &LayerCoefficientVector) -> LayerCoefficientVector {
        LayerCoefficientVector {
            a: self.t11 * coefficient_vector.a + self.t12 * coefficient_vector.b,
            b: self.t21 * coefficient_vector.a + self.t22 * coefficient_vector.b,
        }
    }

    /// Applies the transfer matrix to a raw `(a, b)` pair.
    ///
    /// Returns `(a_out, b_out)`. This is a convenience overload used when
    /// evaluating mode conditions directly from scalar amplitudes.
    ///
    /// # Arguments
    /// * `a` - The forward amplitude.
    /// * `b` - The backward amplitude.
    ///
    /// # Returns
    /// The output `(a_out, b_out)`.
    pub fn apply(&self, a: Complex<f64>, b: Complex<f64>) -> (Complex<f64>, Complex<f64>) {
        (self.t11 * a + self.t12 * b, self.t21 * a + self.t22 * b)
    }
}

/// Calculates the physical forward-propagation transfer matrix of a multilayer system.
///
/// For a system with layers `[L0, L1, …, L_{n-1}]` the matrix is assembled as:
///
/// ```text
/// T = T_int(L_{n-2}, L_{n-1}) · T_prop(L_{n-2}) · … · T_prop(L1) · T_int(L0, L1)
/// ```
///
/// Applied to `[a_in, b_in]` at the **left** boundary it produces `[a_out, b_out]`
/// at the **right** boundary.
///
/// For the standard semi-infinite boundary mode condition the relevant entry is
/// `T[1,1]` (`t22`): starting from `(a_in, b_in) = (0, 1)` (only the decaying wave
/// in the left cladding) and requiring `b_out = 0` in the right cladding gives
/// `T[1,1] · 1 = 0`, i.e. `t22 = 0`.
///
/// # Arguments
/// * `layers` - The layers of the system.
/// * `k0` - The vacuum wavevector.
/// * `k` - The in-plane component of the wavevector.
/// * `polarization` - The polarization of the light.
///
/// # Returns
/// The physical forward-propagation transfer matrix.
pub fn calculate_t_matrix(
    layers: &[Layer],
    k0: f64,
    k: f64,
    polarization: Polarization,
) -> TransferMatrix {
    // Start at the rightmost interface and compose toward the left.
    // compose(A, B) = A · B, so right-multiplying accumulates earlier (leftward) factors.
    let n = layers.len();
    let mut result =
        TransferMatrix::matrix_interface(layers[n - 2].n, layers[n - 1].n, k0, k, polarization);
    for i in (0..n - 2).rev() {
        let prop = TransferMatrix::matrix_propagation(layers[i + 1].n, layers[i + 1].d, k0, k);
        result = result.compose(prop);
        let iface =
            TransferMatrix::matrix_interface(layers[i].n, layers[i + 1].n, k0, k, polarization);
        result = result.compose(iface);
    }
    result
}

/// Calculates the modal coefficients in each layer for a semi-infinite left boundary.
///
/// In the standard case `layers[0]` is the semi-infinite left cladding.  Its field
/// decays exponentially away from the guide, so no propagation step is needed
/// before crossing the first interface.  The supplied `(a, b)` are the amplitudes
/// at the reference plane `x = 0` (the right edge of the left cladding / left edge
/// of `layers[1]`).
///
/// Coefficients are stored in the order `[layer 0, layer 1, …]`, each referenced
/// to the **left edge** of the corresponding layer.
///
/// # Arguments
/// * `layers` - The layers of the system (first element is the semi-infinite left cladding).
/// * `k0` - The vacuum wavevector.
/// * `k` - The in-plane component of the wavevector.
/// * `polarization` - The polarization of the light.
/// * `a` - The forward amplitude in the left cladding (typically 0 for a guided mode).
/// * `b` - The backward amplitude in the left cladding (typically 1, normalised later).
///
/// # Returns
/// The modal coefficient vector for each layer, referenced to the layer's left edge.
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
    // Layer 0: semi-infinite left cladding, coefficients at x = 0 (right edge / reference).
    coefficients.push(current_coefficients);
    // Cross the L0→L1 interface (no propagation inside L0 since it is semi-infinite).
    let transfer_matrix =
        TransferMatrix::matrix_interface(layers[0].n, layers[1].n, k0, k, polarization);
    current_coefficients = transfer_matrix.multiply(&current_coefficients);
    coefficients.push(current_coefficients);
    // For every subsequent layer: propagate to the right edge, then cross the interface.
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

/// Calculates the modal coefficients in each layer for a PEC left boundary.
///
/// Here `layers[0]` is the first *finite* dielectric layer, whose left edge sits
/// at the PEC wall (`x = 0`).  The supplied `(a, b)` are the field amplitudes
/// **at the PEC wall** and must already satisfy the PEC boundary condition
/// (e.g. `a = 1, b = −1` for TE so that `Ey(0) = a + b = 0`).
///
/// Unlike the semi-infinite case, `layers[0]` has finite thickness, so we must
/// propagate through it before crossing the first interface.
///
/// Coefficients are stored in the order `[layer 0, layer 1, …]`, each referenced
/// to the **left edge** of the corresponding layer.
///
/// # Arguments
/// * `layers` - The layers of the system; `layers[0]` is adjacent to the PEC wall.
/// * `k0` - The vacuum wavevector.
/// * `k` - The in-plane component of the wavevector.
/// * `polarization` - The polarization of the light.
/// * `a` - The forward amplitude at the PEC wall (left edge of `layers[0]`).
/// * `b` - The backward amplitude at the PEC wall (left edge of `layers[0]`).
///
/// # Returns
/// The modal coefficient vector for each layer, referenced to the layer's left edge.
pub fn get_propagation_coefficients_pec_left(
    layers: &[Layer],
    k0: f64,
    k: f64,
    polarization: Polarization,
    a: Complex<f64>,
    b: Complex<f64>,
) -> Vec<LayerCoefficientVector> {
    let mut coefficients: Vec<LayerCoefficientVector> = Vec::new();
    let mut current_coefficients = LayerCoefficientVector::new(a, b);
    // Layer 0: coefficients at x = 0 (PEC wall / left edge of layer 0).
    coefficients.push(current_coefficients);
    // Propagate through layer 0 to its right edge, then cross into layer 1.
    let propagation_matrix = TransferMatrix::matrix_propagation(layers[0].n, layers[0].d, k0, k);
    current_coefficients = propagation_matrix.multiply(&current_coefficients);
    let interface_matrix =
        TransferMatrix::matrix_interface(layers[0].n, layers[1].n, k0, k, polarization);
    current_coefficients = interface_matrix.multiply(&current_coefficients);
    coefficients.push(current_coefficients);
    // For every subsequent layer: propagate to the right edge, then cross the interface.
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
