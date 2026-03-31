//! This module contains the implementation of the `MultiLayer` struct and its methods.
extern crate cumsum;
extern crate find_peaks;
extern crate itertools;

use num_complex::Complex64;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::cmp::Ordering;
use std::iter::zip;
use std::iter::Sum;
use std::ops::{Add, Mul, Sub};

use crate::enums::BackEnd;
use crate::enums::BoundaryCondition;
use crate::enums::Polarization;
use crate::layer::{Layer, LayerCoefficientVector, PEC};
use crate::scattering_matrix::calculate_s_matrix;
use crate::transfer_matrix::TransferMatrix;
use crate::transfer_matrix::{
    calculate_t_matrix, get_propagation_coefficients_pec_left,
    get_propagation_coefficients_transfer,
};
use cumsum::cumsum;
use find_peaks::PeakFinder;
use num_complex::Complex;

/// Vacuum impedance.
const Z0: Complex<f64> = Complex {
    re: 376.73031346177066,
    im: 0.0,
};

/// Integrates a function on sampled data using the trapezoidal rule.
/// # Arguments
/// * `y` - The y values of the function to integrate.
/// * `x` - The x values of the function to integrate.
/// # Returns
/// The integral of the function.
fn quadrature_integration<T, U>(y: &[T], x: &[U]) -> Result<T, &'static str>
where
    T: Add<Output = T> + Sub<Output = T> + Mul<U, Output = T> + Sum + Copy,
    U: Add<Output = U> + Sub<Output = U> + Copy,
{
    if y.len() != x.len() {
        return Err("The length of the input vectors must be the same");
    }
    let correction = match (y.last(), y.first()) {
        (Some(&last), Some(&first)) => last + first,
        _ => return Err("The input vectors must not be empty"),
    };
    let dx = x[1] - x[0];
    let sum: T = y.iter().copied().sum();
    let integral = sum - correction;
    Ok(integral * dx)
}

/// Struct representing the grid data used for plitting.
struct GridData {
    /// The x values of the grid.
    xplot: Vec<f64>,
    /// The x starting positions of the layers.
    xstarts: Vec<f64>,
    /// The indices of the x values where the layers start.
    ixstarts: Vec<usize>,
}

/// Struct representing the field data of a mode.
/// This is also available in the Python API.
#[pyclass]
#[allow(non_snake_case)]
pub struct FieldData {
    /// x coordinates of the field data.
    #[pyo3(get)]
    pub x: Vec<f64>,
    /// Electric field in the x direction.
    #[pyo3(get)]
    pub Ex: Vec<Complex<f64>>,
    /// Electric field in the y direction.
    #[pyo3(get)]
    pub Ey: Vec<Complex<f64>>,
    /// Electric field in the z direction.
    #[pyo3(get)]
    pub Ez: Vec<Complex<f64>>,
    /// Magnetic field in the x direction.
    #[pyo3(get)]
    pub Hx: Vec<Complex<f64>>,
    /// Magnetic field in the y direction.
    #[pyo3(get)]
    pub Hy: Vec<Complex<f64>>,
    /// Magnetic field in the z direction.
    #[pyo3(get)]
    pub Hz: Vec<Complex<f64>>,
}

impl FieldData {
    /// Creates a zeroed FieldData (all field components set to zero) for a given x grid.
    /// Used as a safe fallback when a requested mode does not exist.
    pub fn zeros(x: Vec<f64>) -> FieldData {
        let n = x.len();
        FieldData {
            x,
            Ex: vec![Complex::new(0.0, 0.0); n],
            Ey: vec![Complex::new(0.0, 0.0); n],
            Ez: vec![Complex::new(0.0, 0.0); n],
            Hx: vec![Complex::new(0.0, 0.0); n],
            Hy: vec![Complex::new(0.0, 0.0); n],
            Hz: vec![Complex::new(0.0, 0.0); n],
        }
    }

    /// Returns the z component of the Poynting vector of the field.
    pub fn get_poyinting_vector(&self) -> Complex<f64> {
        let poynting: Vec<Complex<f64>> = self
            .Ex
            .iter()
            .zip(self.Hy.iter())
            .zip(self.Ey.iter().zip(self.Hx.iter()))
            .map(|((&ex, &hy), (&ey, &hx))| ex * hy.conj() - ey * hx.conj())
            .collect();
        quadrature_integration(&poynting, &self.x).unwrap()
    }

    /// Normalizes the field data so that the absolute value of z component of the Poynting vector is 1.
    pub fn normalize(self) -> FieldData {
        let poynting_vector = self.get_poyinting_vector();
        let norm = poynting_vector.sqrt();
        let ex = self.Ex.into_iter().map(|x| x / norm).collect();
        let ey = self.Ey.into_iter().map(|x| x / norm).collect();
        let ez = self.Ez.into_iter().map(|x| x / norm).collect();
        let hx = self.Hx.into_iter().map(|x| x / norm).collect();
        let hy = self.Hy.into_iter().map(|x| x / norm).collect();
        let hz = self.Hz.into_iter().map(|x| x / norm).collect();
        FieldData {
            x: self.x,
            Ex: ex,
            Ey: ey,
            Ez: ez,
            Hx: hx,
            Hy: hy,
            Hz: hz,
        }
    }
}

/// Struct representing the index data of the multi-layer.
/// This is also available in the Python API.
#[pyclass]
pub struct IndexData {
    /// The x values of the index data.
    #[pyo3(get)]
    pub x: Vec<f64>,
    /// The index of refraction of the multi-layer.
    #[pyo3(get)]
    pub n: Vec<f64>,
}

/// Calculates the field profile in a layer given the modal coefficients.
/// # Arguments
/// * `a` - The modal coefficient of the forward propagating wave.
/// * `b` - The modal coefficient of the backward propagating wave.
/// * `k0` - The vacuum wavevector.
/// * `k` - The parallel wavevector.
/// * `n` - The index of refraction.
/// * `x` - The x coordinates inside the layer.
/// # Returns
/// The field profile in the layer.
fn get_field_slice(
    a: Complex<f64>,
    b: Complex<f64>,
    k0: f64,
    k: f64,
    n: f64,
    x: Vec<f64>,
) -> Vec<Complex<f64>> {
    let k0 = num_complex::Complex::new(k0, 0.0);
    let k = num_complex::Complex::new(k, 0.0);
    let n = num_complex::Complex::new(n, 0.0);
    let beta = ((k0 * n).powi(2) - k.powi(2)).sqrt();
    x.iter()
        .map(|x| {
            let z = num_complex::Complex::new(0.0, *x);
            let phase_p = z * beta;
            let phase_n = -z * beta;
            a * phase_p.exp() + b * phase_n.exp()
        })
        .collect()
}

type FullLayerCoefficientVector = (
    Vec<LayerCoefficientVector>,
    Vec<LayerCoefficientVector>,
    Vec<LayerCoefficientVector>,
    Vec<LayerCoefficientVector>,
    Vec<LayerCoefficientVector>,
    Vec<LayerCoefficientVector>,
);

/// Structs repressenting the multilayer structure.
/// Implements methods for calculating the modes and fields of the structure.
/// This is also available in the Python API.
#[pyclass]
pub struct MultiLayer {
    /// The layers of the multi-layer.
    layers: Vec<Layer>,
    /// The backend used for the calculations.
    backend: BackEnd,
    /// Number of significant digits requested for neff calculation.
    required_accuracy: i32,
    /// The step size for plotting the field.
    #[pyo3(get, set)]
    pub plot_step: f64,
    /// Boundary condition on the left side of the structure.
    left_bc: BoundaryCondition,
    /// Boundary condition on the right side of the structure.
    right_bc: BoundaryCondition,
}

/// Methods of the MultiLayer struct also available in the Python API.
#[pymethods]
impl MultiLayer {
    #[new]
    /// Creates a new MultiLayer from a list of layers and optional PEC boundary markers.
    /// A PEC element placed first sets a PEC left boundary; placed last sets a PEC right boundary.
    /// PEC cannot appear in the middle or on both ends simultaneously.
    pub fn from_python_layers(layers: Vec<Bound<'_, PyAny>>) -> PyResult<MultiLayer> {
        let mut left_bc = BoundaryCondition::SemiInfinite;
        let mut right_bc = BoundaryCondition::SemiInfinite;
        let mut dielectric_layers: Vec<Layer> = Vec::new();
        let len = layers.len();

        if len < 2 {
            return Err(PyValueError::new_err(
                "MultiLayer requires at least 2 elements",
            ));
        }

        for (i, item) in layers.iter().enumerate() {
            if item.is_instance_of::<PEC>() {
                if i == 0 {
                    left_bc = BoundaryCondition::PEC;
                } else if i == len - 1 {
                    right_bc = BoundaryCondition::PEC;
                } else {
                    return Err(PyValueError::new_err(
                        "PEC can only be the first or last element of the layer list",
                    ));
                }
            } else {
                let layer = item.extract::<Layer>().map_err(|_| {
                    PyValueError::new_err("All non-PEC elements must be Layer instances")
                })?;
                dielectric_layers.push(layer);
            }
        }

        if matches!(left_bc, BoundaryCondition::PEC) && matches!(right_bc, BoundaryCondition::PEC) {
            return Err(PyValueError::new_err(
                "PEC on both sides simultaneously is not supported",
            ));
        }

        if dielectric_layers.len() < 2 {
            return Err(PyValueError::new_err(
                "MultiLayer requires at least 2 dielectric (non-PEC) layers",
            ));
        }

        let mut multilayer = MultiLayer {
            layers: dielectric_layers,
            backend: BackEnd::Transfer,
            required_accuracy: 10,
            plot_step: 1e-3,
            left_bc,
            right_bc,
        };
        multilayer.set_backend(BackEnd::Transfer);
        Ok(multilayer)
    }

    /// Sets the left boundary condition of the structure.
    #[pyo3(name = "set_left_boundary")]
    pub fn python_set_left_boundary(&mut self, bc: BoundaryCondition) {
        self.set_left_boundary(bc);
    }

    /// Sets the right boundary condition of the structure.
    #[pyo3(name = "set_right_boundary")]
    pub fn python_set_right_boundary(&mut self, bc: BoundaryCondition) {
        self.set_right_boundary(bc);
    }

    /// Calculates neff of the requested mode.
    /// # Arguments
    /// * `omega` - The angular frequency of the mode.
    /// * `polarization` - The polarization of the mode.
    /// * `mode` - The mode number.
    /// # Returns
    /// The effective index of refraction of the mode, or None if the mode does not exist.
    #[pyo3(name = "neff")]
    #[pyo3(signature = (omega, polarization=None, mode=None))]
    pub fn python_neff(
        &self,
        omega: f64,
        polarization: Option<Polarization>,
        mode: Option<usize>,
    ) -> Option<f64> {
        let polarization = polarization.unwrap_or(Polarization::TE);
        let mode = mode.unwrap_or(0);
        self.neff(omega, polarization, mode).ok()
    }

    /// Returns all effective indices supported by the structure.
    /// # Arguments
    /// * `omega` - The angular frequency.
    /// * `polarization` - The polarization of the modes.
    /// # Returns
    /// A list of effective indices, sorted from highest to lowest.
    #[pyo3(name = "all_neff")]
    #[pyo3(signature = (omega, polarization=None))]
    pub fn python_all_neff(&self, omega: f64, polarization: Option<Polarization>) -> Vec<f64> {
        let polarization = polarization.unwrap_or(Polarization::TE);
        self.solve(omega, polarization)
    }

    /// Returs the index profile of the multi-layer.
    #[pyo3(name = "index")]
    pub fn python_index(&self) -> IndexData {
        self.index()
    }

    /// Calculates the field profile of the requested mode.
    /// # Arguments
    /// * `omega` - The angular frequency of the mode.
    /// * `polarization` - The polarization of the mode.
    /// * `mode` - The mode number.
    /// # Returns
    /// The field profile of the mode, or a zeroed FieldData if the mode does not exist.
    #[pyo3(name = "field")]
    #[pyo3(signature = (omega, polarization=None, mode=None))]
    pub fn python_field(
        &self,
        omega: f64,
        polarization: Option<Polarization>,
        mode: Option<usize>,
    ) -> FieldData {
        let polarization = polarization.unwrap_or(Polarization::TE);
        let mode = mode.unwrap_or(0);
        self.field(omega, polarization, mode)
            .unwrap_or_else(|_| FieldData::zeros(self.get_grid_data().xplot))
    }
}

impl MultiLayer {
    /// Creates a new MultiLayer from a Vec<Layer> with default SemiInfinite boundary conditions.
    /// Used internally by Rust code and the CLI binary.
    pub fn new(layers: Vec<Layer>) -> MultiLayer {
        let mut multilayer = MultiLayer {
            layers,
            backend: BackEnd::Transfer,
            required_accuracy: 10,
            plot_step: 1e-3,
            left_bc: BoundaryCondition::SemiInfinite,
            right_bc: BoundaryCondition::SemiInfinite,
        };
        multilayer.set_backend(BackEnd::Transfer);
        multilayer
    }

    /// Sets the left boundary condition.
    pub fn set_left_boundary(&mut self, bc: BoundaryCondition) {
        self.left_bc = bc;
    }

    /// Sets the right boundary condition.
    pub fn set_right_boundary(&mut self, bc: BoundaryCondition) {
        self.right_bc = bc;
    }

    /// Switches the backend used for the calculations.
    pub fn set_backend(&mut self, backend: BackEnd) {
        self.backend = backend;
    }

    /// Get the threshold for the findpeak function for a ginen number of significant digits.
    fn get_threshold(accuracy: i32) -> f64 {
        match accuracy {
            0..=2 => -2.0,
            3..=5 => 0.0,
            6..=8 => 3.0,
            9..=11 => 6.0,
            _ => 9.0,
        }
    }

    /// Function to maximise to find the mode, accounting for boundary conditions.
    fn characteristic_function(&self, k0: f64, k: f64, polarization: Polarization) -> Complex<f64> {
        match self.backend {
            BackEnd::Scattering => {
                calculate_s_matrix(&self.layers, k0, k, polarization).determinant()
            }
            BackEnd::Transfer => {
                let t = calculate_t_matrix(&self.layers, k0, k, polarization);
                match (self.left_bc, self.right_bc) {
                    (BoundaryCondition::SemiInfinite, BoundaryCondition::SemiInfinite) => {
                        // Starting from (0, 1) in the left cladding, b_out = T[1,1] · 1.
                        // Mode condition: b_out = 0  →  t22 = 0.
                        1.0 / t.t22
                    }
                    (BoundaryCondition::PEC, BoundaryCondition::SemiInfinite) => {
                        // PEC wall at the left edge of layers[0].
                        // Initial condition: (a, b) = (1, −1) so that the tangential E is
                        // zero at x = 0 (TE: Ey = a + b = 0; TM: Ez = a + b = 0).
                        // Propagate forward through layers[0], then through the rest via T.
                        // Full matrix: T · T_prop(L0).
                        // Mode condition: b_out = 0  (no growing wave in right cladding).
                        let prop = TransferMatrix::matrix_propagation(
                            self.layers[0].n,
                            self.layers[0].d,
                            k0,
                            k,
                        );
                        let t_full = t.compose(prop);
                        let (_, b_out) =
                            t_full.apply(Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0));
                        1.0 / b_out
                    }
                    (BoundaryCondition::SemiInfinite, BoundaryCondition::PEC) => {
                        // PEC wall at the right edge of layers[last].
                        // Initial condition: (0, 1) in the left semi-infinite cladding.
                        // Propagate forward to the PEC wall via T_prop(L_last) · T.
                        // Mode condition: a_out + b_out = 0  (tangential E = 0 at PEC wall).
                        let last = self.layers.last().unwrap();
                        let prop = TransferMatrix::matrix_propagation(last.n, last.d, k0, k);
                        let t_full = prop.compose(t);
                        let (a_out, b_out) =
                            t_full.apply(Complex::new(0.0, 0.0), Complex::new(1.0, 0.0));
                        1.0 / (a_out + b_out)
                    }
                    (BoundaryCondition::PEC, BoundaryCondition::PEC) => {
                        // Prevented at construction time; this branch should never be reached.
                        panic!("Both-PEC boundary condition is not supported")
                    }
                }
            }
        }
    }

    /// Finds the minimum and maximum index of the multi-layer.
    /// # Returns
    /// The minimum and maximum index of the multi-layer.
    fn find_minmax_n(&self) -> (f64, f64) {
        find_minmax_n(&self.layers)
    }

    /// Single step of the maximum finding process.
    /// Given a certain k value, returns the k values corresponding to the peaks in the characteristic function./
    /// # Arguments
    /// * `k0` - The vacuum wavevector.
    /// * `k_min` - The minimum parallel wavevector.
    /// * `k_max` - The maximum parallel wavevector.
    /// * `step` - The step size.
    /// * `treshold` - The treshold for the peak finding.
    /// * `polarization` - The polarization of the mode.
    /// # Returns
    /// The k values corresponding to the peaks in the characteristic function.
    fn solve_step(
        &self,
        k0: f64,
        k_min: f64,
        k_max: f64,
        step: f64,
        treshold: f64,
        polarization: Polarization,
    ) -> Vec<f64> {
        let kv: Vec<f64> = iter_num_tools::arange(k_min..k_max, step).collect();
        let det: Vec<f64> = kv
            .iter()
            .map(|&k| {
                self.characteristic_function(k0, k, polarization)
                    .norm()
                    .log10()
            })
            .collect();

        let mut peak_finder = PeakFinder::new(&det);
        let peaks = peak_finder.with_min_prominence(0.8).find_peaks();

        peaks.into_iter().map(|p| kv[p.middle_position()]).collect()
    }

    /// Finds the modes of the multi-layer.
    /// # Arguments
    /// * `k0` - The vacuum wavevector.
    /// * `polarization` - The polarization of the mode.
    /// # Returns
    /// The effective indices of the modes.
    pub fn solve(&self, k0: f64, polarization: Polarization) -> Vec<f64> {
        let (min_n, max_n) = self.find_minmax_n();
        let k_min = k0 * min_n + 1e-9;
        let k_max = k0 * max_n - 1e-9;

        let mut solution_backets = vec![(k_min, k_max)];

        let mut ksolutions = Vec::new();

        for accuracy in 2..self.required_accuracy {
            let step = 10.0_f64.powi(-accuracy);
            let threshold = Self::get_threshold(accuracy);
            ksolutions.clear();
            for (_kmin, _kmax) in solution_backets {
                let _solutions =
                    self.solve_step(k0, _kmin, _kmax, 0.1 * step, threshold, polarization);
                ksolutions.extend(_solutions);
            }
            solution_backets = ksolutions
                .iter()
                .map(|k| (k - step, k + step))
                .collect::<Vec<_>>();
        }

        let mut n_solutions = ksolutions.into_iter().map(|k| k / k0).collect::<Vec<_>>();

        n_solutions.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        n_solutions
    }

    /// Finds the effective index of a mode.
    /// # Arguments
    /// * `k0` - The vacuum wavevector.
    /// * `polarization` - The polarization of the mode.
    /// * `mode` - The mode to find.
    /// # Returns
    /// The effective index of the mode.
    /// # Errors
    /// If the mode is not found.
    pub fn neff(&self, k0: f64, polarization: Polarization, mode: usize) -> Result<f64, String> {
        let n_solutions = self.solve(k0, polarization);
        match n_solutions.get(mode) {
            Some(&n) => Ok(n),
            None => Err(format!(
                "Mode {} not found. Only {} modes (0->{}) available.",
                mode,
                n_solutions.len(),
                n_solutions.len() - 1
            )),
        }
    }

    /// Calculates the modal coefficients of each layer given the starting coefficients.
    /// # Arguments
    /// * `k0` - The vacuum wavevector.
    /// * `k` - The parallel wavevector.
    /// * `polarization` - The polarization of the mode.
    /// * `a` - In case of transfer matrix method, the forward coefficient of the first layer. Scattering matrix method not implemented yet..
    /// * `b` - In case of transfer matrix method, the backward coefficient of the first layer. Scattering matrix method not implemented yet..
    /// # Returns
    /// The modal coefficients of each layer.
    pub fn get_propagation_coefficients(
        &self,
        k0: f64,
        k: f64,
        polarization: Polarization,
        a: Complex<f64>,
        b: Complex<f64>,
    ) -> Vec<LayerCoefficientVector> {
        match self.backend {
            BackEnd::Transfer => match self.left_bc {
                BoundaryCondition::PEC => {
                    // For PEC-left, layers[0] is the first finite layer adjacent to the PEC
                    // wall. We must propagate through it before crossing the first interface,
                    // which the standard function skips (it assumes layers[0] is a semi-infinite
                    // cladding where no propagation is needed).
                    get_propagation_coefficients_pec_left(&self.layers, k0, k, polarization, a, b)
                }
                BoundaryCondition::SemiInfinite => {
                    get_propagation_coefficients_transfer(&self.layers, k0, k, polarization, a, b)
                }
            },
            BackEnd::Scattering => {
                panic!("Not implemented yet")
            }
        }
    }

    /// Calculates the plotting grid data for the multilayer.
    fn get_grid_data(&self) -> GridData {
        let xstart = match self.left_bc {
            BoundaryCondition::SemiInfinite => -self.layers[0].d,
            // PEC wall is at x=0; no region to the left of it
            BoundaryCondition::PEC => 0.0,
        };
        let xend = self.layers.iter().map(|l| l.d).sum::<f64>() + xstart;
        let xgrid: Vec<f64> = iter_num_tools::arange(xstart..xend, self.plot_step).collect();
        let grid_starts: Vec<f64> = self.layers.iter().map(|l| l.d).collect();
        let grid_starts: Vec<f64> = [vec![0.0_f64], grid_starts].concat();
        let mut grid_starts: Vec<f64> = cumsum(&grid_starts).iter().map(|x| x + xstart).collect();
        let mut grid_istarts: Vec<usize> = vec![0];
        let mut slice_iter = grid_starts.iter();
        let _ = slice_iter.next();
        let mut start = slice_iter.next().unwrap();
        for (i, x) in xgrid.iter().enumerate() {
            if x >= start {
                grid_istarts.push(i);
                start = slice_iter.next().unwrap();
            }
        }
        grid_istarts.push(xgrid.len());

        grid_starts[0] = 0.0;

        GridData {
            xplot: xgrid,
            xstarts: grid_starts,
            ixstarts: grid_istarts,
        }
    }

    /// Calculates the profile of a single field component given the modal coefficients of all the layers.
    /// # Arguments
    /// * `coefficient_vector` - The modal coefficients of each layer.
    /// * `grid_data` - The grid data for the multilayer.
    /// * `k0` - The vacuum wavevector.
    /// * `k` - The parallel wavevector.
    /// # Returns
    /// The field component profile.
    fn get_field_componet(
        &self,
        coefficient_vector: &[LayerCoefficientVector],
        grid_data: &GridData,
        k0: f64,
        k: f64,
    ) -> Vec<Complex<f64>> {
        let x = grid_data.xplot.clone();

        let mut field_vectors: Vec<Complex64> = Vec::new();
        for (i, (&istart, &iend)) in zip(
            grid_data.ixstarts.clone().iter(),
            grid_data.ixstarts.clone().iter().skip(1),
        )
        .enumerate()
        {
            let xstart = grid_data.xstarts[i];
            let coefficients = coefficient_vector[i];
            let layer = &self.layers[i];
            let xslice: Vec<f64> = x[istart..iend]
                .iter()
                .map(|x| x - xstart)
                .collect::<Vec<_>>();
            field_vectors.extend(get_field_slice(
                coefficients.a,
                coefficients.b,
                k0,
                k,
                layer.n,
                xslice,
            ));
        }
        field_vectors
    }

    /// Calculates the modla coefficients for all field components.
    /// # Arguments
    /// * `k0` - The vacuum wavevector.
    /// * `k` - The parallel wavevector.
    /// * `main_coefficients` - The modal coefficients of the main field component.
    /// # Returns
    /// The modal coefficients of all field components.
    pub fn get_coefficient_all_components(
        &self,
        k0: f64,
        k: f64,
        main_coefficients: Vec<LayerCoefficientVector>,
    ) -> FullLayerCoefficientVector {
        let main1 = main_coefficients;
        let k0 = Complex::new(k0, 0.0);
        let k = Complex::new(k, 0.0);
        let mut main2 = Vec::new();
        let mut main3 = Vec::new();
        let mut maink = Vec::new();
        let mut mainb = Vec::new();
        let zeros = vec![
            LayerCoefficientVector::new(Complex::new(0.0, 0.0), Complex::new(0.0, 0.0));
            self.layers.len()
        ];
        for (layer, coefficients) in zip(self.layers.iter(), main1.iter()) {
            let kpar = ((layer.n * k0).powi(2) - k.powi(2)).sqrt();
            let n = Complex::new(layer.n, 0.0);
            main2.push(LayerCoefficientVector::new(
                -coefficients.a * k / kpar,
                coefficients.b * k / kpar,
            ));
            main3.push(LayerCoefficientVector::new(
                -coefficients.a * k0 * n.powi(2) / kpar,
                coefficients.b * k0 * n.powi(2) / kpar,
            ));
            maink.push(LayerCoefficientVector::new(
                coefficients.a * kpar / k0,
                -coefficients.b * kpar / k0,
            ));
            mainb.push(LayerCoefficientVector::new(
                -coefficients.a * k / k0,
                -coefficients.b * k / k0,
            ));
        }
        (main1, main2, main3, maink, mainb, zeros)
    }

    /// Calculates the field profile of the requested mode.
    /// # Arguments
    /// * `k0` - The vacuum wavevector.
    /// * `k` - The parallel wavevector.
    /// * `mode` - The mode number.
    /// * `polarization` - The polarization of the mode.
    /// # Returns
    /// The field profile of the requested mode.
    /// # Errors
    /// Returns an error if the requested mode is not found.
    pub fn field(
        &self,
        k0: f64,
        polarization: Polarization,
        mode: usize,
    ) -> Result<FieldData, String> {
        let neff = match self.neff(k0, polarization, mode) {
            Ok(n) => n,
            Err(e) => return Err(e),
        };

        let (init_a, init_b) = match self.left_bc {
            BoundaryCondition::SemiInfinite => (Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)),
            BoundaryCondition::PEC => (Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0)),
        };
        let mut coefficient_vector =
            self.get_propagation_coefficients(k0, k0 * neff, polarization, init_a, init_b);
        let grid_data = self.get_grid_data();

        match self.right_bc {
            BoundaryCondition::SemiInfinite => {
                // Zero the growing wave in the semi-infinite right cladding
                let last_coefficient = coefficient_vector.pop().unwrap();
                coefficient_vector.push(LayerCoefficientVector {
                    a: last_coefficient.a,
                    b: Complex::new(0.0, 0.0),
                });
            }
            BoundaryCondition::PEC => {
                // Both forward and backward waves are physically present in the last
                // finite layer. The PEC condition (field = 0 at right edge) is
                // satisfied numerically since neff was found from the characteristic
                // function that enforces it.
            }
        }

        let coefficients = self.get_coefficient_all_components(k0, k0 * neff, coefficient_vector);

        let (main1, main2, main3, maink, mainb, zeros) = coefficients;
        let field1 = self.get_field_componet(&main1, &grid_data, k0, k0 * neff);
        let fieldzeros = self.get_field_componet(&zeros, &grid_data, k0, k0 * neff);

        let field_data = match polarization {
            Polarization::TE => {
                let fieldk = self.get_field_componet(&maink, &grid_data, k0, k0 * neff);
                let fieldb = self.get_field_componet(&mainb, &grid_data, k0, k0 * neff);

                FieldData {
                    x: grid_data.xplot.clone(),
                    Ex: fieldzeros.clone(),
                    Ey: field1,
                    Ez: fieldzeros.clone(),
                    Hx: fieldb.iter().map(|x| x / Z0).collect(),
                    Hy: fieldzeros.clone(),
                    Hz: fieldk.iter().map(|x| x / Z0).collect(),
                }
            }
            Polarization::TM => {
                let field2 = self.get_field_componet(&main2, &grid_data, k0, k0 * neff);
                let field3 = self.get_field_componet(&main3, &grid_data, k0, k0 * neff);
                FieldData {
                    x: grid_data.xplot.clone(),
                    Ex: field2,
                    Ey: fieldzeros.clone(),
                    Ez: field1,
                    Hx: fieldzeros.clone(),
                    Hy: field3.iter().map(|x| x / Z0).collect(),
                    Hz: fieldzeros.clone(),
                }
            }
        };

        Ok(field_data.normalize())
    }

    /// Calculates index profile of the multilayer from a grid data object.
    fn get_index(&self, grid_data: &GridData) -> Vec<f64> {
        let xgrid = grid_data.xplot.clone();
        let mut n = vec![self.layers[0].n; xgrid.len()];
        for (i, layer) in self.layers.iter().enumerate() {
            let start = grid_data.ixstarts[i];
            let end = grid_data.ixstarts[i + 1];
            n[start..end].iter_mut().for_each(|x| *x = layer.n);
        }
        n
    }

    /// Calcultes the refractive index profile of the multilayer.
    pub fn index(&self) -> IndexData {
        let grid_data = self.get_grid_data();
        let index = self.get_index(&grid_data);
        IndexData {
            x: grid_data.xplot.clone(),
            n: index,
        }
    }
}

/// Calculates the minimum and maximum refractive index of a list of layers.
fn find_minmax_n(layers: &[Layer]) -> (f64, f64) {
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
    use std::{f64::consts::PI, fmt::Debug};

    trait ApproxEqual {
        fn approx_eq(&self, other: &Self, epsilon: f64) -> bool;
    }

    impl ApproxEqual for f64 {
        fn approx_eq(&self, other: &Self, epsilon: f64) -> bool {
            (self - other).abs() < epsilon
        }
    }

    impl ApproxEqual for Complex<f64> {
        fn approx_eq(&self, other: &Self, epsilon: f64) -> bool {
            self.re.approx_eq(&other.re, epsilon) && self.im.approx_eq(&other.im, epsilon)
        }
    }

    impl ApproxEqual for LayerCoefficientVector {
        fn approx_eq(&self, other: &Self, epsilon: f64) -> bool {
            self.a.approx_eq(&other.a, epsilon) && self.b.approx_eq(&other.b, epsilon)
        }
    }

    fn assert_vec_approx_equal<T>(vec1: &[T], vec2: &[T], epsilon: f64)
    where
        T: ApproxEqual,
        T: Debug,
    {
        assert_eq!(vec1.len(), vec2.len(), "Vectors have different lengths");
        for (i, (a, b)) in vec1.iter().zip(vec2.iter()).enumerate() {
            assert!(
                a.approx_eq(b, epsilon),
                "Values at index {} are not approximately equal: {:?} != {:?}",
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

    fn create_asymmetric_coupled_slab_multilayer() -> MultiLayer {
        let layers: Vec<Layer> = vec![
            Layer::new(1.0, 1.0),
            Layer::new(1.51, 5.0),
            Layer::new(1.5, 2.0),
            Layer::new(2.0, 0.03),
            Layer::new(1.5, 1.0),
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

    #[test]
    fn test_transfer_asymmetric_coupled_slab_te() {
        let mut multi_layer = create_asymmetric_coupled_slab_multilayer();
        let om = 2.0 * PI / 1.55;
        multi_layer.set_backend(BackEnd::Transfer);

        let neff = multi_layer.solve(om, Polarization::TE);
        let expected_neff = vec![1.506483533, 1.502165605];
        assert_vec_approx_equal(&neff, &expected_neff, 1e-9);
    }

    #[test]
    fn test_transfer_asymmetric_coupled_slab_tm() {
        let mut multi_layer = create_asymmetric_coupled_slab_multilayer();
        let om = 2.0 * PI / 1.55;
        multi_layer.set_backend(BackEnd::Transfer);

        let neff = multi_layer.solve(om, Polarization::TM);
        let expected_neff = vec![1.505752112, 1.500196545];
        assert_vec_approx_equal(&neff, &expected_neff, 1e-9);
    }

    #[test]
    fn test_transfer_field_slab() {
        let mut multi_layer = create_slab_multilayer();
        let om = 2.0 * PI / 1.55;
        multi_layer.set_backend(BackEnd::Transfer);

        let n = multi_layer.neff(om, Polarization::TE, 0).unwrap_or(0.0);
        let ref_coefficients = vec![
            LayerCoefficientVector::new(Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)),
            LayerCoefficientVector::new(
                Complex::new(0.5, -0.870271563),
                Complex::new(0.5, 0.870271563),
            ),
            LayerCoefficientVector::new(Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)),
        ];
        let coefficients = multi_layer.get_propagation_coefficients(
            om,
            om * n,
            Polarization::TE,
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
        );
        assert_vec_approx_equal(&coefficients, &ref_coefficients, 1e-9);

        let n = multi_layer.neff(om, Polarization::TM, 0).unwrap_or(0.0);
        let ref_coefficients = vec![
            LayerCoefficientVector::new(Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)),
            LayerCoefficientVector::new(
                Complex::new(0.5, 0.105955587),
                Complex::new(0.5, -0.105955587),
            ),
            LayerCoefficientVector::new(Complex::new(-1.0, 0.0), Complex::new(0.0, 0.0)),
        ];
        let coefficients = multi_layer.get_propagation_coefficients(
            om,
            om * n,
            Polarization::TM,
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
        );
        assert_vec_approx_equal(&coefficients, &ref_coefficients, 2e-9);
    }

    #[test]
    fn test_field_normalization() {
        let slab = create_slab_multilayer();
        let field = slab.field(2.0 * PI / 1.55, Polarization::TE, 0).unwrap();
        let amplitude = field.Ey[1300];
        let reference = Complex::new(21.207074050, 0.0);
        assert!(
            amplitude.approx_eq(&reference, 1e-9),
            "{:?} != {:?}",
            amplitude,
            reference
        );
    }

    fn create_pec_left_half_slab() -> MultiLayer {
        // Half-slab: PEC at left wall of a 2.0-index layer, open on right
        // Equivalent to the ODD TE/TM modes of a symmetric slab [1.0 | 2.0(0.6) | 1.0]
        // because PEC enforces E=0 at x=0, matching the antisymmetric mode profile.
        let mut ml = MultiLayer::new(vec![Layer::new(2.0, 0.3), Layer::new(1.0, 1.0)]);
        ml.set_left_boundary(BoundaryCondition::PEC);
        ml
    }

    fn create_pec_right_half_slab() -> MultiLayer {
        // Mirror image of the PEC-left half slab; must give identical neff by symmetry.
        let mut ml = MultiLayer::new(vec![Layer::new(1.0, 1.0), Layer::new(2.0, 0.3)]);
        ml.set_right_boundary(BoundaryCondition::PEC);
        ml
    }

    #[test]
    fn test_pec_left_te_matches_odd_mode_of_full_slab() {
        // The odd TE mode of slab [1.0 | 2.0(0.6) | 1.0] (mode index 1) has the same
        // neff as the fundamental TE mode of the PEC-left half slab [PEC | 2.0(0.3) | 1.0],
        // because the PEC enforces the antisymmetric (sine-like) field profile.
        let om = 2.0 * PI / 1.55;
        let half_slab = create_pec_left_half_slab();
        let full_slab = create_slab_multilayer();
        let neff_half = half_slab.solve(om, Polarization::TE);
        let neff_full = full_slab.solve(om, Polarization::TE);
        // Full slab mode 1 (odd TE) = half slab mode 0
        assert_eq!(
            neff_half.len(),
            1,
            "PEC half slab should have exactly 1 TE mode"
        );
        assert!(
            neff_half[0].approx_eq(&neff_full[1], 1e-6),
            "PEC-left TE neff {:.9} != odd mode of full slab {:.9}",
            neff_half[0],
            neff_full[1]
        );
    }

    #[test]
    fn test_pec_left_tm_matches_even_hz_mode_of_full_slab() {
        // For TM modes the "main" field component tracked by the transfer matrix is Ez (not Hz).
        // In a symmetric slab the TM *even* mode (even Hz profile) has an *antisymmetric* Ez
        // profile (Ez ∝ dHz/dx ∝ sin), so Ez = 0 at the centre.  A PEC wall at x = 0 enforces
        // Ez = 0, therefore the PEC-left half slab matches TM mode 0 (the even-Hz / highest-neff
        // mode) of the full slab – not TM mode 1 as one might naively expect from the TE analogy.
        let om = 2.0 * PI / 1.55;
        let half_slab = create_pec_left_half_slab();
        let full_slab = create_slab_multilayer();
        let neff_half = half_slab.solve(om, Polarization::TM);
        let neff_full = full_slab.solve(om, Polarization::TM);
        assert_eq!(
            neff_half.len(),
            1,
            "PEC half slab should have exactly 1 TM mode"
        );
        assert!(
            neff_half[0].approx_eq(&neff_full[0], 1e-6),
            "PEC-left TM neff {:.9} != even-Hz mode of full slab {:.9}",
            neff_half[0],
            neff_full[0]
        );
    }

    #[test]
    fn test_pec_left_right_symmetry_te() {
        // A PEC-left slab and its mirror PEC-right slab must give identical neff values
        let om = 2.0 * PI / 1.55;
        let left = create_pec_left_half_slab();
        let right = create_pec_right_half_slab();
        let neff_left = left.solve(om, Polarization::TE);
        let neff_right = right.solve(om, Polarization::TE);
        assert_eq!(neff_left.len(), neff_right.len());
        for (nl, nr) in neff_left.iter().zip(neff_right.iter()) {
            assert!(
                nl.approx_eq(nr, 1e-6),
                "PEC-left TE {:.9} != PEC-right TE {:.9}",
                nl,
                nr
            );
        }
    }

    #[test]
    fn test_pec_left_right_symmetry_tm() {
        let om = 2.0 * PI / 1.55;
        let left = create_pec_left_half_slab();
        let right = create_pec_right_half_slab();
        let neff_left = left.solve(om, Polarization::TM);
        let neff_right = right.solve(om, Polarization::TM);
        assert_eq!(neff_left.len(), neff_right.len());
        for (nl, nr) in neff_left.iter().zip(neff_right.iter()) {
            assert!(
                nl.approx_eq(nr, 1e-6),
                "PEC-left TM {:.9} != PEC-right TM {:.9}",
                nl,
                nr
            );
        }
    }

    #[test]
    fn test_pec_field_satisfies_bc_te() {
        // The TE field of the PEC-left half slab must satisfy Ey = 0 at x=0 (the PEC wall).
        let om = 2.0 * PI / 1.55;
        let mut ml = create_pec_left_half_slab();
        ml.plot_step = 1e-4;
        let field = ml.field(om, Polarization::TE, 0).unwrap();
        // x[0] is the first grid point, which is at or very near x=0
        let ey_at_wall = field.Ey[0];
        assert!(
            ey_at_wall.norm() < 1e-6,
            "Ey at PEC wall should be ~0, got {:?}",
            ey_at_wall
        );
    }

    #[test]
    fn test_pec_field_satisfies_bc_tm() {
        // The TM field of the PEC-left half slab must satisfy Ez = 0 at x=0 (the PEC wall).
        // Note: for TM the even-Hz mode has antisymmetric Ez (Ez = 0 at the symmetry plane),
        // so it is this mode (mode 0, highest neff) that the PEC selects.
        let om = 2.0 * PI / 1.55;
        let mut ml = create_pec_left_half_slab();
        ml.plot_step = 1e-4;
        let field = ml.field(om, Polarization::TM, 0).unwrap();
        // x[0] is the first grid point, at or very near x=0 (the PEC wall)
        let ez_at_wall = field.Ez[0];
        assert!(
            ez_at_wall.norm() < 1e-6,
            "Ez at PEC wall should be ~0, got {:?}",
            ez_at_wall
        );
    }

    #[test]
    fn test_pec_field_continuity_te() {
        // Verify that the TE Ey field has no discontinuities at layer interfaces for PEC-left.
        let om = 2.0 * PI / 1.55;
        let mut ml = create_pec_left_half_slab();
        ml.plot_step = 1e-4;
        let field = ml.field(om, Polarization::TE, 0).unwrap();
        // Find the index corresponding to x ≈ layers[0].d = 0.3 (the only interior interface)
        let interface_x = 0.3_f64;
        let idx = field
            .x
            .iter()
            .position(|&x| (x - interface_x).abs() < 2e-4)
            .expect("Interface x not found in grid");
        // The field just before and just after the interface should be nearly equal
        let diff = (field.Ey[idx] - field.Ey[idx - 1]).norm();
        assert!(
            diff < 1e-2,
            "Ey has a discontinuity at the layer interface: diff = {:.6}",
            diff
        );
    }

    #[test]
    fn test_pec_field_continuity_tm() {
        // Verify that the TM Ez field has no discontinuities at layer interfaces for PEC-left.
        let om = 2.0 * PI / 1.55;
        let mut ml = create_pec_left_half_slab();
        ml.plot_step = 1e-4;
        let field = ml.field(om, Polarization::TM, 0).unwrap();
        let interface_x = 0.3_f64;
        let idx = field
            .x
            .iter()
            .position(|&x| (x - interface_x).abs() < 2e-4)
            .expect("Interface x not found in grid");
        let diff = (field.Ez[idx] - field.Ez[idx - 1]).norm();
        assert!(
            diff < 1e-2,
            "Ez has a discontinuity at the layer interface: diff = {:.6}",
            diff
        );
    }

    // -----------------------------------------------------------------------
    // Multi-layer PEC tests:
    //   [PEC | n=1.0(d=0.2) | n=2.0(d=0.5) | n=1.0] and its mirror.
    // These exercise the full physical propagation path through an intermediate
    // cladding layer between the PEC wall and the guiding core — the original
    // source of the characteristic-function bug that was fixed.
    // -----------------------------------------------------------------------

    fn create_multi_pec_left() -> MultiLayer {
        // [PEC | n=1.0(d=0.2) | n=2.0(d=0.5) | n=1.0(semi-inf)]
        let mut ml = MultiLayer::new(vec![
            Layer::new(1.0, 0.2),
            Layer::new(2.0, 0.5),
            Layer::new(1.0, 2.0),
        ]);
        ml.set_left_boundary(BoundaryCondition::PEC);
        ml
    }

    fn create_multi_pec_right() -> MultiLayer {
        // Mirror of multi_pec_left: [n=1.0(semi-inf) | n=2.0(d=0.5) | n=1.0(d=0.2) | PEC]
        let mut ml = MultiLayer::new(vec![
            Layer::new(1.0, 2.0),
            Layer::new(2.0, 0.5),
            Layer::new(1.0, 0.2),
        ]);
        ml.set_right_boundary(BoundaryCondition::PEC);
        ml
    }

    #[test]
    fn test_multi_pec_left_right_symmetry_te() {
        // Mirror PEC-left / PEC-right structures must give identical TE neff.
        let om = 2.0 * PI / 1.55;
        let left = create_multi_pec_left();
        let right = create_multi_pec_right();
        let neff_left = left.solve(om, Polarization::TE);
        let neff_right = right.solve(om, Polarization::TE);
        assert_eq!(
            neff_left.len(),
            neff_right.len(),
            "PEC-left and PEC-right must find the same number of TE modes"
        );
        for (nl, nr) in neff_left.iter().zip(neff_right.iter()) {
            assert!(
                nl.approx_eq(nr, 1e-6),
                "Multi-layer PEC-left TE {:.9} != PEC-right TE {:.9}",
                nl,
                nr
            );
        }
    }

    #[test]
    fn test_multi_pec_left_right_symmetry_tm() {
        // Mirror PEC-left / PEC-right structures must give identical TM neff.
        let om = 2.0 * PI / 1.55;
        let left = create_multi_pec_left();
        let right = create_multi_pec_right();
        let neff_left = left.solve(om, Polarization::TM);
        let neff_right = right.solve(om, Polarization::TM);
        assert_eq!(
            neff_left.len(),
            neff_right.len(),
            "PEC-left and PEC-right must find the same number of TM modes"
        );
        for (nl, nr) in neff_left.iter().zip(neff_right.iter()) {
            assert!(
                nl.approx_eq(nr, 1e-6),
                "Multi-layer PEC-left TM {:.9} != PEC-right TM {:.9}",
                nl,
                nr
            );
        }
    }

    #[test]
    fn test_multi_pec_left_te_ey_zero_at_wall() {
        // TE Ey must vanish at the PEC wall (x = 0) for the multi-layer structure.
        let om = 2.0 * PI / 1.55;
        let mut ml = create_multi_pec_left();
        ml.plot_step = 1e-4;
        let field = ml.field(om, Polarization::TE, 0).unwrap();
        let ey_at_wall = field.Ey[0];
        assert!(
            ey_at_wall.norm() < 1e-6,
            "Multi-layer PEC-left: Ey at wall should be ~0, got {:?}",
            ey_at_wall
        );
    }

    #[test]
    fn test_multi_pec_left_tm_ez_zero_at_wall() {
        // TM Ez must vanish at the PEC wall (x = 0) for the multi-layer structure.
        let om = 2.0 * PI / 1.55;
        let mut ml = create_multi_pec_left();
        ml.plot_step = 1e-4;
        let field = ml.field(om, Polarization::TM, 0).unwrap();
        let ez_at_wall = field.Ez[0];
        assert!(
            ez_at_wall.norm() < 1e-6,
            "Multi-layer PEC-left: Ez at wall should be ~0, got {:?}",
            ez_at_wall
        );
    }

    #[test]
    fn test_multi_pec_left_te_ey_continuous_at_interfaces() {
        // TE Ey must be continuous at both dielectric interfaces for PEC-left.
        // Interface 1: n=1.0 | n=2.0 at x = 0.2
        // Interface 2: n=2.0 | n=1.0 at x = 0.7
        let om = 2.0 * PI / 1.55;
        let mut ml = create_multi_pec_left();
        ml.plot_step = 1e-4;
        let field = ml.field(om, Polarization::TE, 0).unwrap();
        for (interface_x, label) in [(0.2_f64, "n=1|n=2"), (0.7_f64, "n=2|n=1")] {
            let idx = field
                .x
                .iter()
                .position(|&x| (x - interface_x).abs() < 2e-4)
                .unwrap_or_else(|| panic!("Interface {} not found in grid", label));
            // Compare one step before and one step after the interface
            let diff = (field.Ey[idx + 1] - field.Ey[idx - 1]).norm();
            assert!(
                diff < 0.1,
                "Multi-layer PEC-left TE: Ey has discontinuity at {} interface: diff = {:.6}",
                label,
                diff
            );
        }
    }

    #[test]
    fn test_multi_pec_left_tm_ez_continuous_at_interfaces() {
        // TM Ez must be continuous at both dielectric interfaces for PEC-left.
        let om = 2.0 * PI / 1.55;
        let mut ml = create_multi_pec_left();
        ml.plot_step = 1e-4;
        let field = ml.field(om, Polarization::TM, 0).unwrap();
        for (interface_x, label) in [(0.2_f64, "n=1|n=2"), (0.7_f64, "n=2|n=1")] {
            let idx = field
                .x
                .iter()
                .position(|&x| (x - interface_x).abs() < 2e-4)
                .unwrap_or_else(|| panic!("Interface {} not found in grid", label));
            let diff = (field.Ez[idx + 1] - field.Ez[idx - 1]).norm();
            assert!(
                diff < 0.1,
                "Multi-layer PEC-left TM: Ez has discontinuity at {} interface: diff = {:.6}",
                label,
                diff
            );
        }
    }

    #[test]
    fn test_multi_pec_right_te_ey_continuous_at_interfaces() {
        // TE Ey must be continuous at both dielectric interfaces for PEC-right.
        // For [n=1(d=2), n=2(d=0.5), n=1(d=0.2), PEC] with xstart = -2.0:
        // Interface 1: n=1.0 | n=2.0 at x = 0.0
        // Interface 2: n=2.0 | n=1.0 at x = 0.5
        let om = 2.0 * PI / 1.55;
        let mut ml = create_multi_pec_right();
        ml.plot_step = 1e-4;
        let field = ml.field(om, Polarization::TE, 0).unwrap();
        for (interface_x, label) in [(0.0_f64, "n=1|n=2"), (0.5_f64, "n=2|n=1")] {
            let idx = field
                .x
                .iter()
                .position(|&x| (x - interface_x).abs() < 2e-4)
                .unwrap_or_else(|| panic!("Interface {} not found in grid", label));
            let diff = (field.Ey[idx + 1] - field.Ey[idx - 1]).norm();
            assert!(
                diff < 0.1,
                "Multi-layer PEC-right TE: Ey has discontinuity at {} interface: diff = {:.6}",
                label,
                diff
            );
        }
    }

    #[test]
    fn test_multi_pec_right_tm_ez_continuous_at_interfaces() {
        // TM Ez must be continuous at both dielectric interfaces for PEC-right.
        let om = 2.0 * PI / 1.55;
        let mut ml = create_multi_pec_right();
        ml.plot_step = 1e-4;
        let field = ml.field(om, Polarization::TM, 0).unwrap();
        for (interface_x, label) in [(0.0_f64, "n=1|n=2"), (0.5_f64, "n=2|n=1")] {
            let idx = field
                .x
                .iter()
                .position(|&x| (x - interface_x).abs() < 2e-4)
                .unwrap_or_else(|| panic!("Interface {} not found in grid", label));
            let diff = (field.Ez[idx + 1] - field.Ez[idx - 1]).norm();
            assert!(
                diff < 0.1,
                "Multi-layer PEC-right TM: Ez has discontinuity at {} interface: diff = {:.6}",
                label,
                diff
            );
        }
    }
}
