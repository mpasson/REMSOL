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

    pub fn matrix_propagation<T, U, V, W>(n: T, d: U, om: V, k: W) -> ScatteringMatrix
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
        let a = ((om * n).powi(2) - k.powi(2)).sqrt();
        let phase = Complex::new(0.0, 1.0) * a * d;
        ScatteringMatrix {
            s11: phase.exp(),
            s12: Complex::new(0.0, 0.0),
            s21: Complex::new(0.0, 0.0),
            s22: phase.exp(),
        }
    }

    pub fn matrix_interface_te<T, U, V, W>(n1: T, n2: U, om: V, k: W) -> ScatteringMatrix
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
        ScatteringMatrix {
            s11: 2.0 * k2 / (k1 + k2),
            s12: (k2 - k1) / (k1 + k2),
            s21: (k1 - k2) / (k1 + k2),
            s22: 2.0 * k1 / (k1 + k2),
        }
    }
    pub fn matrix_interface_tm<T, U, V, W>(n1: T, n2: U, om: V, k: W) -> ScatteringMatrix
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
        ScatteringMatrix {
            s11: 2.0 * k2 / (k1 + k2),
            s12: (k2 - k1) / (k1 + k2),
            s21: (k1 - k2) / (k1 + k2),
            s22: 2.0 * k1 / (k1 + k2),
        }
    }

    pub fn matrix_interface<T, U, V, W>(
        n1: T,
        n2: U,
        om: V,
        k: W,
        polarization: Polarization,
    ) -> ScatteringMatrix
    where
        T: Into<Complex<f64>>,
        U: Into<Complex<f64>>,
        V: Into<Complex<f64>>,
        W: Into<Complex<f64>>,
    {
        match polarization {
            Polarization::TE => ScatteringMatrix::matrix_interface_te(n1, n2, om, k),
            Polarization::TM => ScatteringMatrix::matrix_interface_tm(n1, n2, om, k),
        }
    }
}

pub fn calculate_s_matrix<T, U>(
    layers: &Vec<Layer>,
    om: T,
    k: U,
    polarization: Polarization,
) -> ScatteringMatrix
where
    T: Into<Complex<f64>> + Copy,
    U: Into<Complex<f64>> + Copy,
{
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
