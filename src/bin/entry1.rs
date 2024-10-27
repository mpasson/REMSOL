extern crate num_complex;

use num_complex::Complex;
use remsol::enums::Polarization;
use remsol::layer::Layer;
use remsol::multilayer::MultiLayer;
use std::{f64::consts::PI, fmt::Debug};

fn set_slice(slice: &mut [f64], number_to_set: f64) {
    for index in 0..slice.len() {
        slice[index] = number_to_set;
    }
}

fn test_set_slice() {
    let mut vector: Vec<f64> = iter_num_tools::arange(0.0..1.0, 0.1).collect();

    set_slice(&mut vector[1..4], -1.0);
    println!("{:?}", vector)
}

fn main() {
    let multi = MultiLayer::new(vec![
        Layer::new(1.0, 1.0),
        Layer::new(2.0, 0.6),
        Layer::new(1.0, 1.0),
    ]);
    let om = 2.0 * PI / 1.55;
    let n = multi.neff(om, Polarization::TM, 0).unwrap();
    println!("{:?}", n);
    let coefficients = multi.get_propagation_coefficients(
        om,
        om * n,
        Polarization::TM,
        Complex::new(0.0, 0.0),
        Complex::new(1.0, 0.0),
    );
    for vector in coefficients {
        println!("{:?}", vector);
    }
}
