extern crate num_complex;

use multilayer_solver::enums::Polarization;
use multilayer_solver::layer::Layer;
use multilayer_solver::multilayer::GridData;
use multilayer_solver::multilayer::MultiLayer;
use num_complex::Complex;

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
    let om = 2.0 * 3.1415 / 1.55;
    let n = multi.neff(om, Polarization::TE, 0).unwrap();
    println!("{:?}", n);
    let coefficients = multi.get_propagation_coefficients(
        om,
        om * n,
        Polarization::TE,
        Complex::new(0.0, 0.0),
        Complex::new(1.0, 0.0),
    );
    for vector in coefficients {
        println!("{:?}", vector);
    }
}
