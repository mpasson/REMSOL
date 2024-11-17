extern crate num_complex;

use iter_num_tools::arange;
use itertools::iproduct;
use num_complex::Complex;
use remsol::enums::BackEnd;
use remsol::enums::Polarization;
use remsol::layer::Layer;
use remsol::multilayer::GridData;
use remsol::multilayer::MultiLayer;
use std::{f64::consts::PI, fmt::Debug};

fn main() {
    // let mut multi = MultiLayer::new(vec![
    //     Layer::new(1.0.into(), 1.0),
    //     Layer::new(Complex::new(2.0, -0.1), 0.6),
    //     Layer::new(1.0.into(), 1.0),
    // ]);
    let mut multi = MultiLayer::new(vec![
        Layer::new(1.0.into(), 1.0),
        Layer::new(2.0.into(), 0.6),
        Layer::new(1.0.into(), 1.0),
        Layer::new(2.0.into(), 0.5),
    ]);

    let om = 2.0 * PI / 1.55;
    multi.set_backend(BackEnd::Scattering);

    // let k_real = arange(om..2.0 * om, 1e-3);
    let k_real = arange(4.82..4.84, 1e-5);
    let k_imag = arange(-0.0002..0.0002, 1e-6);

    let k_v: Vec<Complex<f64>> = iproduct!(k_real, k_imag)
        .map(|(re, im)| Complex::new(re, im))
        .collect();

    let char = k_v
        .iter()
        .map(|&k| multi.characteristic_function(om, k, Polarization::TE))
        .collect::<Vec<_>>();
    for (k, char) in k_v.iter().zip(char.iter()) {
        println!("{},{},{},{}", k.re, k.im, char.re, char.im);
    }
}
