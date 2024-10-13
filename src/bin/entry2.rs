extern crate num_complex;

use num_complex::Complex;
use remsol::enums::Polarization;
use remsol::layer::Layer;
use remsol::multilayer::GridData;
use remsol::multilayer::MultiLayer;
use std::{f64::consts::PI, fmt::Debug};

fn main() {
    let slab = MultiLayer::new(vec![
        Layer::new(1.0, 1.0),
        Layer::new(2.0, 0.6),
        Layer::new(1.0, 1.0),
    ]);
    let field = slab.field(2.0 * PI / 1.55, Polarization::TE, 0).unwrap();
    let amplitude = field.Ey[1300];
    println!("{:?}", amplitude);
}
