extern crate num_complex;
extern crate rand;

use num_complex::Complex;
use rand::Rng;

fn main() {
    let mut rng = rand::thread_rng();

    let real1: f64 = rng.gen();
    let imag1: f64 = rng.gen();
    let complex1 = Complex::new(real1, imag1);

    let real2: f64 = rng.gen();
    let imag2: f64 = rng.gen();
    let complex2 = Complex::new(real2, imag2);

    let sum = complex1 + complex2;

    println!("The sum of {} and {} is {}", complex1, complex2, sum);
}
