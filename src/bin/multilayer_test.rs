use iter_num_tools;
use std::f64::consts::PI;

use scatterning_matrix::Layer;
use scatterning_matrix::MultiLayer;

use scatterning_matrix::calculate_s_matrix;
use scatterning_matrix::calculate_structure_determinant;
use scatterning_matrix::find_minmax_n;

fn print_det(layers: &Vec<Layer>, om: f64) {
    let (n_min, nmax) = find_minmax_n(layers);
    let (k_min, k_max) = (n_min * om, nmax * om);
    let step = 1e-3;
    for k in iter_num_tools::arange(k_min..k_max, step) {
        println!(
            "{} {}",
            k,
            calculate_structure_determinant(&layers, om, k).norm()
        );
    }
}

fn find_maxima(layers: &Vec<Layer>, om: f64, k_min: f64, k_max: f64, step: f64) -> Vec<f64> {
    let mut going_up = false;
    let mut maxima: Vec<f64> = Vec::new();
    let mut current = calculate_structure_determinant(&layers, om, k_min)
        .norm()
        .log10();
    let next = calculate_structure_determinant(&layers, om, k_min + step)
        .norm()
        .log10();

    if next > current {
        going_up = true;
    }

    current = next;

    for k in iter_num_tools::arange(k_min..k_max, step).skip(2) {
        let next = calculate_structure_determinant(&layers, om, k)
            .norm()
            .log10();
        if going_up {
            if next < current && current > 1.0 {
                going_up = false;
                maxima.push(k - step);
            }
        } else {
            if next > current {
                going_up = true;
            }
        }
        current = next;
        // print!("{} {} {}\n", k, next, going_up);
    }
    maxima
}

fn print_solutions(solution: &Vec<f64>, layers: &Vec<Layer>, om: f64) {
    for k in solution.iter() {
        println!(
            "{} {}",
            k / om,
            calculate_structure_determinant(&layers, om, *k)
                .norm()
                .log10()
        );
    }
    println!("");
}

fn main() {
    let layers: Vec<Layer> = vec![
        Layer::new(1.0, 1.0),
        Layer::new(2.0, 0.6),
        Layer::new(1.0, 0.6),
        Layer::new(2.0, 0.6),
        Layer::new(1.0, 1.0),
        // Layer::new(2.0, 0.4),
        // Layer::new(1.0, 1.0),
    ];
    let om = 2.0 * PI / 1.55;
    let k0 = 0.0;

    // print_det(&layers, om);

    let mut multi_layer = MultiLayer::new(layers);
    multi_layer.set_iteration(3);
    multi_layer.set_solution_threshold(0.5);

    multi_layer.solve(om);

    // let (min_n, max_n) = find_minmax_n(&layers);
    // let (min_k, max_k) = (min_n * om, max_n * om);

    // let solution1 = find_maxima(&layers, om, min_k, max_k, 0.001);
    // print_solutions(&solution1, &layers, om);

    // let mut solution2 = Vec::new();
    // for draft in solution1.iter() {
    //     let mut solution = find_maxima(&layers, om, draft - 0.001, draft + 0.001, 1e-6);
    //     solution2.append(&mut solution);
    // }
    // print_solutions(&solution2, &layers, om);

    // let mut solution3 = Vec::new();
    // for draft in solution2.iter() {
    //     let mut solution = find_maxima(&layers, om, draft - 1e-6, draft + 1e-6, 1e-9);
    //     solution3.append(&mut solution);
    // }
    // print_solutions(&solution3, &layers, om);
    // for k in iter_num_tools::arange(min_k..max_k, 0.001) {
    //     println!(
    //         "{} {}",
    //         k,
    //         calculate_structure_determinant(&layers, om, k).norm()
    //     );
    // }
}

// for k in iter_num_tools::arange(min_k..max_k, 0.001) {
//     println!(
//         "{} {}",
//         k,
//         calculate_structure_determinant(&layers, om, k).norm()
//     );
// }
