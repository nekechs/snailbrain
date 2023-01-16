mod engine;

use engine::comp_graph;
use ndarray::prelude::*;

fn main() {
    let x = comp_graph::zeros(&[15], false);

    let W_0 = comp_graph::zeros(&[20, 15], true);
    let b_0 = comp_graph::zeros(&[20], true);
    let z_0 = W_0.mv_mul(&x).add(&b_0);
    let a_0 = z_0.sigmoid();

    let W_1 = comp_graph::zeros(&[20, 20], true);
    let b_1 = comp_graph::zeros(&[20], true);
    let z_1 = W_1.mv_mul(&a_0).add(&b_1);
    let a_1 = z_1.relu();
}
