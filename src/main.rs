mod engine;

use engine::comp_graph;

fn main() {
    let x = comp_graph::zeros(&[4, 4]);
    let y = comp_graph::zeros(&[4, 4]);
    let z = comp_graph::zeros(&[4, 4]);

    let a1 = x.add(&y);
    let a2 = y.add(&z);
    let a3 = x.add(&z);

    let b1 = z.add(&a2);
    let b2 = a3.add(&a2);
}
