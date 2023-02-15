mod engine;

use engine::comp_graph::*;
use ndarray::prelude::*;

fn main() {
    let mut comp_graph = Graph::new();
    let a = comp_graph.zeros(&[2, 3], true).unwrap();
    let b = comp_graph.zeros(&[2, 3], false).unwrap();
    let c = comp_graph.add(a, b).unwrap();
    let c_prime = comp_graph.sigmoid(c).unwrap();

    let d = comp_graph.zeros(&[3, 4], true).unwrap();
    let e = comp_graph.zeros(&[3, 4], false).unwrap();
    let f = comp_graph.add(d, e).unwrap();
    let f_prime = comp_graph.sigmoid(f).unwrap();
    
    let out = comp_graph.mm(c_prime, f_prime).unwrap();

    // println!("{:#?}", comp_graph);
    println!("{:?}", comp_graph.topsort_backward(f_prime));

    let m = 4;
    // for items in comp_graph.nodes[out].op.iter() {
    //     println!("{items}");
    // }
} 
