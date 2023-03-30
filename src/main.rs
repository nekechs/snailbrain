mod tens;
mod autograd;

use std::{rc::Rc, cell::RefCell, borrow::Borrow};

use ndarray::{prelude::*, array, Ix2, Ix1};

use autograd::tape::Tape;
use tens::view::*;
// use ndarray::prelude::*;


fn main() {
    let arr = Array::<f64, _>::zeros(Ix1(5));//array![1.0, 2.0, 3.0];
    let arr2 = arr.broadcast(Ix2(1, 5));
    println!("{:?}", arr2);

    let graph = Tape::new();
    let A = graph.from_elem_grad(Ix2(3, 4), 5.);
    let x = graph.from_elem(Ix1(4), 3.);

    let b = graph.from_elem_grad(Ix1(3), -1.0);

    let y = &A.mv(&x) + &b;
    // let y = &x + &b;

    graph.forward();
    
    println!("{}", y.output.borrow_mut());

    y.backward(array![1.0, 1.0, 1.0]);

    println!("{}", A.grad.unwrap().borrow_mut());
} 
