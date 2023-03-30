mod tens;
mod autograd;

use std::{rc::Rc, cell::RefCell, borrow::Borrow};

use ndarray::{prelude::*, array, Ix2, Ix1};

use autograd::tape::Tape;
use tens::view::*;
// use ndarray::prelude::*;


fn main() {
    // let arr = array![1.0, 2.0, 3.0];
    // let arr2 = arr.into_dimensionality::<Ix2>();
    // println!("{arr}");

    let graph = Tape::new();
    let A = graph.from_elem_grad(Ix2(3, 4), 5.);
    let x = graph.from_elem(Ix1(4), 2.);

    let b = graph.from_elem(Ix1(3), -1.0);

    let y = &A.mv(&x) + &b;

    graph.forward();
    
    println!("{}", y.output.borrow_mut());

    {
        x.output.borrow_mut().assign(&Array1::zeros(3));
    }

    graph.forward();

    println!("{}", y.output.borrow_mut());
} 
