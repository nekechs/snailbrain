mod engine1;
mod tens;
mod autograd;

use std::{rc::Rc, cell::RefCell, borrow::Borrow};

use ndarray::{prelude::*, array, Ix2, Ix1};

use autograd::tape::Tape;
use engine1::comp_graph::*;
use tens::view::*;
// use ndarray::prelude::*;


fn main() {
    let graph = Tape::new();
    let A = graph.from_elem(Ix2(3, 3), 5.);
    let x = graph.from_elem(Ix1(3), 2.);

    let y = A.mv(&x);

    graph.forward();
    
    println!("{}", y.output.borrow_mut());

    {
        x.output.borrow_mut().assign(&Array1::zeros(3));
    }

    graph.forward();

    println!("{}", y.output.borrow_mut());
} 
