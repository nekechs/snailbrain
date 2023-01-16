/// SUPPORT FOR COMPUTATIONAL GRAPHS
/// So far: Only have support for the graph structure itself
/// Main idea: Dissociate the graph's structure from any computational structures.
/// This is so we may have GPU support down the line, where we keep data in VRAM.

use ndarray::prelude::*;

use std::rc::Rc;

pub struct Variable<T> {
    nodes: Rc<T>,
    dimension: Vec<usize>,
}

pub fn zeros(dim: &[usize], requires_grad: bool) -> Variable<Leaf> {
    Variable {
        nodes: Rc::from(Leaf {requires_grad}),
        dimension: Vec::from(dim)
    }
}

impl <T> Variable <T> {
    pub fn add<R>(&self, rhs: &Variable<R>) -> Variable<Addition<T, R>> {
        if self.dimension != rhs.dimension {
            panic!("Error while adding two variables of different dimensions.");
        }

        let operation = Addition {
            first_operand: self.nodes.clone(),
            second_operand: rhs.nodes.clone(),
        };
        Variable {
            nodes: Rc::from(operation),
            dimension: self.dimension.clone(),
        }
    }

    pub fn subtract<R>(&self, rhs: &Variable<R>) -> Variable<Subtraction<T, R>> {
        if self.dimension != rhs.dimension {
            panic!("Error while adding two variables of different dimensions.");
        }

        let operation = Subtraction {
            first_operand: self.nodes.clone(),
            second_operand: rhs.nodes.clone(),
        };
        Variable {
            nodes: Rc::from(operation),
            dimension: self.dimension.clone(),
        }
    }

    pub fn mm_mul<R>(&self, rhs: &Variable<R>) -> Variable<MatMatMul<T, R>> {
        /* 
        
        TODO: Add logic here that detects dimension mismatch for matrix multiplication.

        */

        if !(self.dimension.len() == 2 && rhs.dimension.len() == 2 && self.dimension[1] == rhs.dimension[0]) {
            panic!("Error with matrix multiplication dimensions.");
        }

        let operation = MatMatMul {
            first_operand: self.nodes.clone(),
            second_operand: rhs.nodes.clone(),
        };
        Variable {
            nodes: Rc::from(operation),
            dimension: vec![self.dimension[0], rhs.dimension[1]],
        }
    }

    pub fn mv_mul<R>(&self, rhs: &Variable<R>) -> Variable<MatVecMul<T, R>> {
        if !(self.dimension.len() == 2 && rhs.dimension.len() == 1 && self.dimension[1] == rhs.dimension[0]) {
            panic!("Error with matrix vector multiplication dimensions.");
        }

        let operation = MatVecMul {
            matrix: self.nodes.clone(),
            vector: rhs.nodes.clone(),
        };
        Variable {
            nodes: Rc::from(operation),
            dimension: vec![self.dimension[0]],
        }
    }

    pub fn sigmoid(&self) -> Variable<Sigmoid<T>> {
        Variable {
            nodes: Rc::from( Sigmoid{input: self.nodes.clone()}),
            dimension: self.dimension.clone(),
        }
    }

    pub fn tanh(&self) -> Variable<Sigmoid<T>> {
        Variable {
            nodes: Rc::from( Sigmoid{input: self.nodes.clone()}),
            dimension: self.dimension.clone(),
        }
    }

    pub fn relu(&self) -> Variable<Sigmoid<T>> {
        Variable {
            nodes: Rc::from( Sigmoid{input: self.nodes.clone()}),
            dimension: self.dimension.clone(),
        }
    }
}


/* Intention here is to specify operations common to nodes */
/* Examples: Checking to see if node requires_grad, serialization, etc.` */
pub trait Node {

}

pub struct Leaf {
    requires_grad: bool
}

pub struct Addition<X, Y> {
    first_operand: Rc<X>,
    second_operand: Rc<Y>
}

pub struct Subtraction<X, Y> {
    first_operand: Rc<X>,
    second_operand: Rc<Y>
}

pub struct MatMatMul<X, Y> {
    first_operand: Rc<X>,
    second_operand: Rc<Y>
}

pub struct MatVecMul<X, Y> {
    matrix: Rc<X>,
    vector: Rc<Y>
}

pub struct Sigmoid<X> {
    input: Rc<X>
}

pub struct Tanh<X> {
    input: Rc<X>
}


pub struct ReLU<X> {
    input: Rc<X>
}

