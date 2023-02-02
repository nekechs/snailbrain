/// SUPPORT FOR COMPUTATIONAL GRAPHS
/// So far: Only have support for the graph structure itself
/// Main idea: Dissociate the graph's structure from any computational structures.
/// This is so we may have GPU support down the line, where we keep data in VRAM.

use ndarray::prelude::*;

use std::rc::Rc;

pub struct Variable<T> {
    nodes: Rc<T>,
    dimension: Vec<usize>,

    forward_cache: Option<ForwardGraph>,
}

pub struct ForwardGraph {
    // topo_nodes: Vec<Rc<dyn Node>>,
    topo_nodes: Vec<>

    /* Essentially, for each index in topo_nodes, the corresp. vector stored in forward_edges contains
    a list of ALL nodes that the original nodes has a forward edge with. */
    forward_edges: Vec<Vec<usize>>,
}

pub fn zeros(dim: &[usize], requires_grad: bool) -> Variable<Leaf> {
    let grad;
    if requires_grad {
        grad = Some(ArrayD::zeros(dim));
    } else {
        grad = None;
    }

    Variable {
        nodes: Rc::from(Leaf {requires_grad: grad, buffer: ArrayD::zeros(dim)}),
        dimension: Vec::from(dim),
        forward_cache: None,
    }
}

impl <T> Variable <T> {
    pub fn forward(&mut self) {

    }

    pub fn backward(&mut self) {
        
    }
}

/* Operations on variables */
impl <T> Variable <T> {
    pub fn add<R>(&self, rhs: &Variable<R>) -> Variable<Addition<T, R>> {
        if self.dimension != rhs.dimension {
            panic!("Error while adding two variables of different dimensions.");
        }

        let operation = Addition {
            first_operand: self.nodes.clone(),
            second_operand: rhs.nodes.clone(),
            buffer: ArrayD::zeros(&self.dimension[..]),
            grad: ArrayD::zeros(&self.dimension[..]),
        };
        Variable {
            nodes: Rc::from(operation),
            dimension: self.dimension.clone(),
            forward_cache: None,
        }
    }

    pub fn subtract<R>(&self, rhs: &Variable<R>) -> Variable<Subtraction<T, R>> {
        if self.dimension != rhs.dimension {
            panic!("Error while adding two variables of different dimensions.");
        }

        let operation = Subtraction {
            first_operand: self.nodes.clone(),
            second_operand: rhs.nodes.clone(),
            buffer: ArrayD::zeros(&self.dimension[..]),
            grad: ArrayD::zeros(&self.dimension[..]),
        };
        Variable {
            nodes: Rc::from(operation),
            dimension: self.dimension.clone(),
            forward_cache: None,
        }
    }

    pub fn mm_mul<R>(&self, rhs: &Variable<R>) -> Variable<MatMatMul<T, R>> {
        /* 
        
        TODO: Add logic here that detects dimension mismatch for matrix multiplication.

        */

        if !(self.dimension.len() == 2 && rhs.dimension.len() == 2 && self.dimension[1] == rhs.dimension[0]) {
            panic!("Error with matrix multiplication dimensions.");
        }

        let res_dimension = vec![self.dimension[0], self.dimension[1]];

        let operation = MatMatMul {
            first_operand: self.nodes.clone(),
            second_operand: rhs.nodes.clone(),
            buffer: ArrayD::zeros(&res_dimension[..]),
            grad: ArrayD::zeros(&res_dimension[..]),
        };
        Variable {
            nodes: Rc::from(operation),
            dimension: res_dimension,
            forward_cache: None,
        }
    }

    pub fn mv_mul<R>(&self, rhs: &Variable<R>) -> Variable<MatVecMul<T, R>> {
        if !(self.dimension.len() == 2 && rhs.dimension.len() == 1 && self.dimension[1] == rhs.dimension[0]) {
            panic!("Error with matrix vector multiplication dimensions.");
        }

        let res_dimension = vec![self.dimension[0]];

        let operation = MatVecMul {
            matrix: self.nodes.clone(),
            vector: rhs.nodes.clone(),
            buffer: ArrayD::zeros(&res_dimension[..]),
            grad: ArrayD::zeros(&res_dimension[..]),
        };
        Variable {
            nodes: Rc::from(operation),
            dimension: res_dimension,
            forward_cache: None,
        }
    }

    pub fn sigmoid(&self) -> Variable<Sigmoid<T>> {
        Variable {
            nodes: Rc::from( Sigmoid{
                input: self.nodes.clone(),
                buffer: ArrayD::zeros(&self.dimension[..]),
                grad: ArrayD::zeros(&self.dimension[..]),
            }),
            dimension: self.dimension.clone(),
            forward_cache: None,
        }
    }

    pub fn tanh(&self) -> Variable<Tanh<T>> {
        Variable {
            nodes: Rc::from( Tanh{
                input: self.nodes.clone(),
                buffer: ArrayD::zeros(&self.dimension[..]),
                grad: ArrayD::zeros(&self.dimension[..]),
            }),
            dimension: self.dimension.clone(),
            forward_cache: None,
        }
    }

    pub fn relu(&self) -> Variable<ReLU<T>> {
        Variable {
            nodes: Rc::from( ReLU{
                input: self.nodes.clone(),
                buffer: ArrayD::zeros(&self.dimension[..]),
                grad: ArrayD::zeros(&self.dimension[..]),
            }),
            dimension: self.dimension.clone(),
            forward_cache: None,
        }
    }
}


/* Intention here is to specify operations common to nodes */
/* Examples: Checking to see if node requires_grad, serialization, etc.` */
pub trait Node<X, Y, Z, W> {    // Note how we have 4 generic parameters. Each node supports up to 4 operands.

}

pub struct Leaf {
    requires_grad: Option<ArrayD<f32>>,
    buffer: ArrayD<f32>,
}

impl<X, Y, Z, W> Node<X, Y, Z, W> for Leaf {

}

pub struct Addition<X: ?Sized, Y: ?Sized> {
    first_operand: Rc<X>,
    second_operand: Rc<Y>,

    buffer: ArrayD<f32>,
    grad: ArrayD<f32>,
}

impl<X, Y, Z, W> Node<X, Y, Z, W> for Addition<X, Y> {

}

pub struct Subtraction<X: ?Sized, Y: ?Sized> {
    first_operand: Rc<X>,
    second_operand: Rc<Y>,

    buffer: ArrayD<f32>,
    grad: ArrayD<f32>,
}

impl<X, Y, Z, W> Node<X, Y, Z, W> for Subtraction<X, Y> {

}

pub struct MatMatMul<X: ?Sized, Y: ?Sized> {
    first_operand: Rc<X>,
    second_operand: Rc<Y>,

    buffer: ArrayD<f32>,
    grad: ArrayD<f32>,
}

impl<X, Y, Z, W> Node<X, Y, Z, W> for MatMatMul<X, Y> {

}

pub struct MatVecMul<X: ?Sized, Y: ?Sized> {
    matrix: Rc<X>,
    vector: Rc<Y>,

    buffer: ArrayD<f32>,
    grad: ArrayD<f32>,
}

impl<X, Y, Z, W> Node<X, Y, Z, W> for MatVecMul<X, Y> {

}

pub struct Sigmoid<X: ?Sized> {
    input: Rc<X>,

    buffer: ArrayD<f32>,
    grad: ArrayD<f32>,
}

impl<X, Y, Z, W> Node<X, Y, Z, W> for Sigmoid<X> {

}

pub struct Tanh<X: ?Sized> {
    input: Rc<X>,

    buffer: ArrayD<f32>,
    grad: ArrayD<f32>,
}

impl<X, Y, Z, W> Node<X, Y, Z, W> for Tanh<X> {

}

pub struct ReLU<X: ?Sized> {
    input: Rc<X>,

    buffer: ArrayD<f32>,
    grad: ArrayD<f32>,
}

impl<X, Y, Z, W> Node<X, Y, Z, W> for ReLU<X> {

}