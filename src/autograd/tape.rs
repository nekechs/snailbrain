
use std::{cell::RefCell, rc::Rc};
use ndarray::{prelude::*, RawData};
use num_traits::Zero;

use super::{operations::{BackwardOp, ForwardOp, leaf::{LeafForward, LeafBackward}}, expr::Expression};

pub struct Node {
    pub(crate) fw: Box<dyn ForwardOp>,
    pub(crate) bw: Option<Box<dyn BackwardOp>>,
    pub(crate) operands: Vec<usize>
}

impl Node {
    pub fn forward(&self) {
        self.fw.forward();
    }

    pub fn backward(&self) {
        if let Some(bw) = &self.bw {
            bw.backward();
        }
    }
}

pub struct Tape {
    pub(crate) nodes: RefCell<Vec<Node>>
}

impl Tape {
    pub fn new() -> Self{
        Tape { nodes: RefCell::new(Vec::new()) }
    }
    // Create methods for pushing 1 and 2 nodes at a time:

    pub fn from_elem<'t, T, D>(&'t self, dim: D, val: T) -> Expression<'t, T, D>
    where
        T: Zero + Clone + 'static,
        D: Dimension + 'static
    {
        let arr_ref = Rc::from(RefCell::new(Array::from_elem(dim, val)));

        let leaf_node = Node {
            fw: Box::from(LeafForward {
                value: arr_ref.clone()
            }),
            bw: None,
            operands: vec![]
        };

        Expression {
            tape: &self,
            index: self.insert(leaf_node),
            output: arr_ref.clone(),
            grad: None
        }
    }

    pub fn from_elem_grad<'t, T, D>(&'t self, dim: D, val: T) -> Expression<'t, T, D>
    where
        T: Zero + Clone + 'static,
        D: Dimension + 'static
    {
        let arr_ref = Rc::from(RefCell::new(Array::from_elem(dim.clone(), val)));
        let grad_ref = Rc::from(RefCell::new(Array::zeros(dim)));

        let leaf_node = Node {
            fw: Box::from(LeafForward {
                value: arr_ref.clone()
            }),
            bw: Some(Box::from(LeafBackward {
                grad: grad_ref.clone()
            })),
            operands: vec![]
        };

        Expression {
            tape: &self,
            index: self.insert(leaf_node),
            output: arr_ref.clone(),
            grad: Some(grad_ref.clone())
        }
    }

    // Inserts a node into the tape, and returns the index of that node.
    pub fn insert(&self, node: Node) -> usize {
        let mut list = self.nodes.borrow_mut();
        let len = list.len();
        list.push(node);
        len
    }

    pub fn forward(&self) {
        let mut nodes = self.nodes.borrow_mut();
        for node in nodes.iter() {
            node.forward();
        }
    }
}