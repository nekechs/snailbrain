
use std::{cell::RefCell, rc::Rc};
use ndarray::prelude::*;
use num_traits::Zero;

use super::{operations::{BackwardOp, ForwardOp, leaf::LeafForward}, expr::Expression};

pub struct Node {
    pub(crate) fw: Box<dyn ForwardOp>,
    pub(crate) bw: Option<Box<dyn BackwardOp>>
}

impl Node {
    pub fn forward(&self) {
        self.fw.forward();
    }
}

pub struct Tape {
    nodes: RefCell<Vec<Node>>
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
            bw: None
        };

        Expression {
            tape: &self,
            index: self.insert(leaf_node),
            output: arr_ref.clone()
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