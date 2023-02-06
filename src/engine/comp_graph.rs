/// SUPPORT FOR COMPUTATIONAL GRAPHS
/// So far: Only have support for the graph structure itself
/// Main idea: Dissociate the graph's structure from any computational structures.
/// This is so we may have GPU support down the line, where we keep data in VRAM.

use super::tensor::TensorView;

use ndarray::prelude::*;

#[derive(Debug)]
pub struct Graph{
    nodes: Vec<Variable>,
    next_id: usize
}

#[derive(Debug)]
pub struct Variable {
    tensor: TensorView,

    id: usize,
    op: Operation,
    forward_refs: Vec<usize>,

    requires_grad: bool,
}

#[derive(Debug)]
pub enum Operation {
    Leaf,
    Sigmoid(usize),
    Tanh(usize),
    ReLU(usize),
    Addition(usize, usize),
    Subtraction(usize, usize),
    MatMatMul(usize, usize),
    MatVecMul(usize, usize),
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            nodes: vec![],
            next_id: 0
        }
    }

    pub fn zeros(&mut self, dim: &[usize], requires_grad: bool) -> Option<usize> {
        let var_id = self.next_id;

        let var = Variable {
            tensor: TensorView::from_dimension(dim),
            id: var_id,
            op: Operation::Leaf,
            forward_refs: vec![],
            requires_grad
        };

        self.next_id += 1;
        self.nodes.push(var);

        Some(var_id)
    }

    pub fn add(&mut self, x_id: usize, y_id: usize) -> Option<usize> {
        let x = &self.nodes[x_id];
        let y = &self.nodes[y_id];

        /* First, check to see if the add operation would even work. */
        if !x.tensor.shares_dim(&y.tensor) {
            None
        } else {
            let var_id = self.next_id;

            let var = Variable{
                tensor: TensorView::from_dimension(&x.tensor.sizes),
                id: self.next_id,
                op: Operation::Addition(x_id, y_id),
                forward_refs: vec![],
                requires_grad: x.requires_grad || y.requires_grad
            };

            self.next_id += 1;
            
            self.nodes[x_id].forward_refs.push(var.id);
            self.nodes[y_id].forward_refs.push(var.id);

            self.nodes.push(var);

            Some(var_id)
        }
    }

    pub fn sub(&mut self, x_id: usize, y_id: usize) -> Option<usize> {
        let x = &self.nodes[x_id];
        let y = &self.nodes[y_id];

        if !x.tensor.shares_dim(&y.tensor) {
            None
        } else {
            let var_id = self.next_id;

            let var = Variable {
                tensor: TensorView::from_dimension(&x.tensor.sizes),
                id: self.next_id,
                op: Operation::Subtraction(x_id, y_id),
                forward_refs: vec![],
                requires_grad: x.requires_grad || y.requires_grad,
            };

            self.next_id += 1;

            self.nodes[x_id].forward_refs.push(var_id);
            self.nodes[y_id].forward_refs.push(var_id);

            self.nodes.push(var);

            Some(var_id)
        }
    }

    pub fn mm(&mut self, x_id: usize, y_id: usize) -> Option<usize> {
        let x = &self.nodes[x_id];
        let y = &self.nodes[y_id];

        if !x.tensor.can_mm(&y.tensor) {
            None
        } else {
            let var_id = self.next_id;

            let var = Variable {
                tensor: TensorView::from_dimension(&[x.tensor.sizes[0], y.tensor.sizes[1]]),
                id: var_id,
                op: Operation::MatMatMul(x_id, y_id),
                forward_refs: vec![],
                requires_grad: x.requires_grad || y.requires_grad,
            };

            self.next_id += 1;

            self.nodes[x_id].forward_refs.push(var_id);
            self.nodes[y_id].forward_refs.push(var_id);

            self.nodes.push(var);

            Some(var_id)
        }
    }

    pub fn mv(&mut self, x_id: usize, y_id: usize) -> Option<usize> {
        let x = &self.nodes[x_id];
        let y = &self.nodes[y_id];

        if !x.tensor.can_mv(&y.tensor) {
            None
        } else {
            let var_id = self.next_id;

            let var = Variable {
                tensor: TensorView::from_dimension(&[x.tensor.sizes[0]]),
                id: var_id,
                op: Operation::MatVecMul(x_id, y_id),
                forward_refs: vec![],
                requires_grad: x.requires_grad || y.requires_grad,
            };

            self.next_id += 1;

            self.nodes[x_id].forward_refs.push(var_id);
            self.nodes[y_id].forward_refs.push(var_id);

            self.nodes.push(var);

            Some(var_id)
        }
    }

    pub fn tanh(&mut self, x_id: usize) -> Option<usize> {
        let x = &self.nodes[x_id];

        let var_id = self.next_id;
        let var = Variable {
            tensor: TensorView::from_dimension(&x.tensor.sizes),
            id: var_id,
            op: Operation::Tanh(x_id),
            forward_refs: vec![],
            requires_grad: x.requires_grad,
        };

        self.next_id += 1;
        self.nodes[x_id].forward_refs.push(var_id);
        self.nodes.push(var);

        Some(var_id)
    }

    pub fn relu(&mut self, x_id: usize) -> Option<usize> {
        let x = &self.nodes[x_id];

        let var_id = self.next_id;
        let var = Variable {
            tensor: TensorView::from_dimension(&x.tensor.sizes),
            id: var_id,
            op: Operation::ReLU(x_id),
            forward_refs: vec![],
            requires_grad: x.requires_grad,
        };

        self.next_id += 1;
        self.nodes[x_id].forward_refs.push(var_id);
        self.nodes.push(var);

        Some(var_id)
    }

    pub fn sigmoid(&mut self, x_id: usize) -> Option<usize> {
        let x = &self.nodes[x_id];

        let var_id = self.next_id;
        let var = Variable {
            tensor: TensorView::from_dimension(&x.tensor.sizes),
            id: var_id,
            op: Operation::Sigmoid(x_id),
            forward_refs: vec![],
            requires_grad: x.requires_grad,
        };

        self.next_id += 1;
        self.nodes[x_id].forward_refs.push(var_id);
        self.nodes.push(var);

        Some(var_id)
    }
}