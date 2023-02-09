/// SUPPORT FOR COMPUTATIONAL GRAPHS
/// So far: Only have support for the graph structure itself
/// Main idea: Dissociate the graph's structure from any computational structures.
/// This is so we may have GPU support down the line, where we keep data in VRAM.

use super::tensor::{TensorView, self};
use super::graph_ops::Operation;

use ndarray::{prelude::*, FoldWhile};

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

    grad: Option<TensorView>,
    backward_topo: Option<Vec<usize>>,
}

impl Variable {
    #[inline]
    pub fn requires_grad(&self) -> bool {
        match self.grad {
            Some(_) => true,
            _ => false
        }
    }

    fn new(tensor: TensorView, id: usize, op: Operation, requires_grad: bool) -> Self {
        let grad = if requires_grad {Some(tensor.gen_from_same())} else {None};

        Variable {
            tensor: tensor,
            id,
            op,
            forward_refs: vec![],
            grad,
            backward_topo: None,
        }
    }
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            nodes: vec![],
            next_id: 0
        }
    }

    /* Returns a topologically sorted list of node IDs from the perspective of the given node. */
    pub fn topsort_backward(&self, node_id: usize) -> Vec<usize> {
        todo!();
    }

    fn topsort_recursive(&self, node_id: usize, )

    pub fn zeros(&mut self, dim: &[usize], requires_grad: bool) -> Option<usize> {
        let var_id = self.next_id;
        let var = Variable::new(
            TensorView::from_dimension(dim),
            var_id,
            Operation::Leaf,
            requires_grad,
        );
        
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
            let var = Variable::new(
                TensorView::from_dimension(&x.tensor.sizes),
                self.next_id,
                Operation::Addition(x_id, y_id),
                x.requires_grad() || y.requires_grad()
            );

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
            let var = Variable::new(
                TensorView::from_dimension(&x.tensor.sizes),
                self.next_id,
                Operation::Subtraction(x_id, y_id),
                x.requires_grad() || y.requires_grad()
            );

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
            let new_dim = [x.tensor.sizes[0], y.tensor.sizes[1]];
            let var = Variable::new(
                TensorView::from_dimension(&new_dim),
                var_id,
                Operation::MatMatMul(x_id, y_id),
                x.requires_grad() || y.requires_grad()
            );

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
            let new_dim = [x.tensor.sizes[0], y.tensor.sizes[1]];
            let var = Variable::new(
                TensorView::from_dimension(&new_dim),
                var_id,
                Operation::MatVecMul(x_id, y_id),
                x.requires_grad() || y.requires_grad()
            );

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
        let var = Variable::new(
            TensorView::from_dimension(&x.tensor.sizes),
            var_id,
            Operation::Tanh(x_id),
            x.requires_grad()
        );

        self.next_id += 1;
        self.nodes[x_id].forward_refs.push(var_id);
        self.nodes.push(var);

        Some(var_id)
    }

    pub fn relu(&mut self, x_id: usize) -> Option<usize> {
        let x = &self.nodes[x_id];
        let var_id = self.next_id;
        let var = Variable::new(
            TensorView::from_dimension(&x.tensor.sizes),
            var_id,
            Operation::ReLU(x_id),
            x.requires_grad()
        );

        self.next_id += 1;
        self.nodes[x_id].forward_refs.push(var_id);
        self.nodes.push(var);

        Some(var_id)
    }

    pub fn sigmoid(&mut self, x_id: usize) -> Option<usize> {
        let x = &self.nodes[x_id];
        let var_id = self.next_id;
        let var = Variable::new(
            TensorView::from_dimension(&x.tensor.sizes),
            var_id,
            Operation::Sigmoid(x_id),
            x.requires_grad()
        );

        self.next_id += 1;
        self.nodes[x_id].forward_refs.push(var_id);
        self.nodes.push(var);

        Some(var_id)
    }
}