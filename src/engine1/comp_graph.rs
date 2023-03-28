use std::collections::BTreeSet;

use super::addition::Addition;
use super::leaf::Leaf;
use super::matmatmul::MatMatMul;
use super::matvecmul::MatVecMul;
use super::relu::ReLU;
use super::sigmoid::Sigmoid;
use super::subtraction::Subtraction;
use super::tanh::Tanh;
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
    // tensor2: ArrayBase<

    id: usize,
    op: Box<dyn Operation>,
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

    #[inline]
    fn new(tensor: TensorView, id: usize, operation: Box<dyn Operation>, requires_grad: bool) -> Self {
        let grad = if requires_grad {Some(tensor.gen_from_same())} else {None};

        Variable {
            tensor: tensor,
            id,
            op: Box::from(operation),
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

    /* Given that the inputs have propagated forward to all the nodes, what is the gradient of this node's value WRT all other nodes? */
    /* Updates the gradient field of every single node to reflect this. */
    pub fn backward(&mut self, var_id: usize) {
        // TODO: FINISH
        unimplemented!();
    }

    /* Performs topological sort on a node and all of the nodes that stem from the child references */
    pub fn topsort_backward(&self, var_id: usize) -> Vec<usize> {
        let mut visited_nodes= BTreeSet::new();
        let mut sorted_list = vec![];
        
        self.topsort_recursive(var_id, &mut visited_nodes, &mut sorted_list);

        sorted_list
    }

    fn topsort_recursive(&self, node_id: usize, visited_nodes: &mut BTreeSet<usize>, sorted_list: &mut Vec<usize>) {
        visited_nodes.insert(node_id);
        let node = &self.nodes[node_id];

        for operand_id in node.op.iter() {
            if !visited_nodes.contains(&operand_id) {
                self.topsort_recursive(operand_id, visited_nodes, sorted_list);
            }
        }
        sorted_list.push(node_id);
        
    }

    // pub fn from_fn(&mut self, dim: &[usize]) -> Option<usize> {

    // }

    /* Generates a node in the graph that essentially represents a variable that starts out @ zero. */
    pub fn zeros(&mut self, dim: &[usize], requires_grad: bool) -> Option<usize> {
        let var_id = self.next_id;
        let var = Variable::new(
            TensorView::zeros_from_dimension(dim),
            var_id,
            Box::from(Leaf::new()),
            requires_grad,
        );
        
        self.next_id += 1;
        self.nodes.push(var);

        Some(var_id)
    }

    /* Represents the sum of two graph variables. Must have the same dimension. */
    pub fn add(&mut self, x_id: usize, y_id: usize) -> Option<usize> {
        let x = &self.nodes[x_id];
        let y = &self.nodes[y_id];

        /* First, check to see if the add operation would even work. */
        if !x.tensor.shares_dim(&y.tensor) {
            None
        } else {
            let var_id = self.next_id;
            let var = Variable::new(
                TensorView::zeros_from_dimension(&x.tensor.sizes),
                self.next_id,
                Box::from(Addition::new(x_id, y_id)),
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
                TensorView::zeros_from_dimension(&x.tensor.sizes),
                self.next_id,
                Box::from(Subtraction::new(x_id, y_id)),
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
                TensorView::zeros_from_dimension(&new_dim),
                var_id,
                Box::from(MatMatMul::new(x_id, y_id)),
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
                TensorView::zeros_from_dimension(&new_dim),
                var_id,
                Box::from(MatVecMul::new(x_id, y_id)),
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
            TensorView::zeros_from_dimension(&x.tensor.sizes),
            var_id,
            Box::from(Tanh::new(x_id)),
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
            TensorView::zeros_from_dimension(&x.tensor.sizes),
            var_id,
            Box::from(ReLU::new(x_id)),
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
            TensorView::zeros_from_dimension(&x.tensor.sizes),
            var_id,
            Box::from(Sigmoid::new(x_id)),
            x.requires_grad()
        );

        self.next_id += 1;
        self.nodes[x_id].forward_refs.push(var_id);
        self.nodes.push(var);

        Some(var_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ex_iterator() {
        let mut comp_graph = Graph::new();
        let a = comp_graph.zeros(&[2, 3], true).unwrap();
        let b = comp_graph.zeros(&[2, 3], false).unwrap();

        let c = comp_graph.add(a, b).unwrap();
        let d = comp_graph.sub(a, b).unwrap();

        let c_children: Vec<_> = comp_graph.nodes[c].op.iter().collect();
        let d_children: Vec<_> = comp_graph.nodes[d].op.iter().collect();
        assert_eq!(vec![0, 1], c_children);
        assert_eq!(vec![0, 1], d_children);
        // println!("{:?}", vector);
    }
}