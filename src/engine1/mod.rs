pub mod comp_graph;
pub mod device;
pub mod tensor;
pub mod graph_ops;

pub mod leaf;
pub mod sigmoid;
pub mod relu;
pub mod tanh;
pub mod addition;
pub mod subtraction;
pub mod matmatmul;
pub mod matvecmul;

use graph_ops::{Operation, OperandIter, ProtoOperation};

use comp_graph::{Variable, Graph};