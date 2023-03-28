use super::{ Operation, OperandIter, ProtoOperation, graph_ops::bi_op };

#[derive(Debug)]
pub struct Subtraction {
    prot_op: ProtoOperation<2>
}

impl Subtraction {
    #[inline]
    pub fn new(x_id: usize, y_id: usize) -> Self {
        Subtraction {
            prot_op: bi_op(x_id, y_id)
        }
    }
}

impl Operation for Subtraction {
    fn iter(&self) -> OperandIter<'_> {
        let operands = &self.prot_op.children;
        OperandIter { operands, op_num: 0 }
    }
}