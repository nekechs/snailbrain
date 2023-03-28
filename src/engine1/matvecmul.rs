use super::{ Operation, OperandIter, ProtoOperation, graph_ops::bi_op };

#[derive(Debug)]
pub struct MatVecMul {
    prot_op: ProtoOperation<2>
}

impl MatVecMul {
    #[inline]
    pub fn new(x_id: usize, y_id: usize) -> Self {
        MatVecMul {
            prot_op: bi_op(x_id, y_id)
        }
    }
}

impl Operation for MatVecMul {
    fn iter(&self) -> OperandIter<'_> {
        let operands = &self.prot_op.children;
        OperandIter { operands, op_num: 0 }
    }
}