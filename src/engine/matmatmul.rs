use super::{ Operation, OperandIter, ProtoOperation, graph_ops::bi_op };

#[derive(Debug)]
pub struct MatMatMul {
    prot_op: ProtoOperation<2>
}

impl MatMatMul {
    #[inline]
    pub fn new(x_id: usize, y_id: usize) -> Self {
        MatMatMul {
            prot_op: bi_op(x_id, y_id)
        }
    }
}

impl Operation for MatMatMul {
    fn iter(&self) -> OperandIter<'_> {
        let operands = &self.prot_op.children;
        OperandIter { operands, op_num: 2 }
    }
}