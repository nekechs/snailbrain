use super::{ Operation, OperandIter, ProtoOperation, graph_ops::mono_op };

#[derive(Debug)]
pub struct ReLU {
    prot_op: ProtoOperation<1>
}

impl ReLU {
    #[inline]
    pub fn new(x_id: usize) -> Self {
        ReLU {
            prot_op: mono_op(x_id)
        }
    }
}

impl Operation for ReLU {
    fn iter(&self) -> OperandIter<'_> {
        let operands = &self.prot_op.children;
        OperandIter { operands, op_num: 1 }
    }
}