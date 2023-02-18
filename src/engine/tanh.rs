use super::{ Operation, OperandIter, ProtoOperation, graph_ops::mono_op };

#[derive(Debug)]
pub struct Tanh {
    prot_op: ProtoOperation<1>
}

impl Tanh {
    #[inline]
    pub fn new(x_id: usize) -> Self {
        Tanh {
            prot_op: mono_op(x_id)
        }
    }
}

impl Operation for Tanh {
    fn iter(&self) -> OperandIter<'_> {
        let operands = &self.prot_op.children;
        OperandIter { operands, op_num: 1 }
    }
}