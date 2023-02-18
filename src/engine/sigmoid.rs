use super::{ Operation, OperandIter, ProtoOperation, graph_ops::mono_op };

#[derive(Debug)]
pub struct Sigmoid {
    prot_op: ProtoOperation<1>
}

impl Sigmoid {
    #[inline]
    pub fn new(x_id: usize) -> Self {
        Sigmoid {
            prot_op: mono_op(x_id)
        }
    }
}

impl Operation for Sigmoid {
    fn iter(&self) -> OperandIter<'_> {
        let operands = &self.prot_op.children;
        OperandIter { operands, op_num: 1 }
    }
}