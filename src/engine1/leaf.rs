use super::{ Operation, OperandIter, ProtoOperation, graph_ops::null_op };

#[derive(Debug)]
pub struct Leaf {
    prot_op: ProtoOperation<0>
}

impl Leaf {
    #[inline]
    pub fn new() -> Self {
        Leaf {
            prot_op: null_op()
        }
    }
}

impl Operation for Leaf {
    fn iter(&self) -> OperandIter<'_> {
        let operands = &self.prot_op.children;
        OperandIter { operands, op_num: 0 }
    }
}