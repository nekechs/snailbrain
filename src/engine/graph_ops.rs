use std::fmt::Debug;

pub trait Operation: Debug{
    fn iter(&self) -> OperandIter<'_>;
}

/* We use this for storing children in an operation. Used by all the operation structs. */
#[derive(Debug)]
pub(crate) struct ProtoOperation<const N: usize> {
    pub(crate) children: [usize; N]
}

#[inline]
pub(crate) fn null_op() -> ProtoOperation<0> {
    ProtoOperation {
        children: []
    }
}

#[inline]
pub(crate) fn mono_op(x_id: usize) -> ProtoOperation<1> {
    ProtoOperation { children: [x_id] }
}

#[inline]
pub(crate) fn bi_op(x_id: usize, y_id: usize) -> ProtoOperation<2> {
    ProtoOperation { children: [x_id, y_id] }
}

impl <const N: usize> ProtoOperation<N> {
    
}

/* Iterates over all the operands of an operation */
pub struct OperandIter<'a> {
    // TODO: Finish the iterator maintaier for this.
    pub(crate) operands: &'a [usize],
    pub(crate) op_num: usize,
}

impl <'a> Iterator for OperandIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let mut ret = None;
        if self.op_num < self.operands.len() {
            ret = Some(self.operands[self.op_num]);
        }

        self.op_num += 1;

        ret

        // todo!();
    }
}