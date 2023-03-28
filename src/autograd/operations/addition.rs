use std::fmt::Debug;
use std::{cell::RefCell, ops::Add};
use std::rc::Rc;

use ndarray::{prelude::*, Zip};
use super::{ForwardOp, BackwardOp};

pub struct AdditionForward<T, D> {
    pub(crate) operands: [Rc<RefCell<Array<T, D>>>; 2],
    pub(crate) output: Rc<RefCell<Array<T, D>>>
}

impl <T, D> ForwardOp for AdditionForward<T, D>
where
    D: Dimension,
    T: Copy + Add<Output = T> + Debug
{
    fn forward(&self) {
        let x = &*self.operands[0].borrow();
        let y = &*self.operands[1].borrow();

        // println!("Adding {:?} and {:?}", x, y);

        let sum_ref =  &mut *self.output.borrow_mut();
        // azip!((sum_ref in &mut sum_ref, x in &x, y in y) sum_ref = x + y);
        Zip::from(sum_ref)
            .and(x)
            .and(y)
            .for_each(|out, &x, &y| *out = x + y);
    }
}

