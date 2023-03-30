use std::{cell::RefCell, ops::Add};
use std::rc::Rc;

use num_traits::Zero;
use ndarray::{prelude::*, Zip};

use super::{ForwardOp, BackwardOp};

pub struct LeafForward<T, D>{
    pub(crate) value: Rc<RefCell<Array<T, D>>>
}

impl <T, D> LeafForward<T, D>
where
    D: Dimension,
    T: Clone + Zero
{
    pub fn zeros<Sh: Dimension>(shape: D) -> (Self, Rc<RefCell<Array<T, D>>>){
        let arr: Array<T, D> = ArrayBase::zeros(shape);
        let arr_ref = Rc::from(RefCell::new(arr));
        (
            LeafForward {
                value: arr_ref.clone(),
            },
            arr_ref
        )
    }
}

impl <T, D> ForwardOp for LeafForward<T, D> {
    fn forward(&self) {
        // Nothing. We don't want anything here.
    }
}

pub struct LeafBackward<T, D> {
    pub(crate) grad: Rc<RefCell<Array<T, D>>>
}

impl <T, D> BackwardOp for LeafBackward<T, D> {
    fn backward(&self) {
        // Again, no operation required.
    }
}