use std::ops::{Sub, Div};
use std::process::Output;
use std::{cell::RefCell, ops::Add};
use std::rc::Rc;

use num_traits::{Zero,One};
use ndarray::linalg::{general_mat_vec_mul, Dot};
use ndarray::{prelude::*, Zip, LinalgScalar};
use super::{ForwardOp, BackwardOp};

pub struct MatVecMulForward<T> {
    pub(crate) matrix: Rc<RefCell<Array2<T>>>,
    pub(crate) vector: Rc<RefCell<Array1<T>>>,
    pub(crate) output: Rc<RefCell<Array1<T>>>
}

impl <T> MatVecMulForward<T> 
where
    T: LinalgScalar
{
    pub fn new(mat_ref: &Rc<RefCell<Array2<T>>>,
                vec_ref: &Rc<RefCell<Array1<T>>>) -> Self
    {
        let mat = mat_ref.borrow();
        let vec = vec_ref.borrow();
        
        // let x = Ix2
        let new = Rc::from(RefCell::new((mat).dot(&*vec)));

        MatVecMulForward {
            matrix: mat_ref.clone(),
            vector: vec_ref.clone(),
            output: new.clone()
        }
    }
}

impl <T> ForwardOp for MatVecMulForward<T>
where
    T: Zero + One + Copy + Sub<Output = T> + Div<Output = T> + 'static
{
    fn forward(&self) {
        let output_ref = &mut * self.output.borrow_mut();
        let matrix_ref = & * self.matrix.borrow();
        let vector_ref = & * self.vector.borrow();
        general_mat_vec_mul(T::one(), matrix_ref, vector_ref, T::zero(), output_ref);
    }
}