use std::ops::{Sub, Div};
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

pub struct MatVecMulBackward<T> {
    pub(crate) matrix_out: Rc<RefCell<Array2<T>>>,
    pub(crate) vector_out: Rc<RefCell<Array1<T>>>,
    
    pub(crate) output_grad: Rc<RefCell<Array1<T>>>,
    pub(crate) matrix_grad: Option<Rc<RefCell<Array2<T>>>>,
    pub(crate) vector_grad: Option<Rc<RefCell<Array1<T>>>>
}

impl <T> BackwardOp for MatVecMulBackward<T> 
where
    T: Zero + One + Copy + Sub<Output = T> + Div<Output = T> + 'static
{
    fn backward(&self) {
        let outgrad_ref = &*self.output_grad.borrow();
        if let Some(matrix_grad) = &self.matrix_grad {
            let matgrad_ref = &mut * matrix_grad.borrow_mut();
            let vecout_ref = &*self.vector_out.borrow();

            let grad_len = outgrad_ref.dim();

            let vecout_t = vecout_ref.view().broadcast((grad_len, grad_len)).unwrap().t();
            let outgrad_1dview = outgrad_ref.view();
            let outgrad_2d = outgrad_1dview.broadcast((grad_len, grad_len)).unwrap();

            // matgrad_ref.assign((outgrad_ref).into_dimensionality::<Ix2>().unwrap().dot(&transposed_view));
            matgrad_ref.assign(&outgrad_2d.dot(vecout_ref));
        }

        if let Some(vector_grad) = &self.vector_grad {
            let vecgrad_ref = &mut *vector_grad.borrow_mut();
            let matout_ref = & *self.matrix_out.borrow();
            general_mat_vec_mul(T::one(), matout_ref, outgrad_ref, T::zero(), vecgrad_ref);
        }
    }
}