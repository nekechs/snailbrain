use std::fmt::Debug;
use std::{ops::{Add, Sub, Div}, cell::RefCell, rc::Rc, borrow::BorrowMut};

use ndarray::linalg::Dot;
use ndarray::prelude::*;
use num_traits::{Zero, One};
use super::operations::BackwardOp;
use super::operations::matvecmul::{MatVecMulForward, MatVecMulBackward};
use super::tape::{Node, Tape};
use super::operations::addition::{AdditionForward, AdditionBackward};

// use super::tape::Tape;

// pub struct Var<'t> {
//     tape: &'t Tape,
//     index: usize,
//     value: ArrayD<f32>,
// }

pub struct Expression<'t, T, D> {
    pub(crate) tape: &'t Tape,
    pub(crate) index: usize,
    pub(crate) output: Rc<RefCell<Array<T, D>>>,
    pub(crate) grad: Option<Rc<RefCell<Array<T, D>>>>
}

impl <'t, T, D> Expression<'t, T, D> {
    #[inline]
    pub fn grad_exists(&self) -> bool {
        self.grad.is_some()
    }
}

impl <'t, T, D> Add<&Expression<'t, T, D>> for &Expression<'t, T, D>
where
    T: Zero + Add<Output = T> + Clone + Copy + Debug + 'static,
    D: Dimension + 'static
{
    type Output = Expression<'t, T, D>;
    // TODO: Change this to &self instead of self
    fn add(self, rhs: &Expression<'t, T, D>) -> Self::Output
    {
        let lhs_arr = self.output.borrow();
        let rhs_arr = rhs.output.borrow();

        if lhs_arr.dim() != rhs_arr.dim() {
            // Handle the case when the dimensions are incompatible --- whenever the dimensions
            // do not match.
            panic!("Dimension incompatibility for pointwise addition.");
        }

        let mut grad_ref = None;
        let bw: Option<Box<dyn BackwardOp + 'static>> = if self.grad_exists() || rhs.grad_exists() {
            let grad = Rc::from(RefCell::new(Array::zeros(lhs_arr.dim())));
            grad_ref = Some(grad.clone());
            Some(
                Box::from(
                    AdditionBackward {
                        operands_grad: [
                            self.grad.clone(),
                            rhs.grad.clone()
                        ],
                        output_grad: grad.clone()
                    }
                )
            )
        } else {
            None
        };

        let sum_ref = Rc::from(RefCell::new(&*lhs_arr + &*rhs_arr));
        let new_node = Node {
            fw: Box::from(AdditionForward {
                operands: [self.output.clone(), rhs.output.clone()],
                output: sum_ref.clone()
            }),
            bw
        };

        new_node.forward();

        Self::Output {
            tape: self.tape,
            index: self.tape.insert(new_node),
            output: sum_ref.clone(),
            grad: grad_ref
        }
    }
}

impl <'t, T> Expression<'t, T, Ix2> 
where
    T: Zero + One + Copy + Sub<Output = T> + Div<Output = T> + 'static
{
    pub fn mv(&self, rhs: &Expression<'t, T, Ix1>) -> Expression<'t, T, Ix1>{
        let mat_ref = self.output.borrow();
        let vec_ref = rhs.output.borrow();

        let (mat_rows, mat_cols) = mat_ref.dim();
        let vec_size = vec_ref.dim();

        if mat_cols != vec_size {
            // Matrix vector product is not valid, because dimensions are not valid for doing so.
            panic!("Dimension incompatibility for matrix vector multiplication.");
        }

        let mut grad_ref = None;
        let bw: Option<Box<dyn BackwardOp + 'static>> = if self.grad_exists() || rhs.grad_exists() {
            let grad = Rc::from(RefCell::new(Array::zeros(mat_rows)));
            grad_ref = Some(grad.clone());
            Some(
                Box::from(
                    MatVecMulBackward {
                        matrix_out: self.output.clone(),
                        vector_out: rhs.output.clone(),

                        output_grad: grad.clone(),
                        matrix_grad: self.grad.clone(),
                        vector_grad: rhs.grad.clone()
                    }
                )
            )
        } else {
            None
        };

        let output_ref = Rc::from(RefCell::new(Array1::zeros(mat_rows)));
        let new_node = Node {
            fw: Box::from(MatVecMulForward {
                matrix: self.output.clone(),
                vector: rhs.output.clone(),
                output: output_ref.clone()
            }),
            bw
        };

        Expression {
            tape: self.tape,
            index: self.tape.insert(new_node),
            output: output_ref.clone(),
            grad: grad_ref
        }   
    }
}