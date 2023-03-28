use std::fmt::Debug;
use std::{ops::{Add, Sub, Div}, cell::RefCell, rc::Rc, borrow::BorrowMut};

use ndarray::linalg::Dot;
use ndarray::prelude::*;
use num_traits::{Zero, One};
use super::operations::matvecmul::MatVecMulForward;
use super::tape::{Node, Tape};
use super::operations::addition::AdditionForward;

// use super::tape::Tape;

// pub struct Var<'t> {
//     tape: &'t Tape,
//     index: usize,
//     value: ArrayD<f32>,
// }

pub struct Expression<'t, T, D> {
    pub(crate) tape: &'t Tape,
    pub(crate) index: usize,
    pub(crate) output: Rc<RefCell<Array<T, D>>>
}

impl <'t, T, D> Add<&Expression<'t, T, D>> for &Expression<'t, T, D>
where
    T: Add<Output = T> + Clone + Copy + Debug + 'static,
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
            
        }

        // println!("arrs: {:?}, {:?}", lhs_arr, rhs_arr);

        let sum_ref = Rc::from(RefCell::new(&*lhs_arr + &*rhs_arr));
        let new_node = Node {
            fw: Box::from(AdditionForward {
                operands: [self.output.clone(), rhs.output.clone()],
                output: sum_ref.clone()
            }),
            bw: None
        };

        new_node.forward();

        Self::Output {
            tape: self.tape,
            index: self.tape.insert(new_node),
            output: sum_ref.clone()
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

        let sum_ref = Rc::from(RefCell::new(Array1::zeros((vec_size))));
        let new_node = Node {
            fw: Box::from(MatVecMulForward {
                matrix: self.output.clone(),
                vector: rhs.output.clone(),
                output: sum_ref.clone()
            }),
            bw: None
        };

        Expression {
            tape: self.tape,
            index: self.tape.insert(new_node),
            output: sum_ref.clone()
        }   
    }
}