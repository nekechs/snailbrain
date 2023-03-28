
// How does a TensorView work exactly?
// Need to figure out: 

use super::types::*;

#[derive(Debug)]
pub struct TensorView {
    pub(crate) sizes: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) offset: usize,
    pub(crate) dtype: DType,

    // pub(crate) storage_ref: Rc<VecStorage>,

    // storage: TensorStorage
}