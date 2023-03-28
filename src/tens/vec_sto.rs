use std::fmt::Debug;

use super::types::*;

#[derive(Debug)]
pub struct VecStorage {
    dtype: DType,
    sizes: Vec<usize>,
    
}
