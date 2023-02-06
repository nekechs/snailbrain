
#[derive(Debug)]
pub struct TensorView {
    pub(crate) sizes: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) offset: usize,

    // storage: TensorStorage
}

impl TensorView {
    #[inline]
    pub fn shares_dim(&self, other: &TensorView) -> bool {
        self.sizes == other.sizes
    }

    #[inline]
    pub fn can_mm(&self, other: &TensorView) -> bool {
        self.sizes.len() == 2 && other.sizes.len() == 2 && self.sizes[1] == other.sizes[0]
    }

    #[inline]
    pub fn can_mv(&self, other: &TensorView) -> bool {
        self.sizes.len() == 2 && other.sizes.len() == 1 && self.sizes[1] == other.sizes[0]
    }

    #[inline]
    pub fn from_dimension(dim: &[usize]) -> Self {
        let sizes = dim.to_vec();
        
        let mut strides = vec![0; dim.len()];
        let mut acc = 1;

        for (ind, value) in dim.iter().map(|num| *num).enumerate().rev() {
            strides[ind] = acc;
            acc *= value;
        }

        TensorView { sizes, strides, offset: 0 }
    }

    pub fn gen_from_same(&self) -> Self {
        Self::from_dimension(&self.sizes)
    }
}

pub(crate) struct TensorStorage {
    sizes: Vec<usize>

}