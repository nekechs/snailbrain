pub(crate) mod addition;
pub(crate) mod matvecmul;
pub(crate) mod leaf;

pub trait ForwardOp {
    fn forward(&self);
}

pub trait BackwardOp {
    fn backward(&self);
}