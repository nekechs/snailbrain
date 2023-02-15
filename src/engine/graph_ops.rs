use super::tensor::Datatype;

// pub enum Operation {
//     Leaf,
//     Sigmoid(usize),
//     Tanh(usize),
//     ReLU(usize),
//     Addition(usize, usize),
//     Subtraction(usize, usize),
//     MatMatMul(usize, usize),
//     MatVecMul(usize, usize),
// }

#[derive(Debug)]
pub enum Operation {
    Leaf(Leaf),                 // leaf
    Sigmoid(Sigmoid),           // sigmoid
    Tanh(Tanh),                 // tanh
    ReLU(ReLU),                 // relu
    Addition(Addition),         // add
    Subtraction(Subtraction),   // sub
    MatMatMul(MatMatMul),       // mm
    MatVecMul(MatVecMul)        // mv
}

impl Operation {
    #[inline]
    pub fn leaf() -> Self {
        Operation::Leaf(Leaf{})
    }

    #[inline]
    pub fn sigmoid(x_id: usize) -> Self {
        Operation::Sigmoid(Sigmoid {x_id})
    }

    #[inline]
    pub fn tanh(x_id: usize) -> Self {
        Operation::Tanh(Tanh {x_id})
    }

    #[inline]
    pub fn relu(x_id: usize) -> Self {
        Operation::ReLU(ReLU {x_id})
    }

    #[inline]
    pub fn add(x_id: usize, y_id: usize) -> Self {
        Operation::Addition(Addition { x_id, y_id })
    }

    #[inline]
    pub fn sub(x_id: usize, y_id: usize) -> Self {
        Operation::Subtraction(Subtraction { x_id, y_id })
    }

    #[inline]
    pub fn mm(x_id: usize, y_id: usize) -> Self {
        Operation::MatMatMul(MatMatMul { x_id, y_id })
    }

    #[inline]
    pub fn mv(x_id: usize, y_id: usize) -> Self {
        Operation::MatVecMul(MatVecMul { x_id, y_id })
    }
}

#[derive(Debug)]
struct Leaf {

}

#[derive(Debug)]
struct Sigmoid {
    x_id: usize
}

#[derive(Debug)]
struct Tanh {
    x_id: usize
}

#[derive(Debug)]
struct ReLU {
    x_id: usize
}

#[derive(Debug)]
struct Addition {
    x_id: usize,
    y_id: usize
}

#[derive(Debug)]
struct Subtraction {
    x_id: usize,
    y_id: usize
}

#[derive(Debug)]
struct MatMatMul {
    x_id: usize,
    y_id: usize
}

#[derive(Debug)]
struct MatVecMul {
    x_id: usize,
    y_id: usize
}

impl Operation {
    pub fn iter(&self) -> OperandIter<'_> {
        OperandIter{op: &self, op_num: 0}
    }

    // pub fn dtype_res(&self, )
}

pub struct OperandIter<'a> {
    // TODO: Finish the iterator maintaier for this.
    op: &'a Operation,
    op_num: usize,
}

impl <'a> Iterator for OperandIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let ret = match self.op {
            Operation::Leaf(_) => None,
            Operation::Sigmoid(oper_impl) => {
                match self.op_num {
                    0 => Some(oper_impl.x_id),
                    _ => None
                }
            },
            Operation::Tanh(oper_impl) => {
                match self.op_num {
                    0 => Some(oper_impl.x_id),
                    _ => None
                }
            },
            Operation::ReLU(oper_impl) => {
                match self.op_num {
                    0 => Some(oper_impl.x_id),
                    _ => None
                }
            },
            Operation::Addition(oper_impl) => {
                match self.op_num {
                    0 => Some(oper_impl.x_id),
                    1 => Some(oper_impl.y_id),
                    _ => None
                }
            } 
            Operation::Subtraction(oper_impl) => {
                match self.op_num {
                    0 => Some(oper_impl.x_id),
                    1 => Some(oper_impl.y_id),
                    _ => None
                }
            },
            Operation::MatMatMul(oper_impl) => {
                match self.op_num {
                    0 => Some(oper_impl.x_id),
                    1 => Some(oper_impl.y_id),
                    _ => None
                }
            },
            Operation::MatVecMul(oper_impl) => {
                match self.op_num {
                    0 => Some(oper_impl.x_id),
                    1 => Some(oper_impl.y_id),
                    _ => None
                }
            }
        };

        self.op_num += 1;

        ret

        // todo!();
    }
}