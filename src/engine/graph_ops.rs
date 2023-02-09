
#[derive(Debug)]
pub enum Operation {
    Leaf,
    Sigmoid(usize),
    Tanh(usize),
    ReLU(usize),
    Addition(usize, usize),
    Subtraction(usize, usize),
    MatMatMul(usize, usize),
    MatVecMul(usize, usize),
}

impl Operation {
    pub fn iter(&self) -> OperandIter<'_> {
        OperandIter{op: &self, op_num: 0}
    } 
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
            Leaf => None,
            Operation::Sigmoid(x) |
            Operation::Tanh(x) |
            Operation::ReLU(x) => {
                match self.op_num {
                    0 => Some(x),
                    _ => None
                }
            },
            Operation::Addition(x, y) |
            Operation::Subtraction(x, y) |
            Operation::MatMatMul(x, y) |
            Operation::MatVecMul(x, y) => {
                match self.op_num {
                    0 => Some(x),
                    1 => Some(y),
                    _ => None
                }
            }
        };

        todo!();
    }
}