use std::default;

#[derive(Debug)]
pub(crate) enum DType {
    re_i32,     // 4 bytes
    re_i64,     // 8 bytes
    re_f32,     // 4 bytes
    re_f64,     // 8 bytes

    cm_i32,     // 8 bytes
    cm_f32,     // 8 bytes
}

impl DType {
    // This is the number of bytes that we need. Wow.
    #[inline]
    pub(crate) fn num_bytes(&self) -> usize {
        match self {
            DType::re_i32 |
            DType::re_f32 => 4,
            DType::re_i64 |
            DType::re_f64 |
            DType::cm_i32 |
            DType::cm_f32 => 8
        }
    }
}