use gl;

#[allow(unused)]
#[derive(Debug, Clone, Copy)]
pub enum GlError {
    InvalidEnum,
    InvalidValue,
    InvalidOperation,
    StackOverflow,
    StackUnderflow,
    OutOfMemory,
    InvalidFramebufferOperation,
}
impl GlError {
    #[allow(unused)]
    pub fn check() -> Result<(), GlError> {
        let err = unsafe { gl::GetError() };
        use self::GlError::*;
        match err {
            gl::NO_ERROR => Ok(()),
            gl::INVALID_ENUM => Err(InvalidEnum),
            gl::INVALID_VALUE => Err(InvalidValue),
            gl::INVALID_OPERATION => Err(InvalidOperation),
            gl::STACK_OVERFLOW => Err(StackOverflow),
            gl::STACK_UNDERFLOW => Err(StackUnderflow),
            gl::OUT_OF_MEMORY => Err(OutOfMemory),
            gl::INVALID_FRAMEBUFFER_OPERATION => Err(InvalidFramebufferOperation),
            x => unimplemented!("unknown glError: {:?}", x),
        }
    }
}

#[allow(unused)]
pub enum GlType {
    UnsignedByte,          // GL_UNSIGNED_BYTE,
    Byte,                  // GL_BYTE,
    UnsignedShort,         // GL_UNSIGNED_SHORT,
    Short,                 // GL_SHORT,
    UnsignedInt,           // GL_UNSIGNED_INT,
    Int,                   // GL_INT,
    Float,                 // GL_FLOAT,
    UnsignedByte332,       // GL_UNSIGNED_BYTE_3_3_2,
    UnsignedByte233Rev,    // GL_UNSIGNED_BYTE_2_3_3_REV,
    UnsignedShort565,      // GL_UNSIGNED_SHORT_5_6_5,
    UnsignedShort565Rev,   // GL_UNSIGNED_SHORT_5_6_5_REV,
    UnsignedShort4444,     // GL_UNSIGNED_SHORT_4_4_4_4,
    UnsignedShort4444Rev,  // GL_UNSIGNED_SHORT_4_4_4_4_REV,
    UnsignedShort5551,     // GL_UNSIGNED_SHORT_5_5_5_1,
    UnsignedShort1555Rev,  // GL_UNSIGNED_SHORT_1_5_5_5_REV,
    UnsignedInt8888,       // GL_UNSIGNED_INT_8_8_8_8,
    UnsignedInt8888Rev,    // GL_UNSIGNED_INT_8_8_8_8_REV,
    UnsignedInt1010102,    // GL_UNSIGNED_INT_10_10_10_2,
    UnsignedInt2101010Rev, // GL_UNSIGNED_INT_2_10_10_10_REV,
}
impl Into<u32> for GlType {
    fn into(self) -> u32 {
        use GlType::*;
        match self {
            UnsignedByte => gl::UNSIGNED_BYTE,
            Byte => gl::BYTE,
            UnsignedShort => gl::UNSIGNED_SHORT,
            Short => gl::SHORT,
            UnsignedInt => gl::UNSIGNED_INT,
            Int => gl::INT,
            Float => gl::FLOAT,
            UnsignedByte332 => gl::UNSIGNED_BYTE_3_3_2,
            UnsignedByte233Rev => gl::UNSIGNED_BYTE_2_3_3_REV,
            UnsignedShort565 => gl::UNSIGNED_SHORT_5_6_5,
            UnsignedShort565Rev => gl::UNSIGNED_SHORT_5_6_5_REV,
            UnsignedShort4444 => gl::UNSIGNED_SHORT_4_4_4_4,
            UnsignedShort4444Rev => gl::UNSIGNED_SHORT_4_4_4_4_REV,
            UnsignedShort5551 => gl::UNSIGNED_SHORT_5_5_5_1,
            UnsignedShort1555Rev => gl::UNSIGNED_SHORT_1_5_5_5_REV,
            UnsignedInt8888 => gl::UNSIGNED_INT_8_8_8_8,
            UnsignedInt8888Rev => gl::UNSIGNED_INT_8_8_8_8_REV,
            UnsignedInt1010102 => gl::UNSIGNED_INT_10_10_10_2,
            UnsignedInt2101010Rev => gl::UNSIGNED_INT_2_10_10_10_REV,
        }
    }
}
