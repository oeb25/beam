use gl;
use image;
use std::{borrow::Cow,
          ffi,
          fs,
          marker::PhantomData,
          mem,
          os::{self, raw::c_void},
          path::Path,
          ptr};

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

#[derive(Debug, Clone, Copy)]
pub enum ShaderKind {
    Vertex,
    Fragment,
    Geometry,
}
impl Into<u32> for ShaderKind {
    fn into(self) -> u32 {
        match self {
            ShaderKind::Vertex => gl::VERTEX_SHADER,
            ShaderKind::Fragment => gl::FRAGMENT_SHADER,
            ShaderKind::Geometry => gl::GEOMETRY_SHADER,
        }
    }
}
pub struct Shader {
    pub kind: ShaderKind,
    pub id: gl::types::GLuint,
}
impl Shader {
    pub fn new_from_path<T: AsRef<Path>>(path: T, kind: ShaderKind) -> Result<Shader, ()> {
        let src = fs::read_to_string(path).expect("unable to load vertex shader");
        Shader::new(&src, kind)
    }
    pub fn new(src: &str, kind: ShaderKind) -> Result<Shader, ()> {
        let id = unsafe {
            let shader_id = gl::CreateShader(kind.into());
            let source = ffi::CString::new(src.as_bytes()).unwrap();
            gl::ShaderSource(
                shader_id,
                1,
                [source.as_ptr()].as_ptr(),
                ptr::null() as *const _,
            );
            gl::CompileShader(shader_id);

            let mut success = mem::uninitialized();
            gl::GetShaderiv(shader_id, gl::COMPILE_STATUS, &mut success);
            if success == 0 {
                let mut error_log_size = 512;
                let mut buffer: Vec<u8> = Vec::with_capacity(error_log_size as usize);
                gl::GetShaderInfoLog(
                    shader_id,
                    error_log_size,
                    &mut error_log_size,
                    buffer.as_mut_ptr() as *mut _,
                );
                buffer.set_len(error_log_size as usize);
                let error_msg = String::from_utf8(buffer);
                println!("Error while compiling shader of type {:?}", kind);
                for line in error_msg.unwrap().lines() {
                    println!("{}", line);
                }
                panic!();
            }

            shader_id
        };

        Ok(Shader { kind, id })
    }
}
impl Drop for Shader {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteShader(self.id);
        }
    }
}

pub struct VertexShader(pub Shader);
impl VertexShader {
    pub fn new_from_path<T: AsRef<Path>>(path: T) -> Result<VertexShader, ()> {
        let shader = Shader::new_from_path(path, ShaderKind::Vertex)?;
        Ok(VertexShader(shader))
    }
    pub fn new(src: &str) -> Result<VertexShader, ()> {
        let shader = Shader::new(src, ShaderKind::Vertex)?;
        Ok(VertexShader(shader))
    }
}

pub struct FragmentShader(pub Shader);
impl FragmentShader {
    pub fn new_from_path<T: AsRef<Path>>(path: T) -> Result<FragmentShader, ()> {
        let shader = Shader::new_from_path(path, ShaderKind::Fragment)?;
        Ok(FragmentShader(shader))
    }
    pub fn new(src: &str) -> Result<FragmentShader, ()> {
        let shader = Shader::new(src, ShaderKind::Fragment)?;
        Ok(FragmentShader(shader))
    }
}

pub struct GeometryShader(pub Shader);
impl GeometryShader {
    pub fn new_from_path<T: AsRef<Path>>(path: T) -> Result<GeometryShader, ()> {
        let shader = Shader::new_from_path(path, ShaderKind::Geometry)?;
        Ok(GeometryShader(shader))
    }
    pub fn new(src: &str) -> Result<GeometryShader, ()> {
        let shader = Shader::new(src, ShaderKind::Geometry)?;
        Ok(GeometryShader(shader))
    }
}

pub struct UniformLocation<'a>(gl::types::GLint, PhantomData<&'a u8>);
impl<'a> UniformLocation<'a> {
    fn new(loc: gl::types::GLint) -> UniformLocation<'a> {
        UniformLocation(loc, PhantomData)
    }
}

pub struct UniformBlockIndex<'a>(gl::types::GLuint, PhantomData<&'a u8>);
impl<'a> UniformBlockIndex<'a> {
    fn new(loc: gl::types::GLuint) -> UniformBlockIndex<'a> {
        UniformBlockIndex(loc, PhantomData)
    }
}

pub struct Program {
    id: gl::types::GLuint,
}
impl Program {
    pub fn new(
        vs: &VertexShader,
        gs: Option<&GeometryShader>,
        fs: &FragmentShader,
    ) -> Result<Program, ()> {
        let id = unsafe { gl::CreateProgram() };
        let program = Program { id };

        program.attach_shader(&vs.0);
        if let Some(gs) = gs {
            program.attach_shader(&gs.0);
        }
        program.attach_shader(&fs.0);

        program.link()?;

        Ok(program)
    }
    pub fn attach_shader(&self, shader: &Shader) {
        unsafe {
            gl::AttachShader(self.id, shader.id);
        }
    }
    fn link(&self) -> Result<(), ()> {
        let success = unsafe {
            gl::LinkProgram(self.id);

            let mut success = mem::uninitialized();
            gl::GetProgramiv(self.id, gl::LINK_STATUS, &mut success);
            success
        };

        if success == 0 {
            let mut error_log_size = 512;
            let mut buffer: Vec<u8> = Vec::with_capacity(error_log_size as usize);
            unsafe {
                gl::GetProgramInfoLog(
                    self.id,
                    error_log_size,
                    &mut error_log_size,
                    buffer.as_mut_ptr() as *mut _,
                );
                buffer.set_len(error_log_size as usize);
            }
            let error_msg = String::from_utf8(buffer);
            for line in error_msg.unwrap().lines() {
                println!("{}", line);
            }
            Err(())
        } else {
            Ok(())
        }
    }
    pub fn new_from_disk(
        vs_path: &str,
        gs_path: Option<&str>,
        fs_path: &str,
    ) -> Result<Program, ()> {
        let vs = VertexShader::new_from_path(vs_path).expect("unable to load vertex shader");
        let gs = gs_path.map(|gs_path| {
            GeometryShader::new_from_path(gs_path).expect("unable to load geometry shader")
        });
        let fs = FragmentShader::new_from_path(fs_path).expect("unable to load fragment shader");

        Program::new(&vs, gs.as_ref(), &fs)
    }
    #[allow(unused)]
    pub fn new_from_src(
        vs_src: Cow<str>,
        gs_src: Option<Cow<str>>,
        fs_src: Cow<str>,
    ) -> Result<Program, ()> {
        let vs = VertexShader::new(&vs_src)?;
        let gs = gs_src
            .map(|gs_src| GeometryShader::new(&gs_src))
            .transpose()?;
        let fs = FragmentShader::new(&fs_src)?;

        Program::new(&vs, gs.as_ref(), &fs)
    }
    pub fn bind(&mut self) -> ProgramBinding {
        ProgramBinding::new(self)
    }
    // pub fn hot_swap_vertex_shader(&self, vs_path: &str) {
    //     let vs = VertexShader::new_from_path(vs_path).expect("unable to load vertex shader");
    //     self.attach_shader(&vs.0);
    // }
    // pub fn hot_swap_fragment_shader(&self, fs_path: &str) {
    //     let vs = FragmentShader::new_from_path(fs_path).expect("unable to load fragment shader");
    //     self.attach_shader(&vs.0);
    //     self.link();
    // }
}
impl Drop for Program {
    fn drop(&mut self) {
        unsafe { gl::DeleteProgram(self.id) }
    }
}
pub struct ProgramBinding<'a>(&'a mut Program);
impl<'a> ProgramBinding<'a> {
    fn new(program: &'a mut Program) -> ProgramBinding<'a> {
        unsafe {
            gl::UseProgram(program.id);
        }
        ProgramBinding(program)
    }
    pub fn get_uniform_location<'b>(&'a self, name: &str) -> UniformLocation<'b> {
        let loc = unsafe {
            gl::GetUniformLocation(
                self.0.id,
                ffi::CString::new(name)
                    .expect("unable to create a CString from passes str")
                    .as_ptr(),
            )
        };
        UniformLocation::new(loc)
    }
    pub fn bind_mat4<T: Into<[[f32; 4]; 4]>>(&self, name: &str, mat: T) -> &ProgramBinding {
        let mat = mat.into();
        let loc = self.get_uniform_location(name);
        unsafe {
            gl::UniformMatrix4fv(loc.0, 1, gl::FALSE, &mat as *const _ as *const _);
        }
        self
    }
    pub fn bind_mat4s<T: Into<[[f32; 4]; 4]> + Copy>(
        &self,
        name: &str,
        mats: &[T],
    ) -> &ProgramBinding {
        for (i, mat) in mats.iter().enumerate() {
            self.bind_mat4(&format!("{}[{}]", name, i), *mat);
        }
        self
    }
    pub fn bind_int(&self, name: &str, i: i32) -> &ProgramBinding {
        let loc = self.get_uniform_location(name);
        unsafe {
            gl::Uniform1i(loc.0, i);
        }
        self
    }
    pub fn bind_uint(&self, name: &str, i: u32) -> &ProgramBinding {
        let loc = self.get_uniform_location(name);
        unsafe {
            gl::Uniform1ui(loc.0, i);
        }
        self
    }
    pub fn bind_bool(&self, name: &str, i: bool) -> &ProgramBinding {
        self.bind_uint(name, if i { 1 } else { 0 })
    }
    pub fn bind_texture(
        &self,
        name: &str,
        texture: &Texture,
        slot: TextureSlot,
    ) -> &ProgramBinding {
        texture.bind_to(slot);
        self.bind_int(name, slot.into())
    }
    pub fn bind_float(&self, name: &str, f: f32) -> &ProgramBinding {
        let loc = self.get_uniform_location(name);
        unsafe {
            gl::Uniform1f(loc.0, f);
        }
        self
    }
    pub fn bind_vec3<T: Into<[f32; 3]>>(&self, name: &str, v: T) -> &ProgramBinding {
        let v = v.into();
        let loc = self.get_uniform_location(name);
        unsafe {
            gl::Uniform3f(loc.0, v[0], v[1], v[2]);
        }
        self
    }
    #[allow(unused)]
    pub fn get_uniform_block_index(&'a self, name: &str) -> UniformBlockIndex<'a> {
        let loc = unsafe {
            gl::GetUniformBlockIndex(self.0.id, ffi::CString::new(name).unwrap().as_ptr())
        };
        UniformBlockIndex::new(loc)
    }
    #[allow(unused)]
    pub fn uniform_block_binding(&self, name: &str, index: usize) -> &ProgramBinding {
        let block_index = self.get_uniform_block_index(name);
        unsafe {
            gl::UniformBlockBinding(self.0.id, block_index.0, index as u32);
        }
        self
    }
}

#[allow(unused)]
#[derive(Debug, Clone, Copy)]
pub enum TextureSlot {
    Zero,
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Ten,
}
impl Into<u32> for TextureSlot {
    fn into(self) -> u32 {
        let i: i32 = self.into();
        gl::TEXTURE0 + i as u32
    }
}
impl Into<i32> for TextureSlot {
    fn into(self) -> i32 {
        use self::TextureSlot::*;
        match self {
            Zero => 0,
            One => 1,
            Two => 2,
            Three => 3,
            Four => 4,
            Five => 5,
            Six => 6,
            Seven => 7,
            Eight => 8,
            Nine => 9,
            Ten => 10,
        }
    }
}
impl From<usize> for TextureSlot {
    fn from(nth: usize) -> TextureSlot {
        use self::TextureSlot::*;
        match nth {
            0 => Zero,
            1 => One,
            2 => Two,
            3 => Three,
            4 => Four,
            5 => Five,
            6 => Six,
            7 => Seven,
            8 => Eight,
            9 => Nine,
            10 => Ten,
            x => unimplemented!("{:?}", x),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum TextureKind {
    Texture2d,
    CubeMap,
}
impl Into<u32> for TextureKind {
    fn into(self) -> u32 {
        match self {
            TextureKind::Texture2d => gl::TEXTURE_2D,
            TextureKind::CubeMap => gl::TEXTURE_CUBE_MAP,
        }
    }
}

#[derive(Debug)]
pub struct Texture {
    id: gl::types::GLuint,
    kind: TextureKind,
}

impl Texture {
    pub fn new(kind: TextureKind) -> Texture {
        let id = unsafe {
            let mut tex_id = mem::uninitialized();
            gl::GenTextures(1, &mut tex_id);
            tex_id
        };

        Texture { id, kind }
    }
    pub fn bind(&self) -> TextureBinder {
        TextureBinder::new(self)
    }
    pub fn bind_to(&self, slot: TextureSlot) {
        unsafe {
            gl::ActiveTexture(slot.into());
        }
        self.bind();
        GlError::check().expect(&format!("unable to bind texture to slot {:?}", slot));
    }
}
impl Drop for Texture {
    fn drop(&mut self) {
        unsafe { gl::DeleteTextures(1, &self.id as *const _) }
    }
}

#[allow(unused)]
#[derive(Debug, Clone, Copy)]
pub enum TextureTarget {
    Texture2d,
    ProxyTexture2d,
    Texture1dArray,
    ProxyTexture1dArray,
    TextureRectangle,
    ProxyTextureRectangle,
    TextureCubeMapPositiveX,
    TextureCubeMapNegativeX,
    TextureCubeMapPositiveY,
    TextureCubeMapNegativeY,
    TextureCubeMapPositiveZ,
    TextureCubeMapNegativeZ,
    ProxyTextureCubeMap,
}
impl TextureTarget {
    pub fn cubemap_faces() -> [TextureTarget; 6] {
        use self::TextureTarget::*;
        [
            TextureCubeMapPositiveX,
            TextureCubeMapNegativeX,
            TextureCubeMapPositiveY,
            TextureCubeMapNegativeY,
            TextureCubeMapPositiveZ,
            TextureCubeMapNegativeZ,
        ]
    }
}
impl Into<u32> for TextureTarget {
    fn into(self) -> u32 {
        use TextureTarget::*;
        match self {
            Texture2d => gl::TEXTURE_2D,
            ProxyTexture2d => gl::PROXY_TEXTURE_2D,
            Texture1dArray => gl::TEXTURE_1D_ARRAY,
            ProxyTexture1dArray => gl::PROXY_TEXTURE_1D_ARRAY,
            TextureRectangle => gl::TEXTURE_RECTANGLE,
            ProxyTextureRectangle => gl::PROXY_TEXTURE_RECTANGLE,
            TextureCubeMapPositiveX => gl::TEXTURE_CUBE_MAP_POSITIVE_X,
            TextureCubeMapNegativeX => gl::TEXTURE_CUBE_MAP_NEGATIVE_X,
            TextureCubeMapPositiveY => gl::TEXTURE_CUBE_MAP_POSITIVE_Y,
            TextureCubeMapNegativeY => gl::TEXTURE_CUBE_MAP_NEGATIVE_Y,
            TextureCubeMapPositiveZ => gl::TEXTURE_CUBE_MAP_POSITIVE_Z,
            TextureCubeMapNegativeZ => gl::TEXTURE_CUBE_MAP_NEGATIVE_Z,
            ProxyTextureCubeMap => gl::PROXY_TEXTURE_CUBE_MAP,
        }
    }
}

#[allow(unused)]
pub enum TextureInternalFormat {
    Rgba32f,
    Rgba32i,
    Rgba32ui,
    Rgba16,
    Rgba16f,
    Rgba16i,
    Rgba16ui,
    Rgba,
    Rgba8,
    Rgba8ui,
    Srgb8Alpha8,
    Rgb10A2,
    Rgb10A2ui,
    R11fG11fB10f,
    Rg32f,
    Rg32i,
    Rg32ui,
    Rg16,
    Rg16f,
    Rgb16i,
    Rgb16ui,
    Rg8,
    Rg8i,
    Rg8ui,
    R32f,
    R32i,
    R32ui,
    R16f,
    R16i,
    R16ui,
    R8,
    R8i,
    R8ui,
    Rgba16Snorm,
    Rgba8Snorm,
    Rgb32f,
    Rgb32i,
    Rgb32ui,
    Rgb16Snorm,
    Rgb16f,
    Rgb16,
    Rgb8Snorm,
    Rgb,
    Rgb8,
    Rgb8i,
    Rgb8ui,
    Srgb8,
    Srgb,
    Rgb9E5,
    Rg16Snorm,
    Rg8Snorm,
    CompressedRgRgtc2,
    CompressedSignedRgRgtc2,
    R16Snorm,
    R8Snorm,
    CompressedRedRgtc1,
    CompressedSignedRedRgtc1,
    DepthComponent,
    DepthComponent32f,
    DepthComponent24,
    DepthComponent16,
    Depth32fStencil8,
    Depth24Stencil8,
}

impl Into<u32> for TextureInternalFormat {
    fn into(self) -> u32 {
        use TextureInternalFormat::*;
        match self {
            Rgba32f => gl::RGBA32F,
            Rgba32i => gl::RGBA32I,
            Rgba32ui => gl::RGBA32UI,
            Rgba16 => gl::RGBA16,
            Rgba16f => gl::RGBA16F,
            Rgba16i => gl::RGBA16I,
            Rgba16ui => gl::RGBA16UI,
            Rgba => gl::RGBA,
            Rgba8 => gl::RGBA8,
            Rgba8ui => gl::RGBA8UI,
            Srgb8Alpha8 => gl::SRGB8_ALPHA8,
            Rgb10A2 => gl::RGB10_A2,
            Rgb10A2ui => gl::RGB10_A2UI,
            R11fG11fB10f => gl::R11F_G11F_B10F,
            Rg32f => gl::RG32F,
            Rg32i => gl::RG32I,
            Rg32ui => gl::RG32UI,
            Rg16 => gl::RG16,
            Rg16f => gl::RG16F,
            Rgb16i => gl::RGB16I,
            Rgb16ui => gl::RGB16UI,
            Rg8 => gl::RG8,
            Rg8i => gl::RG8I,
            Rg8ui => gl::RG8UI,
            R32f => gl::R32F,
            R32i => gl::R32I,
            R32ui => gl::R32UI,
            R16f => gl::R16F,
            R16i => gl::R16I,
            R16ui => gl::R16UI,
            R8 => gl::R8,
            R8i => gl::R8I,
            R8ui => gl::R8UI,
            Rgba16Snorm => gl::RGBA16_SNORM,
            Rgba8Snorm => gl::RGBA8_SNORM,
            Rgb32f => gl::RGB32F,
            Rgb32i => gl::RGB32I,
            Rgb32ui => gl::RGB32UI,
            Rgb16Snorm => gl::RGB16_SNORM,
            Rgb16f => gl::RGB16F,
            Rgb16 => gl::RGB16,
            Rgb8Snorm => gl::RGB8_SNORM,
            Rgb => gl::RGB,
            Rgb8 => gl::RGB8,
            Rgb8i => gl::RGB8I,
            Rgb8ui => gl::RGB8UI,
            Srgb => gl::SRGB,
            Srgb8 => gl::SRGB8,
            Rgb9E5 => gl::RGB9_E5,
            Rg16Snorm => gl::RG16_SNORM,
            Rg8Snorm => gl::RG8_SNORM,
            CompressedRgRgtc2 => gl::COMPRESSED_RG_RGTC2,
            CompressedSignedRgRgtc2 => gl::COMPRESSED_SIGNED_RG_RGTC2,
            R16Snorm => gl::R16_SNORM,
            R8Snorm => gl::R8_SNORM,
            CompressedRedRgtc1 => gl::COMPRESSED_RED_RGTC1,
            CompressedSignedRedRgtc1 => gl::COMPRESSED_SIGNED_RED_RGTC1,
            DepthComponent => gl::DEPTH_COMPONENT,
            DepthComponent32f => gl::DEPTH_COMPONENT32F,
            DepthComponent24 => gl::DEPTH_COMPONENT24,
            DepthComponent16 => gl::DEPTH_COMPONENT16,
            Depth32fStencil8 => gl::DEPTH32F_STENCIL8,
            Depth24Stencil8 => gl::DEPTH24_STENCIL8,
        }
    }
}

#[allow(unused)]
#[derive(Debug, Clone, Copy)]
pub enum TextureFormat {
    Red,  // GL_RED,
    Rg,   // GL_RG,
    Rgb,  // GL_RGB,
    Bgr,  // GL_BGR,
    Rgba, // GL_RGBA,
    Bgra, // GL_BGRA,

    DepthComponent,
}
impl Into<u32> for TextureFormat {
    fn into(self) -> u32 {
        use TextureFormat::*;
        match self {
            Red => gl::RED,
            Rg => gl::RG,
            Rgb => gl::RGB,
            Bgr => gl::BGR,
            Rgba => gl::RGBA,
            Bgra => gl::BGRA,
            DepthComponent => gl::DEPTH_COMPONENT,
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

pub struct TextureBinder<'a>(&'a Texture);
impl<'a> TextureBinder<'a> {
    fn new(texture: &Texture) -> TextureBinder {
        unsafe {
            gl::BindTexture(texture.kind.into(), texture.id);
        }
        TextureBinder(texture)
    }
    pub fn empty(
        &self,
        target: TextureTarget,
        level: usize,
        internal_format: TextureInternalFormat,
        width: u32,
        height: u32,
        format: TextureFormat,
        typ: GlType,
    ) -> &TextureBinder {
        unsafe {
            self.image_2d(
                target,
                level,
                internal_format,
                width,
                height,
                format,
                typ,
                ptr::null(),
            );
        }
        self
    }
    pub unsafe fn image_2d(
        &self,
        target: TextureTarget,
        level: usize,
        internal_format: TextureInternalFormat,
        width: u32,
        height: u32,
        format: TextureFormat,
        typ: GlType,
        data: *const c_void,
    ) {
        let internal_format: u32 = internal_format.into();
        gl::TexImage2D(
            target.into(),
            level as i32,
            internal_format as i32,
            width as i32,
            height as i32,
            0,
            format.into(),
            typ.into(),
            data,
        );
    }
    pub fn load_image(
        &self,
        target: TextureTarget,
        internal_format: TextureInternalFormat,
        format: TextureFormat,
        img: &image::DynamicImage,
    ) -> &TextureBinder {
        use image::GenericImage;
        let (w, h) = img.dimensions();

        unsafe {
            self.image_2d(
                target,
                0,
                internal_format,
                w,
                h,
                format,
                GlType::UnsignedByte,
                &(match format {
                    // TextureFormat::Red => img.to_red().into_raw(),
                    TextureFormat::Red => unimplemented!(),
                    // TextureFormat::Rg => img.to_rg().into_raw(),
                    TextureFormat::Rg => unimplemented!(),
                    TextureFormat::Rgb => img.to_rgb().into_raw(),
                    // TextureFormat::Bgr => img.to_bgr().into_raw(),
                    TextureFormat::Bgr => unimplemented!(),
                    TextureFormat::Rgba => img.to_rgba().into_raw(),
                    // TextureFormat::Bgra => img.to_bgra().into_raw(),
                    TextureFormat::Bgra => unimplemented!(),
                    TextureFormat::DepthComponent => unimplemented!(),
                })[0] as *const _ as *const _,
            );
        }
        self
    }
    pub fn parameter_int(&self, pname: TextureParameter, param: i32) -> &TextureBinder {
        unsafe {
            gl::TexParameteri(self.0.kind.into(), pname.into(), param);
        }
        self
    }
}

#[allow(unused)]
pub enum TextureParameter {
    BaseLevel,
    CompareFunc,
    CompareMode,
    LodBias,
    MinFilter,
    MagFilter,
    MinLod,
    MaxLod,
    MaxLevel,
    SwizzleR,
    SwizzleG,
    SwizzleB,
    SwizzleA,
    WrapS,
    WrapT,
    WrapR,
}
impl Into<u32> for TextureParameter {
    fn into(self) -> u32 {
        use TextureParameter::*;
        match self {
            BaseLevel => gl::TEXTURE_BASE_LEVEL,
            CompareFunc => gl::TEXTURE_COMPARE_FUNC,
            CompareMode => gl::TEXTURE_COMPARE_MODE,
            LodBias => gl::TEXTURE_LOD_BIAS,
            MinFilter => gl::TEXTURE_MIN_FILTER,
            MagFilter => gl::TEXTURE_MAG_FILTER,
            MinLod => gl::TEXTURE_MIN_LOD,
            MaxLod => gl::TEXTURE_MAX_LOD,
            MaxLevel => gl::TEXTURE_MAX_LEVEL,
            SwizzleR => gl::TEXTURE_SWIZZLE_R,
            SwizzleG => gl::TEXTURE_SWIZZLE_G,
            SwizzleB => gl::TEXTURE_SWIZZLE_B,
            SwizzleA => gl::TEXTURE_SWIZZLE_A,
            WrapS => gl::TEXTURE_WRAP_S,
            WrapT => gl::TEXTURE_WRAP_T,
            WrapR => gl::TEXTURE_WRAP_R,
        }
    }
}

#[allow(unused)]
#[derive(Debug, Clone, Copy)]
pub enum Attachment {
    Color0,
    Color1,
    Color2,
    Color3,
    Color4,
    Depth,
    Stencil,
    DepthStencil,
}
impl Into<u32> for Attachment {
    fn into(self) -> u32 {
        use Attachment::*;
        match self {
            Color0 => gl::COLOR_ATTACHMENT0,
            Color1 => gl::COLOR_ATTACHMENT1,
            Color2 => gl::COLOR_ATTACHMENT2,
            Color3 => gl::COLOR_ATTACHMENT3,
            Color4 => gl::COLOR_ATTACHMENT4,
            Depth => gl::DEPTH_ATTACHMENT,
            Stencil => gl::STENCIL_ATTACHMENT,
            DepthStencil => gl::DEPTH_STENCIL_ATTACHMENT,
        }
    }
}

#[allow(unused)]
#[derive(Debug, Clone, Copy)]
pub enum BufferSlot {
    None,
    FrontLeft,
    FrontRight,
    BackLeft,
    BackRight,
    Front,
    Back,
    Left,
    Right,
    FrontAndBack,
}
impl Into<u32> for BufferSlot {
    fn into(self) -> u32 {
        use BufferSlot::*;
        match self {
            None => gl::NONE,
            FrontLeft => gl::FRONT_LEFT,
            FrontRight => gl::FRONT_RIGHT,
            BackLeft => gl::BACK_LEFT,
            BackRight => gl::BACK_RIGHT,
            Front => gl::FRONT,
            Back => gl::BACK,
            Left => gl::LEFT,
            Right => gl::RIGHT,
            FrontAndBack => gl::FRONT_AND_BACK,
        }
    }
}

#[derive(Debug)]
pub struct Framebuffer {
    id: gl::types::GLuint,
}
impl Framebuffer {
    pub unsafe fn window() -> Framebuffer {
        Framebuffer { id: 0 }
    }
    pub fn new() -> Framebuffer {
        let id = unsafe {
            let mut fb_id = mem::uninitialized();
            gl::GenFramebuffers(1, &mut fb_id);
            fb_id
        };

        Framebuffer { id }
    }
    pub fn read(&mut self) -> FramebufferBinderRead {
        FramebufferBinderRead::new(self)
    }
    pub fn draw(&mut self) -> FramebufferBinderDraw {
        FramebufferBinderDraw::new(self)
    }
    pub fn read_draw(&mut self) -> FramebufferBinderReadDraw {
        FramebufferBinderReadDraw::new(self)
    }
    pub fn bind(&mut self) -> FramebufferBinderReadDraw {
        self.read_draw()
    }
}
impl Drop for Framebuffer {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteFramebuffers(1, &self.id as *const _);
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[allow(unused)]
pub enum FramebufferTarget {
    Read,
    Draw,
    ReadDraw,
}
impl Into<u32> for FramebufferTarget {
    fn into(self) -> u32 {
        use FramebufferTarget::*;
        match self {
            Read => gl::READ_FRAMEBUFFER,
            Draw => gl::DRAW_FRAMEBUFFER,
            ReadDraw => gl::FRAMEBUFFER,
        }
    }
}

pub trait FramebufferBinderBase {
    fn id(&self) -> gl::types::GLuint;
    fn target() -> FramebufferTarget;
    fn is_complete(&self) -> bool {
        let status = unsafe { gl::CheckFramebufferStatus(Self::target().into()) };
        status == gl::FRAMEBUFFER_COMPLETE
    }
    fn check_status(&self) -> Result<(), ()> {
        if self.is_complete() {
            Ok(())
        } else {
            Err(())
        }
    }
    fn draw_buffer(&self, slot: BufferSlot) -> &Self {
        unsafe {
            gl::DrawBuffer(slot.into());
        }
        self
    }
    fn read_buffer(&self, slot: BufferSlot) -> &Self {
        unsafe {
            gl::ReadBuffer(slot.into());
        }
        self
    }
}

pub trait FramebufferBinderReader: FramebufferBinderBase {}

#[allow(unused)]
#[derive(Debug, Clone, Copy)]
pub enum ClearMask {
    Color,
    Depth,
    Stencil,

    ColorDepth,
    ColorStencil,
    DepthStencil,
}
impl Into<u32> for ClearMask {
    fn into(self) -> u32 {
        use ClearMask::*;
        match self {
            Color => gl::COLOR_BUFFER_BIT,
            Depth => gl::DEPTH_BUFFER_BIT,
            Stencil => gl::STENCIL_BUFFER_BIT,

            ColorDepth => gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT,
            ColorStencil => gl::COLOR_BUFFER_BIT | gl::STENCIL_BUFFER_BIT,
            DepthStencil => gl::DEPTH_BUFFER_BIT | gl::STENCIL_BUFFER_BIT,
        }
    }
}

pub trait FramebufferBinderDrawer: FramebufferBinderBase {
    fn clear(&self, mask: ClearMask) -> &Self {
        unsafe {
            gl::Clear(mask.into());
        }
        self
    }
    fn texture(&self, attachment: Attachment, texture: &Texture, level: usize) -> &Self {
        unsafe {
            gl::FramebufferTexture(
                Self::target().into(),
                attachment.into(),
                texture.id,
                level as i32,
            );
        }
        self
    }
    fn texture_2d(
        &self,
        attachment: Attachment,
        textarget: TextureTarget,
        texture: &Texture,
        level: usize,
    ) -> &Self {
        unsafe {
            gl::FramebufferTexture2D(
                Self::target().into(),
                attachment.into(),
                textarget.into(),
                texture.id,
                level as i32,
            );
        }
        self
    }
    fn renderbuffer(&self, attachment: Attachment, renderbuffer: &Renderbuffer) -> &Self {
        unsafe {
            gl::FramebufferRenderbuffer(
                Self::target().into(),
                attachment.into(),
                gl::RENDERBUFFER,
                renderbuffer.id,
            );
        }
        self
    }
    fn draw_buffers(&self, bufs: &[Attachment]) -> &Self {
        let mut data: Vec<u32> = Vec::with_capacity(bufs.len());
        for buf in bufs {
            data.push((*buf).into());
        }
        unsafe {
            gl::DrawBuffers(data.len() as i32, &data[0] as *const _ as *const _);
        }
        self
    }
    fn blit_framebuffer<T: FramebufferBinderReader>(
        &self,
        _from: &T,
        src: (i32, i32, i32, i32),
        dst: (i32, i32, i32, i32),
        mask: u32,
        filter: u32,
    ) -> &Self {
        unsafe {
            gl::BlitFramebuffer(
                src.0, src.1, src.2, src.3, dst.0, dst.1, dst.2, dst.3, mask, filter,
            );
        }
        self
    }
}

#[derive(Debug)]
pub struct FramebufferBinderRead<'a>(&'a mut Framebuffer);
impl<'a> FramebufferBinderBase for FramebufferBinderRead<'a> {
    fn id(&self) -> gl::types::GLuint {
        self.0.id
    }
    fn target() -> FramebufferTarget {
        FramebufferTarget::Read
    }
}
impl<'a> FramebufferBinderRead<'a> {
    fn new(fb: &mut Framebuffer) -> FramebufferBinderRead {
        unsafe {
            gl::BindFramebuffer(Self::target().into(), fb.id);
        }
        FramebufferBinderRead(fb)
    }
}
impl<'a> FramebufferBinderReader for FramebufferBinderRead<'a> {}
impl<'a> Drop for FramebufferBinderRead<'a> {
    fn drop(&mut self) {
        unsafe {
            gl::BindFramebuffer(Self::target().into(), 0);
        }
    }
}

#[derive(Debug)]
pub struct FramebufferBinderDraw<'a>(&'a mut Framebuffer);
impl<'a> FramebufferBinderBase for FramebufferBinderDraw<'a> {
    fn id(&self) -> gl::types::GLuint {
        self.0.id
    }
    fn target() -> FramebufferTarget {
        FramebufferTarget::Draw
    }
}
impl<'a> FramebufferBinderDraw<'a> {
    fn new(fb: &mut Framebuffer) -> FramebufferBinderDraw {
        unsafe {
            gl::BindFramebuffer(Self::target().into(), fb.id);
        }
        FramebufferBinderDraw(fb)
    }
}
impl<'a> FramebufferBinderDrawer for FramebufferBinderDraw<'a> {}
impl<'a> Drop for FramebufferBinderDraw<'a> {
    fn drop(&mut self) {
        unsafe {
            gl::BindFramebuffer(Self::target().into(), 0);
        }
    }
}

#[derive(Debug)]
pub struct FramebufferBinderReadDraw<'a>(&'a mut Framebuffer);
impl<'a> FramebufferBinderBase for FramebufferBinderReadDraw<'a> {
    fn id(&self) -> gl::types::GLuint {
        self.0.id
    }
    fn target() -> FramebufferTarget {
        FramebufferTarget::ReadDraw
    }
}
impl<'a> FramebufferBinderReadDraw<'a> {
    fn new(fb: &mut Framebuffer) -> FramebufferBinderReadDraw {
        unsafe {
            gl::BindFramebuffer(Self::target().into(), fb.id);
        }
        FramebufferBinderReadDraw(fb)
    }
}
impl<'a> FramebufferBinderReader for FramebufferBinderReadDraw<'a> {}
impl<'a> FramebufferBinderDrawer for FramebufferBinderReadDraw<'a> {}
impl<'a> Drop for FramebufferBinderReadDraw<'a> {
    fn drop(&mut self) {
        unsafe {
            gl::BindFramebuffer(Self::target().into(), 0);
        }
    }
}

pub struct Renderbuffer {
    id: gl::types::GLuint,
}
impl Renderbuffer {
    pub fn new() -> Renderbuffer {
        let id = unsafe {
            let mut fb_id = mem::uninitialized();
            gl::GenRenderbuffers(1, &mut fb_id);
            fb_id
        };

        Renderbuffer { id }
    }
    pub fn bind(&mut self) -> RenderbufferBinder {
        RenderbufferBinder::new(self)
    }
}
impl Drop for Renderbuffer {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteRenderbuffers(1, &self.id as *const _);
        }
    }
}

pub struct RenderbufferBinder<'a>(&'a mut Renderbuffer);
impl<'a> RenderbufferBinder<'a> {
    fn new(fb: &mut Renderbuffer) -> RenderbufferBinder {
        unsafe {
            gl::BindRenderbuffer(gl::RENDERBUFFER, fb.id);
        }
        RenderbufferBinder(fb)
    }
    pub fn storage(&self, internal_format: TextureInternalFormat, width: u32, height: u32) {
        unsafe {
            gl::RenderbufferStorage(
                gl::RENDERBUFFER,
                internal_format.into(),
                width as i32,
                height as i32,
            );
        }
    }
}
impl<'a> Drop for RenderbufferBinder<'a> {
    fn drop(&mut self) {
        unsafe {
            gl::BindRenderbuffer(gl::RENDERBUFFER, 0);
        }
    }
}

pub struct VertexArray(gl::types::GLuint);

#[derive(Debug, Clone, Copy)]
pub enum BufferKind {
    Array,
    ElementArray,
    Uniform,
}

impl Into<u32> for BufferKind {
    fn into(self) -> u32 {
        match self {
            BufferKind::Array => gl::ARRAY_BUFFER,
            BufferKind::ElementArray => gl::ELEMENT_ARRAY_BUFFER,
            BufferKind::Uniform => gl::UNIFORM_BUFFER,
        }
    }
}

pub struct Buffer<T> {
    kind: BufferKind,
    id: gl::types::GLuint,
    buffer_size: Option<usize>,
    phantom: PhantomData<T>,
}

impl<T> Buffer<T> {
    pub fn new(kind: BufferKind) -> Buffer<T> {
        let id = unsafe {
            let mut id = mem::uninitialized();
            gl::GenBuffers(1, &mut id);
            id
        };
        Buffer {
            kind,
            id,
            buffer_size: None,
            phantom: PhantomData,
        }
    }
    pub fn bind(&mut self) -> BufferBinder<T> {
        BufferBinder::new(self)
    }
    pub fn len(&self) -> usize {
        self.size() / mem::size_of::<T>()
    }
    pub fn size(&self) -> usize {
        self.buffer_size.unwrap_or(0)
    }
}
impl<V> Drop for Buffer<V> {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteBuffers(1, &self.id as *const _);
        }
    }
}

pub struct BufferBinder<'a, T: 'a>(&'a mut Buffer<T>);
impl<'a, T> BufferBinder<'a, T> {
    pub fn new(buffer: &'a mut Buffer<T>) -> BufferBinder<'a, T> {
        unsafe { gl::BindBuffer(buffer.kind.into(), buffer.id) }
        GlError::check().expect("error binding buffer BufferBinder::new");
        BufferBinder(buffer)
    }
    unsafe fn buffer_raw_data(&mut self, size: usize, data: *const os::raw::c_void) {
        gl::BufferData(self.0.kind.into(), size as isize, data, gl::STATIC_DRAW);
        GlError::check().expect("error writing BufferBinder::buffer_raw_data");
        self.0.buffer_size = Some(size);
    }
    unsafe fn buffer_raw_sub_data(
        &mut self,
        offset: usize,
        size: usize,
        data: *const os::raw::c_void,
    ) {
        gl::BufferSubData(self.0.kind.into(), offset as isize, size as isize, data);
        GlError::check().expect("error writing BufferBinder::buffer_raw_sub_data");
    }
    pub fn alloc(&mut self, size: usize) {
        unsafe {
            self.buffer_raw_data(size, ptr::null());
        }
    }
    pub fn alloc_elements(&mut self, num_elements: usize) {
        self.alloc(num_elements * mem::size_of::<T>());
    }
    pub fn buffer_data(&mut self, data: &[T]) {
        let size = data.len() * mem::size_of::<T>();
        if size > 0 {
            let data_ptr = &data[0] as *const _ as *const _;
            unsafe {
                self.buffer_raw_data(size, data_ptr);
            }
        }
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
}
impl<'a, T> Drop for BufferBinder<'a, T> {
    fn drop(&mut self) {
        unsafe {
            gl::BindBuffer(self.0.kind.into(), 0);
        }
    }
}

pub struct VertexBuffer<T> {
    buffer: Buffer<T>,
}
pub struct Ebo<T>(Buffer<T>);

impl VertexArray {
    pub fn new() -> VertexArray {
        unsafe {
            let mut vao = mem::uninitialized();
            gl::GenVertexArrays(1, &mut vao);
            VertexArray(vao)
        }
    }
    pub fn bind(&mut self) -> VertexArrayBinder {
        VertexArrayBinder::new(self)
    }
}

impl<T> VertexBuffer<T> {
    pub fn new() -> VertexBuffer<T> {
        VertexBuffer {
            buffer: Buffer::new(BufferKind::Array),
        }
    }
    pub fn from_data(data: &[T]) -> VertexBuffer<T> {
        let mut vbo = VertexBuffer::new();
        vbo.bind().buffer_data(data);
        vbo
    }
    #[allow(unused)]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    pub fn bind(&mut self) -> VboBinder<T> {
        VboBinder::new(self)
    }
}

impl<T> Ebo<T> {
    pub fn new() -> Ebo<T> {
        Ebo(Buffer::new(BufferKind::ElementArray))
    }
}

pub struct UniformBuffer<T> {
    buffer: Buffer<T>,
}
#[allow(unused)]
impl<T> UniformBuffer<T> {
    pub fn new() -> UniformBuffer<T> {
        let mut buffer = Buffer::new(BufferKind::Uniform);
        buffer.bind().alloc_elements(1);

        UniformBuffer { buffer }
    }
    pub fn bind(&mut self) -> UniformBufferBinder<T> {
        UniformBufferBinder::new(self)
    }
    pub fn mount(&self, location: usize) {
        unsafe {
            gl::BindBufferRange(
                BufferKind::Uniform.into(),
                location as u32,
                self.buffer.id,
                0,
                mem::size_of::<T>() as isize,
            );
        }
    }
}

#[allow(unused)]
pub struct UniformBufferBinder<'a, T: 'a>(BufferBinder<'a, T>);
#[allow(unused)]
impl<'a, T> UniformBufferBinder<'a, T> {
    pub fn new(uniform_buffer: &mut UniformBuffer<T>) -> UniformBufferBinder<T> {
        UniformBufferBinder(uniform_buffer.buffer.bind())
    }
    pub fn update_with(&mut self, data: &T) {
        unsafe {
            self.0
                .buffer_raw_sub_data(0, mem::size_of::<T>(), data as *const _ as *const _);
        }
    }
}

#[allow(unused)]
#[derive(Debug, Clone, Copy)]
pub enum DrawMode {
    Points,
    LineStrip,
    LineLoop,
    Lines,
    TriangleStrip,
    TriangleFan,
    Triangles,
}

impl Into<u32> for DrawMode {
    fn into(self) -> u32 {
        match self {
            DrawMode::Points => gl::POINTS,
            DrawMode::LineStrip => gl::LINE_STRIP,
            DrawMode::LineLoop => gl::LINE_LOOP,
            DrawMode::Lines => gl::LINES,
            DrawMode::TriangleStrip => gl::TRIANGLE_STRIP,
            DrawMode::TriangleFan => gl::TRIANGLE_FAN,
            DrawMode::Triangles => gl::TRIANGLES,
        }
    }
}

pub struct VertexArrayBinder<'a>(&'a mut VertexArray);
impl<'a> VertexArrayBinder<'a> {
    pub fn new(vao: &'a mut VertexArray) -> VertexArrayBinder<'a> {
        unsafe {
            gl::BindVertexArray(vao.0);
        }
        VertexArrayBinder(vao)
    }
    pub fn draw_arrays<T>(
        &mut self,
        _fbo: &T,
        mode: DrawMode,
        first: usize,
        count: usize,
    ) -> &VertexArrayBinder
    where
        T: FramebufferBinderDrawer,
    {
        unsafe {
            gl::DrawArrays(mode.into(), first as i32, count as i32);
        }
        self
    }
    pub fn draw_arrays_instanced<T>(
        &mut self,
        _fbo: &T,
        mode: DrawMode,
        first: usize,
        count: usize,
        instances: usize,
    ) -> &VertexArrayBinder
    where
        T: FramebufferBinderDrawer,
    {
        unsafe {
            gl::DrawArraysInstanced(mode.into(), first as i32, count as i32, instances as i32);
        }
        self
    }
    #[allow(unused)]
    pub fn draw_elements<T>(
        &mut self,
        _fbo: T,
        mode: DrawMode,
        count: usize,
        typ2: u32,
        xx: usize,
    ) -> &VertexArrayBinder
    where
        T: FramebufferBinderDrawer,
    {
        unsafe {
            gl::DrawElements(
                mode.into(),
                count as i32,
                typ2,
                ptr::null::<os::raw::c_void>().add(xx),
            );
        }
        self
    }
    unsafe fn attrib_pointer(
        &self,
        index: usize,
        size: usize,
        typ: GlType,
        normalized: bool,
        stride: usize,
        offset: usize,
    ) -> &VertexArrayBinder {
        gl::VertexAttribPointer(
            index as u32,
            size as i32,
            typ.into(),
            if normalized { gl::TRUE } else { gl::FALSE },
            stride as i32,
            ptr::null::<os::raw::c_void>().add(offset) as *const _,
        );
        self
    }
    unsafe fn enable_attrib_array(&self, index: usize) -> &VertexArrayBinder {
        gl::EnableVertexAttribArray(index as u32);
        self
    }
    pub fn attrib(
        &self,
        index: usize,
        size: usize,
        stride: usize,
        offset: usize,
    ) -> &VertexArrayBinder {
        unsafe {
            self.attrib_pointer(index, size, GlType::Float, false, stride, offset);
            self.enable_attrib_array(index);
        }
        self
    }
    pub fn vbo_attrib<T>(
        &self,
        _vbo: &VboBinder<T>,
        index: usize,
        size: usize,
        offset: usize,
    ) -> &VertexArrayBinder {
        self.attrib(index, size, mem::size_of::<T>(), offset);
        self
    }
    pub fn attrib_divisor(&self, index: usize, divisor: usize) -> &VertexArrayBinder {
        unsafe {
            gl::VertexAttribDivisor(index as u32, divisor as u32);
        }
        self
    }
}
impl<'a> Drop for VertexArrayBinder<'a> {
    fn drop(&mut self) {
        unsafe {
            gl::BindVertexArray(0);
        }
    }
}

pub struct VboBinder<'a, V: 'a>(BufferBinder<'a, V>);
impl<'a, V> VboBinder<'a, V> {
    pub fn new(vbo: &'a mut VertexBuffer<V>) -> VboBinder<'a, V> {
        VboBinder(vbo.buffer.bind())
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn buffer_data(&mut self, data: &[V]) {
        self.0.buffer_data(data)
    }
}
