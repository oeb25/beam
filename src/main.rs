#![feature(fs_read_write, stmt_expr_attributes, transpose_result)]

extern crate cgmath;
extern crate genmesh;
extern crate gl;
extern crate glutin;
extern crate image;
extern crate obj;
extern crate time;

use cgmath::{InnerSpace, Rad};

use glutin::GlContext;

use image::GenericImage;

use time::{PreciseTime, Duration};

use std::{ffi, fs, mem, os, ptr, borrow::Cow, collections::HashMap, marker::PhantomData,
          path::Path, rc::Rc};

macro_rules! offset_of {
    ($ty:ty, $field:ident) => {
        #[allow(unused_unsafe)]
        unsafe {
            &(*(0 as *const $ty)).$field as *const _ as usize
        }
    };
}

#[allow(unused_macros)]
macro_rules! offset_ptr {
    ($ty:ty, $field:ident) => {
        ptr::null::<os::raw::c_void>().add(offset_of!($ty, $field))
    };
}

macro_rules! size_of {
    ($ty:ty, $field:ident) => {
        #[allow(unused_unsafe)]
        unsafe {
            mem::size_of_val(&(*(0 as *const $ty)).$field)
        }
    };
}

#[allow(unused)]
#[derive(Debug, Clone, Copy)]
enum GlError {
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
    fn check() -> Result<(), GlError> {
        let err = unsafe { gl::GetError() };
        use GlError::*;
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
enum ShaderType {
    Vertex,
    Fragment,
    Geometry,
}
impl Into<u32> for ShaderType {
    fn into(self) -> u32 {
        match self {
            ShaderType::Vertex => gl::VERTEX_SHADER,
            ShaderType::Fragment => gl::FRAGMENT_SHADER,
            ShaderType::Geometry => gl::GEOMETRY_SHADER,
        }
    }
}
struct Shader(ShaderType, gl::types::GLuint);
impl Shader {
    fn new(src: &str, shader_type: ShaderType) -> Result<Shader, ()> {
        let shader_id = unsafe {
            let shader_id = gl::CreateShader(shader_type.into());
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
                println!("Error while compiling shader of type {:?}", shader_type);
                for line in error_msg.unwrap().lines() {
                    println!("{}", line);
                }
                panic!();
            }

            shader_id
        };

        Ok(Shader(shader_type, shader_id))
    }
}
impl Drop for Shader {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteShader(self.1);
        }
    }
}

struct VertexShader(Shader);
impl VertexShader {
    fn new(src: &str) -> Result<VertexShader, ()> {
        let shader = Shader::new(src, ShaderType::Vertex)?;
        Ok(VertexShader(shader))
    }
}

struct FragmentShader(Shader);
impl FragmentShader {
    fn new(src: &str) -> Result<FragmentShader, ()> {
        let shader = Shader::new(src, ShaderType::Fragment)?;
        Ok(FragmentShader(shader))
    }
}

struct GeometryShader(Shader);
impl GeometryShader {
    fn new(src: &str) -> Result<GeometryShader, ()> {
        let shader = Shader::new(src, ShaderType::Geometry)?;
        Ok(GeometryShader(shader))
    }
}

struct UniformLocation<'a>(gl::types::GLint, std::marker::PhantomData<&'a u8>);
impl<'a> UniformLocation<'a> {
    fn new(loc: gl::types::GLint) -> UniformLocation<'a> {
        UniformLocation(loc, std::marker::PhantomData)
    }
}

struct UniformBlockIndex<'a>(gl::types::GLuint, std::marker::PhantomData<&'a u8>);
impl<'a> UniformBlockIndex<'a> {
    fn new(loc: gl::types::GLuint) -> UniformBlockIndex<'a> {
        UniformBlockIndex(loc, std::marker::PhantomData)
    }
}

struct Program {
    id: gl::types::GLuint,
}
impl Program {
    fn new_from_disk(vs_path: &str, gs_path: Option<&str>, fs_path: &str) -> Result<Program, ()> {
        let vs_src = fs::read_to_string(vs_path).expect("unable to load vertex shader");
        let gs_src = gs_path.map(|gs_path| {
            fs::read_to_string(gs_path)
                .expect("unable to load geometry shader")
                .into()
        });
        let fs_src = fs::read_to_string(fs_path).expect("unable to load fragment shader");

        Program::new(vs_src.into(), gs_src, fs_src.into())
    }
    fn new(vs_src: Cow<str>, gs_src: Option<Cow<str>>, fs_src: Cow<str>) -> Result<Program, ()> {
        let vs = VertexShader::new(&vs_src)?;
        let gs = gs_src
            .map(|gs_src| GeometryShader::new(&gs_src))
            .transpose()?;
        let fs = FragmentShader::new(&fs_src)?;

        let program_id = unsafe {
            let program_id = gl::CreateProgram();

            gl::AttachShader(program_id, (vs.0).1);
            if let Some(gs) = gs {
                gl::AttachShader(program_id, (gs.0).1);
            }
            gl::AttachShader(program_id, (fs.0).1);

            gl::LinkProgram(program_id);

            let mut success = mem::uninitialized();
            gl::GetProgramiv(program_id, gl::LINK_STATUS, &mut success);
            if success == 0 {
                let mut error_log_size = 512;
                let mut buffer: Vec<u8> = Vec::with_capacity(error_log_size as usize);
                gl::GetProgramInfoLog(
                    program_id,
                    error_log_size,
                    &mut error_log_size,
                    buffer.as_mut_ptr() as *mut _,
                );
                buffer.set_len(error_log_size as usize);
                let error_msg = String::from_utf8(buffer);
                for line in error_msg.unwrap().lines() {
                    println!("{}", line);
                }
                panic!();
            }

            program_id
        };

        Ok(Program { id: program_id })
    }
    fn bind(&mut self) -> ProgramBinding {
        ProgramBinding::new(self)
    }
}
impl Drop for Program {
    fn drop(&mut self) {
        unsafe { gl::DeleteProgram(self.id) }
    }
}
struct ProgramBinding<'a>(&'a mut Program);
impl<'a> ProgramBinding<'a> {
    fn new(program: &'a mut Program) -> ProgramBinding<'a> {
        unsafe {
            gl::UseProgram(program.id);
        }
        ProgramBinding(program)
    }
    fn get_uniform_location<'b>(&'a self, name: &str) -> UniformLocation<'b> {
        let loc =
            unsafe { gl::GetUniformLocation(self.0.id, ffi::CString::new(name).unwrap().as_ptr()) };
        UniformLocation::new(loc)
    }
    fn bind_mat4_<'b>(&'b self, loc: UniformLocation<'b>, mat: &Mat4) -> &ProgramBinding {
        unsafe {
            gl::UniformMatrix4fv(loc.0, 1, gl::FALSE, mat as *const _ as *const _);
        }
        self
    }
    fn bind_mat4(&self, name: &str, mat: &Mat4) -> &ProgramBinding {
        let loc = self.get_uniform_location(name);
        self.bind_mat4_(loc, mat);
        self
    }
    fn bind_int(&self, name: &str, i: i32) -> &ProgramBinding {
        let loc = self.get_uniform_location(name);
        unsafe {
            gl::Uniform1i(loc.0, i);
        }
        self
    }
    fn bind_uint(&self, name: &str, i: u32) -> &ProgramBinding {
        let loc = self.get_uniform_location(name);
        unsafe {
            gl::Uniform1ui(loc.0, i);
        }
        self
    }
    fn bind_bool(&self, name: &str, i: bool) -> &ProgramBinding {
        self.bind_uint(name, if i { 1 } else { 0 });
        self
    }
    fn bind_texture(&self, name: &str, texture: &Texture, slot: TextureSlot) -> &ProgramBinding {
        texture.bind_to(slot);
        self.bind_int(name, slot.into())
    }
    fn bind_float(&self, name: &str, f: f32) -> &ProgramBinding {
        let loc = self.get_uniform_location(name);
        unsafe {
            gl::Uniform1f(loc.0, f);
        }
        self
    }
    fn bind_vec3(&self, name: &str, v: &V3) -> &ProgramBinding {
        let loc = self.get_uniform_location(name);
        unsafe {
            gl::Uniform3f(loc.0, v.x, v.y, v.z);
        }
        self
    }
    fn get_uniform_block_index(&'a self, name: &str) -> UniformBlockIndex<'a> {
        let loc = unsafe {
            gl::GetUniformBlockIndex(self.0.id, ffi::CString::new(name).unwrap().as_ptr())
        };
        UniformBlockIndex::new(loc)
    }
    fn uniform_block_binding(&self, name: &str, index: usize) -> &ProgramBinding {
        let block_index = self.get_uniform_block_index(name);
        unsafe {
            gl::UniformBlockBinding(self.0.id, block_index.0, index as u32);
        }
        self
    }
    fn bind_light(&self, name: &str, light: &Light) -> &ProgramBinding {
        let ext = |e| format!("{}.{}", name, e);

        match light {
            Light::Directional(DirectionalLight {
                ambient,
                diffuse,
                specular,
                direction,
            }) => {
                self.bind_vec3(&ext("ambient"), ambient);
                self.bind_vec3(&ext("diffuse"), diffuse);
                self.bind_vec3(&ext("specular"), specular);

                self.bind_vec3(&ext("direction"), direction);
            }
            Light::Point(PointLight {
                position,
                ambient,
                diffuse,
                specular,
                constant,
                linear,
                quadratic,
            }) => {
                self.bind_vec3(&ext("ambient"), ambient);
                self.bind_vec3(&ext("diffuse"), diffuse);
                self.bind_vec3(&ext("specular"), specular);

                self.bind_vec3(&ext("position"), position);

                self.bind_float(&ext("constant"), *constant);
                self.bind_float(&ext("linear"), *linear);
                self.bind_float(&ext("quadratic"), *quadratic);
            }
            Light::Spot(SpotLight {
                position,
                ambient,
                diffuse,
                specular,
                direction,
                cut_off,
                outer_cut_off,
            }) => {
                self.bind_vec3(&ext("ambient"), ambient);
                self.bind_vec3(&ext("diffuse"), diffuse);
                self.bind_vec3(&ext("specular"), specular);

                self.bind_vec3(&ext("position"), position);
                self.bind_vec3(&ext("direction"), direction);

                self.bind_float(&ext("cutOff"), cut_off.0.cos());
                self.bind_float(&ext("outerCutOff"), outer_cut_off.0.cos());
            }
        }
        self
    }
    fn bind_lights(
        &self,
        (directional_num, directional_array): (&str, &str),
        (point_num, point_array): (&str, &str),
        (spot_num, spot_array): (&str, &str),
        lights: &[Light],
    ) -> &ProgramBinding {
        let mut n_directional = 0;
        let mut n_point = 0;
        let mut n_spot = 0;
        for light in lights {
            let name = match light {
                Light::Directional(_) => {
                    let name = format!("{}[{}]", directional_array, n_directional);
                    n_directional += 1;
                    name
                }
                Light::Point(_) => {
                    let name = format!("{}[{}]", point_array, n_point);
                    n_point += 1;
                    name
                }
                Light::Spot(_) => {
                    let name = format!("{}[{}]", spot_array, n_spot);
                    n_spot += 1;
                    name
                }
            };
            self.bind_light(&name, light);
        }
        self.bind_int(directional_num, n_directional);
        self.bind_int(point_num, n_point);
        self.bind_int(spot_num, n_spot);
        self
    }
}

type V2 = cgmath::Vector2<f32>;
type V3 = cgmath::Vector3<f32>;
type V4 = cgmath::Vector4<f32>;
type Mat4 = cgmath::Matrix4<f32>;

trait Vertexable {
    // (pos, norm, tex, trangent, bitangent)
    fn sizes() -> (usize, usize, usize, usize, usize);
    fn offsets() -> (usize, usize, usize, usize, usize);
}

struct Vertex {
    pos: V3,
    norm: V3,
    tex: V2,
    tangent: V3,
    bitangent: V3,
}

impl Vertexable for Vertex {
    fn sizes() -> (usize, usize, usize, usize, usize) {
        let float_size = mem::size_of::<f32>();
        (
            size_of!(Vertex, pos) / float_size,
            size_of!(Vertex, norm) / float_size,
            size_of!(Vertex, tex) / float_size,
            size_of!(Vertex, tangent) / float_size,
            size_of!(Vertex, bitangent) / float_size,
        )
    }
    fn offsets() -> (usize, usize, usize, usize, usize) {
        (
            offset_of!(Vertex, pos),
            offset_of!(Vertex, norm),
            offset_of!(Vertex, tex),
            offset_of!(Vertex, tangent),
            offset_of!(Vertex, bitangent),
        )
    }
}

struct Camera {
    pos: V3,
    fov: Rad<f32>,
    aspect: f32,
    yaw: f32,
    pitch: f32,
}

impl Camera {
    fn new(pos: V3, fov: Rad<f32>, aspect: f32) -> Camera {
        Camera {
            pos,
            fov,
            aspect,
            yaw: 0.0,
            pitch: 0.0,
        }
    }

    fn up(&self) -> V3 {
        V3::new(0.0, 1.0, 0.0)
    }
    fn front(&self) -> V3 {
        let (ps, pc) = self.pitch.sin_cos();
        let (ys, yc) = self.yaw.sin_cos();

        V3::new(pc * yc, ps, pc * ys).normalize()
    }
    #[allow(unused)]
    fn front_look_at(&self, target: &V3) -> V3 {
        (target - self.pos).normalize()
    }
    fn get_view(&self) -> Mat4 {
        let origo = cgmath::Point3::new(0.0, 0.0, 0.0);
        Mat4::look_at(origo + self.pos, origo + self.pos + self.front(), self.up())
    }
    #[allow(unused)]
    fn get_view_look_at(&self, target: &V3) -> Mat4 {
        let origo = cgmath::Point3::new(0.0, 0.0, 0.0);
        Mat4::look_at(
            origo + self.pos,
            origo + self.pos + self.front_look_at(target),
            self.up(),
        )
    }
    fn get_projection(&self) -> Mat4 {
        cgmath::PerspectiveFov {
            fovy: self.fov,
            aspect: self.aspect,
            near: 0.01,
            far: 100.0,
        }.into()
    }
}

#[allow(unused)]
#[derive(Debug, Clone, Copy)]
enum TextureSlot {
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
        use TextureSlot::*;
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
        use TextureSlot::*;
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
            _ => unimplemented!(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum TextureKind {
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
struct Texture {
    id: gl::types::GLuint,
    kind: TextureKind,
}

impl Texture {
    #[allow(unused)]
    unsafe fn gen(n: usize, kind: TextureKind) -> Vec<Texture> {
        let mut tex_ids = Vec::with_capacity(n);
        for _ in 0..n {
            tex_ids.push(mem::uninitialized());
        }
        gl::GenTextures(n as i32, &mut tex_ids[0] as *mut _);
        tex_ids.into_iter().map(|id| Texture { id, kind }).collect()
    }
    fn new(kind: TextureKind) -> Texture {
        let id = unsafe {
            let mut tex_id = mem::uninitialized();
            gl::GenTextures(1, &mut tex_id);
            tex_id
        };

        Texture { id, kind }
    }
    fn bind(&self) -> TextureBinder {
        TextureBinder::new(self)
    }
    fn bind_to(&self, slot: TextureSlot) {
        unsafe { gl::ActiveTexture(slot.into()); }
        self.bind();
        GlError::check().unwrap();
    }
}
impl Drop for Texture {
    fn drop(&mut self) {
        unsafe { gl::DeleteTextures(1, &self.id as *const _) }
    }
}

#[allow(unused)]
#[derive(Debug, Clone, Copy)]
enum TextureTarget {
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
enum TextureInternalFormat {
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
enum TextureFormat {
    Red,  // GL_RED,
    Rg,   // GL_RG,
    Rgb,  // GL_RGB,
    Bgr,  // GL_BGR,
    Rgba, // GL_RGBA,
    Bgra, // GL_BGRA,
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
        }
    }
}

#[allow(unused)]
enum GlType {
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

struct TextureBinder<'a>(&'a Texture);
impl<'a> TextureBinder<'a> {
    fn new(texture: &Texture) -> TextureBinder {
        unsafe {
            gl::BindTexture(texture.kind.into(), texture.id);
        }
        TextureBinder(texture)
    }
    fn empty(
        &self,
        target: TextureTarget,
        level: usize,
        internal_format: TextureInternalFormat,
        width: u32,
        height: u32,
        format: TextureFormat,
        typ: GlType
    ) -> &TextureBinder {
        unsafe {
            self.image_2d(target, level, internal_format, width, height, format, typ, ptr::null());
        }
        self
    }
    unsafe fn image_2d(
        &self,
        target: TextureTarget,
        level: usize,
        internal_format: TextureInternalFormat,
        width: u32,
        height: u32,
        format: TextureFormat,
        typ: GlType,
        data: *const os::raw::c_void,
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
    fn load_image(
        &self,
        target: TextureTarget,
        internal_format: TextureInternalFormat,
        format: TextureFormat,
        img: &image::DynamicImage,
    ) {
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
                })[0] as *const _ as *const _,
            );
        }
    }
    fn parameter_int(&self, pname: TextureParameter, param: i32) -> &TextureBinder {
        unsafe {
            gl::TexParameteri(self.0.kind.into(), pname.into(), param);
        }
        self
    }
}

#[allow(unused)]
enum TextureParameter {
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
enum Attachment {
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

#[derive(Debug)]
struct Framebuffer {
    id: gl::types::GLuint
}
impl Framebuffer {
    fn new() -> Framebuffer {
        let id = unsafe {
            let mut fb_id = mem::uninitialized();
            gl::GenFramebuffers(1, &mut fb_id);
            fb_id
        };

        Framebuffer { id }
    }
    fn bind(&mut self) -> FramebufferBinder {
        FramebufferBinder::new(self)
    }
}
impl Drop for Framebuffer {
    fn drop(&mut self) {
        unsafe { gl::DeleteFramebuffers(1, &self.id as *const _); }
    }
}

#[derive(Debug)]
struct FramebufferBinder<'a>(&'a mut Framebuffer);
impl<'a> FramebufferBinder<'a> {
    fn new(fb: &mut Framebuffer) -> FramebufferBinder {
        unsafe { gl::BindFramebuffer(gl::FRAMEBUFFER, fb.id); }
        FramebufferBinder(fb)
    }
    fn texture_2d(
        &self,
        attachment: Attachment,
        textarget: TextureTarget,
        texture: &Texture,
        level: usize,
    ) -> &FramebufferBinder {
        unsafe {
            gl::FramebufferTexture2D(
                gl::FRAMEBUFFER,
                attachment.into(),
                textarget.into(),
                texture.id,
                level as i32,
            );
        }
        self
    }
    fn renderbuffer(
        &self,
        attachment: Attachment,
        renderbuffer: &Renderbuffer,
    ) -> &FramebufferBinder {
        unsafe {
            gl::FramebufferRenderbuffer(
                gl::FRAMEBUFFER,
                attachment.into(),
                gl::RENDERBUFFER,
                renderbuffer.id,
            );
        }
        self
    }
    fn check_status(&self) -> Result<(), ()> {
        let status = unsafe {
            gl::CheckFramebufferStatus(gl::FRAMEBUFFER)
        };
        if status != gl::FRAMEBUFFER_COMPLETE {
            Err(())
        } else {
            Ok(())
        }
    }
}
impl<'a> Drop for FramebufferBinder<'a> {
    fn drop(&mut self) {
        unsafe { gl::BindFramebuffer(gl::FRAMEBUFFER, 0); }
    }
}

struct Renderbuffer {
    id: gl::types::GLuint
}
impl Renderbuffer {
    fn new() -> Renderbuffer {
        let id = unsafe {
            let mut fb_id = mem::uninitialized();
            gl::GenRenderbuffers(1, &mut fb_id);
            fb_id
        };

        Renderbuffer { id }
    }
    fn bind(&mut self) -> RenderbufferBinder {
        RenderbufferBinder::new(self)
    }
}
impl Drop for Renderbuffer {
    fn drop(&mut self) {
        unsafe { gl::DeleteRenderbuffers(1, &self.id as *const _); }
    }
}

struct RenderbufferBinder<'a>(&'a mut Renderbuffer);
impl<'a> RenderbufferBinder<'a> {
    fn new(fb: &mut Renderbuffer) -> RenderbufferBinder {
        unsafe { gl::BindRenderbuffer(gl::RENDERBUFFER, fb.id); }
        RenderbufferBinder(fb)
    }
    fn storage(&self, internal_format: TextureInternalFormat, width: u32, height: u32) {
        unsafe { gl::RenderbufferStorage(gl::RENDERBUFFER, internal_format.into(), width as i32, height as i32); }
    }
}
impl<'a> Drop for RenderbufferBinder<'a> {
    fn drop(&mut self) {
        unsafe { gl::BindRenderbuffer(gl::RENDERBUFFER, 0); }
    }
}

#[allow(unused)]
#[derive(Debug, Clone, Copy)]
enum ImageKind {
    Diffuse,
    Ambient,
    Specular,
    Reflection,
    CubeMap,
    NormalMap,
}

#[derive(Debug)]
struct Image {
    texture: Texture,
    path: String,
    kind: ImageKind,
}
impl Image {
    fn new_from_disk(path: &str, kind: ImageKind) -> Image {
        let img = image::open(path).expect(&format!("unable to read {}", path));
        let tex_kind = match kind {
            ImageKind::Diffuse
            | ImageKind::Ambient
            | ImageKind::Specular
            | ImageKind::NormalMap
            | ImageKind::Reflection => TextureKind::Texture2d,
            ImageKind::CubeMap => unimplemented!(),
        };
        let texture = Texture::new(tex_kind);
        texture.bind()
            .parameter_int(TextureParameter::WrapS, gl::REPEAT as i32)
            .parameter_int(TextureParameter::WrapT, gl::REPEAT as i32)
            .parameter_int(TextureParameter::MinFilter, gl::LINEAR as i32)
            .parameter_int(TextureParameter::MagFilter, gl::LINEAR as i32)
            .load_image(
                TextureTarget::Texture2d,
                TextureInternalFormat::Srgb,
                TextureFormat::Rgb,
                &img,
            );

        Image {
            texture,
            path: path.to_string(),
            kind,
        }
    }
}

struct VertexArray(gl::types::GLuint);

#[derive(Debug, Clone, Copy)]
enum BufferKind {
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

struct Buffer<T> {
    kind: BufferKind,
    id: gl::types::GLuint,
    buffer_size: Option<usize>,
    phantom: PhantomData<T>,
}

impl<T> Buffer<T> {
    fn new(kind: BufferKind) -> Buffer<T> {
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
    fn bind(&mut self) -> BufferBinder<T> {
        BufferBinder::new(self)
    }
    fn len(&self) -> usize {
        self.size() / mem::size_of::<T>()
    }
    fn size(&self) -> usize {
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

struct BufferBinder<'a, T: 'a>(&'a mut Buffer<T>);
impl<'a, T> BufferBinder<'a, T> {
    fn new(buffer: &'a mut Buffer<T>) -> BufferBinder<'a, T> {
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
    fn alloc(&mut self, size: usize) {
        unsafe {
            self.buffer_raw_data(size, ptr::null());
        }
    }
    fn alloc_elements(&mut self, num_elements: usize) {
        self.alloc(num_elements * mem::size_of::<T>());
    }
    fn buffer_data(&mut self, data: &[T]) {
        let size = data.len() * mem::size_of::<T>();
        let data_ptr = &data[0] as *const _ as *const _;
        unsafe {
            self.buffer_raw_data(size, data_ptr);
        }
    }
    fn len(&self) -> usize {
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

struct VertexBuffer<T> {
    buffer: Buffer<T>,
}
struct Ebo<T>(Buffer<T>);

impl VertexArray {
    fn new() -> VertexArray {
        unsafe {
            let mut vao = mem::uninitialized();
            gl::GenVertexArrays(1, &mut vao);
            VertexArray(vao)
        }
    }
    fn bind(&mut self) -> VertexArrayBinder {
        VertexArrayBinder::new(self)
    }
}

impl<T> VertexBuffer<T> {
    fn new() -> VertexBuffer<T> {
        VertexBuffer {
            buffer: Buffer::new(BufferKind::Array),
        }
    }
    fn from_data(data: &[T]) -> VertexBuffer<T> {
        let mut vbo = VertexBuffer::new();
        vbo.bind().buffer_data(data);
        vbo
    }
    #[allow(unused)]
    fn len(&self) -> usize {
        self.buffer.len()
    }
    fn bind(&mut self) -> VboBinder<T> {
        VboBinder::new(self)
    }
}

impl<T> Ebo<T> {
    fn new() -> Ebo<T> {
        Ebo(Buffer::new(BufferKind::ElementArray))
    }
}

struct UniformBuffer<T> {
    buffer: Buffer<T>,
}
#[allow(unused)]
impl<T> UniformBuffer<T> {
    fn new() -> UniformBuffer<T> {
        let mut buffer = Buffer::new(BufferKind::Uniform);
        buffer.bind().alloc_elements(1);

        UniformBuffer { buffer }
    }
    fn bind(&mut self) -> UniformBufferBinder<T> {
        UniformBufferBinder::new(self)
    }
    fn mount(&self, location: usize) {
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
struct UniformBufferBinder<'a, T: 'a>(BufferBinder<'a, T>);
#[allow(unused)]
impl<'a, T> UniformBufferBinder<'a, T> {
    fn new(uniform_buffer: &mut UniformBuffer<T>) -> UniformBufferBinder<T> {
        UniformBufferBinder(uniform_buffer.buffer.bind())
    }
    fn update_with(&mut self, data: &T) {
        unsafe {
            self.0
                .buffer_raw_sub_data(0, mem::size_of::<T>(), data as *const _ as *const _);
        }
    }
}

#[allow(unused)]
#[derive(Debug, Clone, Copy)]
enum DrawMode {
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

struct VertexArrayBinder<'a>(&'a mut VertexArray);
impl<'a> VertexArrayBinder<'a> {
    fn new(vao: &'a mut VertexArray) -> VertexArrayBinder<'a> {
        unsafe {
            gl::BindVertexArray(vao.0);
        }
        VertexArrayBinder(vao)
    }
    fn draw_arrays(&mut self, mode: DrawMode, first: usize, count: usize) -> &VertexArrayBinder {
        unsafe {
            gl::DrawArrays(mode.into(), first as i32, count as i32);
        }
        self
    }
    fn draw_arrays_instanced(
        &mut self,
        mode: DrawMode,
        first: usize,
        count: usize,
        instances: usize,
    ) -> &VertexArrayBinder {
        unsafe {
            gl::DrawArraysInstanced(mode.into(), first as i32, count as i32, instances as i32);
        }
        self
    }
    #[allow(unused)]
    fn draw_elements(&mut self, mode: DrawMode, count: usize, typ2: u32, xx: usize) -> &VertexArrayBinder {
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
    fn attrib(&self, index: usize, size: usize, stride: usize, offset: usize) -> &VertexArrayBinder {
        unsafe {
            self.attrib_pointer(index, size, GlType::Float, false, stride, offset);
            self.enable_attrib_array(index);
        }
        self
    }
    fn vbo_attrib<T>(&self, _vbo: &VboBinder<T>, index: usize, size: usize, offset: usize) -> &VertexArrayBinder {
        self.attrib(index, size, mem::size_of::<T>(), offset);
        self
    }
    fn attrib_divisor(&self, index: usize, divisor: usize) -> &VertexArrayBinder {
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

struct VboBinder<'a, V: 'a>(BufferBinder<'a, V>);
impl<'a, V> VboBinder<'a, V> {
    fn new(vbo: &'a mut VertexBuffer<V>) -> VboBinder<'a, V> {
        VboBinder(vbo.buffer.bind())
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn buffer_data(&mut self, data: &[V]) {
        self.0.buffer_data(data)
    }
}

// struct EboBinder<'a: 'b, 'b>(&'b VertexArrayBinder<'a>, &'b mut Ebo);
// impl<'a, 'b> EboBinder<'a, 'b> {
//     fn new(vao_binder: &'b VertexArrayBinder<'a>, ebo: &'b mut Ebo) -> EboBinder<'a, 'b> {
//         unsafe { gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo.0) }
//         EboBinder(vao_binder, ebo)
//     }
//     fn buffer_data<T>(&self, data: &[T]) {
//         unsafe {
//             gl::BufferData(
//                 gl::ELEMENT_ARRAY_BUFFER,
//                 (data.len() * mem::size_of::<T>()) as isize,
//                 &data[0] as *const _ as *const _,
//                 gl::STATIC_DRAW,
//             );
//         }
//     }
// }
// impl<'a, 'b> Drop for EboBinder<'a, 'b> {
//     fn drop(&mut self) {
//         unsafe { gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, 0); }
//     }
// }

#[allow(unused)]
struct Mesh<V> {
    vertices: Vec<V>,
    indecies: Option<Vec<usize>>,
    textures: Vec<Rc<Image>>,

    vao: VertexArray,
    vbo: VertexBuffer<V>,
    ebo: Option<Ebo<usize>>,
}

impl<V: Vertexable> Mesh<V> {
    fn new(vertices: Vec<V>, indecies: Option<Vec<usize>>, textures: Vec<Rc<Image>>) -> Mesh<V> {
        let mut vao = VertexArray::new();
        let mut vbo = VertexBuffer::from_data(&vertices);
        let ebo = if indecies.is_some() {
            Some(Ebo::new())
        } else {
            None
        };
        {
            // if let Some(ref indecies) = &indecies {
            //     if let Some(ref mut ebo) = &mut ebo {
            //         let ebo_binder = vao_binder.bind_ebo(ebo);
            //         ebo_binder.buffer_data(&indecies);
            //     }
            // }

            let (pos_s, norm_s, tex_s, tangent_s, _) = V::sizes();
            let (pos_o, norm_o, tex_o, tangent_o, _) = V::offsets();

            let vbo_binder = vbo.bind();

            vao.bind()
                .vbo_attrib(&vbo_binder, 0, pos_s, pos_o)
                .vbo_attrib(&vbo_binder, 1, norm_s, norm_o)
                .vbo_attrib(&vbo_binder, 2, tex_s, tex_o)
                .vbo_attrib(&vbo_binder, 3, tangent_s, tangent_o);
        }

        Mesh {
            vertices,
            indecies,
            textures,
            vao,
            vbo,
            ebo,
        }
    }

    fn bind(&mut self) -> MeshBinding<V> {
        MeshBinding(self)
    }
}

struct MeshBinding<'a, V: 'a>(&'a mut Mesh<V>);
impl<'a, V> MeshBinding<'a, V> {
    fn bind_textures(&self, program: &ProgramBinding) {
        let mut diffuse_n = 0;
        let mut ambient_n = 0;
        let mut specular_n = 0;
        let mut reflection_n = 0;
        let mut normal_n = 0;
        for (i, tex) in self.0.textures.iter().enumerate() {
            let (name, number) = match tex.kind {
                ImageKind::Diffuse => {
                    diffuse_n += 1;
                    ("diffuse", diffuse_n)
                }
                ImageKind::Ambient => {
                    ambient_n += 1;
                    ("ambient", ambient_n)
                }
                ImageKind::Specular => {
                    specular_n += 1;
                    ("specular", specular_n)
                }
                ImageKind::Reflection => {
                    reflection_n += 1;
                    ("reflection", reflection_n)
                }
                ImageKind::NormalMap => {
                    normal_n += 1;
                    ("normal", normal_n)
                }
                ImageKind::CubeMap => unimplemented!(),
            };

            assert_eq!(number, 1);

            program.bind_texture(&format!("tex_{}{}", name, number), &tex.texture, i.into());
        }
        program.bind_bool("useNormalMap", normal_n > 0);
    }
    fn draw(&mut self, program: &ProgramBinding) {
        self.bind_textures(program);

        self.0
            .vao
            .bind()
            .draw_arrays(DrawMode::Triangles, 0, self.0.vertices.len());
    }
    fn draw_instanced(&mut self, program: &ProgramBinding, transforms: &VboBinder<Mat4>) {
        self.bind_textures(program);

        let mut vao = self.0.vao.bind();
        let offset = 4;
        let width = 4;
        for i in 0..width {
            let index = i + offset;
            vao
                .vbo_attrib(&transforms, index, width, width * i * mem::size_of::<f32>())
                .attrib_divisor(index, 1);
        }

        vao.draw_arrays_instanced(
            DrawMode::Triangles,
            0,
            self.0.vertices.len(),
            transforms.len(),
        );
    }
}

#[allow(unused)]
struct Model {
    meshes: Vec<Mesh<Vertex>>,
    texture_cache: HashMap<String, Rc<Image>>,
}
impl Model {
    fn new_from_disk(path: &str) -> Model {
        let path = Path::new(path);
        let mut raw_model = obj::Obj::load(path).unwrap();
        let _ = raw_model.load_mtls().unwrap();
        let obj::Obj {
            position,
            texture,
            normal,
            material_libs,
            objects,
            ..
        } = raw_model;

        println!("{:?}", position.len());
        println!("{:?}", texture.len());
        println!("{:?}", normal.len());
        println!("{:?}", material_libs);
        println!("{:?}", objects.len());

        let mut meshes = vec![];
        let mut texture_cache: HashMap<String, Rc<Image>> = HashMap::new();

        for mut o in objects.into_iter() {
            let mut vertices = vec![];
            let mut materials = vec![];
            macro_rules! add_tex {
                ($name:expr, $tex_kind:expr) => {{
                    let path = path.with_file_name($name);
                    let path_string = path.to_str().unwrap().to_string();
                    let tex: Rc<Image> = if texture_cache.contains_key(&path_string) {
                        texture_cache.get(&path_string).unwrap().clone()
                    } else {
                        let tex = Image::new_from_disk(path.to_str().unwrap(), $tex_kind);
                        let tex = Rc::new(tex);
                        texture_cache.insert(path_string, tex.clone());
                        tex
                    };
                    materials.push(tex);
                }};
            }
            for group in o.groups {
                if let Some(mat) = group.material {
                    println!("{:?}", mat);
                    mat.map_kd
                        .as_ref()
                        .map(|diff| add_tex!(diff, ImageKind::Diffuse));
                    mat.map_ks
                        .as_ref()
                        .map(|spec| add_tex!(spec, ImageKind::Specular));
                    // mat.map_ka.as_ref().map(|ambient| add_tex!(ambient, ImageKind::Ambient));
                    mat.map_ka
                        .as_ref()
                        .map(|refl| add_tex!(refl, ImageKind::Reflection));
                    mat.map_refl
                        .as_ref()
                        .map(|_| unimplemented!("REFLECTION MAP!"));
                    mat.map_bump
                        .as_ref()
                        .map(|bump| add_tex!(bump, ImageKind::NormalMap));
                }
                for ps in group.polys {
                    match ps {
                        genmesh::Polygon::PolyTri(genmesh::Triangle {
                            x: v1,
                            y: v2,
                            z: v3,
                        }) => {
                            let res =
                                [v1, v2, v3]
                                    .into_iter()
                                    .map(|v| {
                                        let obj::IndexTuple(vert, tex, norm) = v;
                                        let vert = position[*vert].into();
                                        let norm = norm.map(|i| normal[i].into())
                                            .unwrap_or(V3::new(0.0, 0.0, 0.0));
                                        let tex = tex.map(|i| texture[i].into())
                                            .unwrap_or(V2::new(0.0, 0.0));
                                        (vert, tex, norm)
                                    })
                                    .collect::<Vec<_>>();
                            let v1 = res[0];
                            let v2 = res[1];
                            let v3 = res[2];
                            let meh = [(v1, v2, v3), (v2, v1, v3), (v3, v1, v2)];
                            for ((vert, tex, norm), (avert, atex, _anorm), (bvert, btex, _bnorm)) in
                                meh.iter()
                            {
                                let pos = *vert;
                                let norm = *norm;
                                let tex = *tex;

                                // Tangets and bitangents

                                let delta_pos1 = avert - pos;
                                let delta_pos2 = bvert - pos;

                                let delta_uv1 = atex - tex;
                                let delta_uv2 = btex - tex;

                                let r =
                                    1.0 / (delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x);
                                let tangent =
                                    (delta_pos1 * delta_uv2.y - delta_pos2 * delta_uv1.y) * r;
                                let bitangent =
                                    (delta_pos2 * delta_uv1.x - delta_pos1 * delta_uv2.x) * r;

                                let v = Vertex {
                                    pos,
                                    norm,
                                    tex,
                                    tangent,
                                    bitangent,
                                };
                                vertices.push(v);
                            }
                        }
                        _ => unimplemented!(),
                    }
                }
            }

            let mesh = Mesh::new(vertices, None, materials);

            meshes.push(mesh);
        }

        Model {
            meshes,
            texture_cache,
        }
    }
    #[allow(unused)]
    fn draw(&mut self, program: &ProgramBinding) {
        for mut mesh in self.meshes.iter_mut() {
            mesh.bind().draw(&program);
        }
    }
    fn draw_instanced(&mut self, program: &ProgramBinding, offsets: &VboBinder<Mat4>) {
        for mut mesh in self.meshes.iter_mut() {
            mesh.bind().draw_instanced(&program, offsets);
        }
    }
}

struct CubeMapBuilder<T> {
    back: T,
    front: T,
    right: T,
    bottom: T,
    left: T,
    top: T,
}

impl<'a> CubeMapBuilder<&'a str> {
    fn load(self) -> Image {
        let texture = Texture::new(TextureKind::CubeMap);
        let mut sum_path = String::new();
        {
            let tex = texture.bind();

            let faces = [
                (TextureTarget::TextureCubeMapPositiveX, self.right),
                (TextureTarget::TextureCubeMapNegativeX, self.left),
                (TextureTarget::TextureCubeMapPositiveY, self.top),
                (TextureTarget::TextureCubeMapNegativeY, self.bottom),
                (TextureTarget::TextureCubeMapPositiveZ, self.front),
                (TextureTarget::TextureCubeMapNegativeZ, self.back),
            ];

            for (target, path) in faces.into_iter() {
                sum_path += path;
                let img = image::open(path).expect(&format!(
                    "failed to read texture {} while loading cubemap",
                    path
                ));

                tex.load_image(
                    *target,
                    TextureInternalFormat::Srgb,
                    TextureFormat::Rgb,
                    &img,
                );
            }

            tex
                .parameter_int(TextureParameter::MinFilter, gl::LINEAR as i32)
                .parameter_int(TextureParameter::MagFilter, gl::LINEAR as i32)
                .parameter_int(TextureParameter::WrapS, gl::CLAMP_TO_EDGE as i32)
                .parameter_int(TextureParameter::WrapT, gl::CLAMP_TO_EDGE as i32)
                .parameter_int(TextureParameter::WrapR, gl::CLAMP_TO_EDGE as i32);
        }

        Image {
            texture,
            path: sum_path,
            kind: ImageKind::CubeMap,
        }
    }
}

#[repr(C)]
struct DirectionalLight {
    ambient: V3,
    diffuse: V3,
    specular: V3,

    direction: V3,
}
#[repr(C)]
struct PointLight {
    ambient: V3,
    diffuse: V3,
    specular: V3,

    position: V3,

    constant: f32,
    linear: f32,
    quadratic: f32,
}
#[repr(C)]
struct SpotLight {
    ambient: V3,
    diffuse: V3,
    specular: V3,

    position: V3,
    direction: V3,

    cut_off: Rad<f32>,
    outer_cut_off: Rad<f32>,
}

#[allow(unused)]
enum Light {
    Directional(DirectionalLight),
    Point(PointLight),
    Spot(SpotLight),
}

#[derive(Default)]
struct Input {
    w: f32,
    a: f32,
    s: f32,
    d: f32,

    up: f32,
    down: f32,
    left: f32,
    right: f32,

    space: f32,
    shift: f32,
}

fn main() {
    let (screen_width, screen_height) = (800, 600);
    let mut events_loop = glutin::EventsLoop::new();
    let window = glutin::WindowBuilder::new()
        .with_title("Hello, world!")
        .with_dimensions(screen_width, screen_height);
    let context = glutin::ContextBuilder::new().with_vsync(true);
    let gl_window = glutin::GlWindow::new(window, context, &events_loop).unwrap();

    unsafe {
        gl_window.make_current().unwrap();
    }

    unsafe {
        gl::load_with(|symbol| gl_window.get_proc_address(symbol) as *const _);
        gl::ClearColor(0.0, 1.0, 0.0, 1.0);
    }

    let mut program = Program::new_from_disk(
        "shaders/shader.vs",
        // Some("shaders/shader.gs"),
        None,
        "shaders/shader.fs",
    ).unwrap();
    let mut normal_program = Program::new_from_disk(
        "shaders/normal.vs",
        Some("shaders/normal.gs"),
        "shaders/normal.fs",
    ).unwrap();
    let mut skybox_program =
        Program::new_from_disk("shaders/skybox.vs", None, "shaders/skybox.fs").unwrap();
    let mut hdr_program = Program::new_from_disk(
        "shaders/hdr.vs",
        None,
        "shaders/hdr.fs",
    ).unwrap();

    let skybox = CubeMapBuilder {
        back: "assets/skybox/back.jpg",
        front: "assets/skybox/front.jpg",
        right: "assets/skybox/right.jpg",
        bottom: "assets/skybox/bottom.jpg",
        left: "assets/skybox/left.jpg",
        top: "assets/skybox/top.jpg",
    }.load();
    let mut nanosuit = Model::new_from_disk("assets/nanosuit_reflection/nanosuit.obj");
    let mut cyborg = Model::new_from_disk("assets/cyborg/cyborg.obj");
    // let mut teapot = Model::new_from_disk("teapot.obj");

    let tex1 = Image::new_from_disk("assets/container2.png", ImageKind::Diffuse);
    let tex2 = Image::new_from_disk("assets/container2_specular.png", ImageKind::Specular);
    let (tex1, tex2) = (Rc::new(tex1), Rc::new(tex2));

    let mut cube_mesh = Mesh::new(cube_vertices(), None, vec![tex1, tex2]);
    let mut rect_mesh = Mesh::new(rect_verticies(), None, vec![]);

    let mut t: f32 = 0.0;

    unsafe {
        gl::Enable(gl::DEPTH_TEST);
        gl::Enable(gl::CULL_FACE);
        gl::Enable(gl::FRAMEBUFFER_SRGB);
    }

    let mut camera = Camera::new(
        V3::new(t.sin(), (t / 10.0).sin(), 1.0),
        Rad(std::f32::consts::PI / 2.0),
        (screen_width as f32) / (screen_height as f32),
    );

    let mut inputs = Input::default();

    let draw_normals = false;

    let mut running = true;
    let mut last_pos = None;
    let mut is = vec![];
    for i in 0..30 {
        for n in 0..i {
            let x = i as f32 / 2.0;
            let v = V3::new(n as f32 - x, -i as f32, i as f32 / 2.0) * 2.0;
            let v = Mat4::from_translation(v) * Mat4::from_angle_y(Rad(i as f32 - 1.0));
            is.push(v);
        }
    }
    println!("drawing {} nanosuits", is.len());
    let mut instances = VertexBuffer::from_data(&is);

    let mut hdr_fbo = Framebuffer::new();
    let color_buffer = Texture::new(TextureKind::Texture2d);
    let (w, h) = gl_window.get_inner_size().unwrap();
    let (w, h) = (w * 2, h * 2);
    color_buffer
        .bind()
        .empty(
            TextureTarget::Texture2d,
            0,
            TextureInternalFormat::Rgba16f,
            w,
            h,
            TextureFormat::Rgba,
            GlType::Float,
        )
        .parameter_int(TextureParameter::MinFilter, gl::LINEAR as i32)
        .parameter_int(TextureParameter::MagFilter, gl::LINEAR as i32);
    let mut rbo_depth = Renderbuffer::new();
    rbo_depth
        .bind()
        .storage(TextureInternalFormat::DepthComponent, w, h);
    hdr_fbo
        .bind()
        .texture_2d(Attachment::Color0, TextureTarget::Texture2d, &color_buffer, 0)
        .renderbuffer(Attachment::Depth, &rbo_depth)
        .check_status().expect("framebuffer not complete");


    let mut last_time = PreciseTime::now();
    let fps_step = Duration::seconds(1);
    let mut number_of_frames = 0;

    while running {
        let mut mouse_delta = (0.0, 0.0);

        number_of_frames += 1;
        let now = PreciseTime::now();
        let delta = last_time.to(now);
        if delta > fps_step {
            last_time = now;
            gl_window.set_title(&format!("{}", number_of_frames));
            number_of_frames = 0;
        }

        events_loop.poll_events(|event| match event {
            glutin::Event::WindowEvent { event, .. } => match event {
                glutin::WindowEvent::Closed => running = false,
                glutin::WindowEvent::Resized(w, h) => gl_window.resize(w, h),
                glutin::WindowEvent::CursorMoved { position, .. } => {
                    match last_pos {
                        None => {
                            last_pos = Some(position);
                        }
                        Some(lp) => {
                            last_pos = Some(position);
                            mouse_delta = (
                                position.0 as f32 - lp.0 as f32,
                                position.1 as f32 - lp.1 as f32,
                            );
                        }
                    }
                    // let (w, h) = gl_window.get_outer_size().unwrap();
                    // let (x, y) = gl_window.get_position().unwrap();
                    // ignore_next_mouse_move = true;
                    // gl_window.set_cursor_position(x + w as i32 / 2, y + h as i32 / 2).unwrap();
                }
                glutin::WindowEvent::KeyboardInput { input, .. } => {
                    if let Some(keycode) = input.virtual_keycode {
                        use glutin::VirtualKeyCode as Kc;

                        let value = if input.state == glutin::ElementState::Pressed {
                            1.0
                        } else {
                            0.0
                        };

                        match keycode {
                            Kc::Escape => running = false,
                            Kc::W => inputs.w = value,
                            Kc::A => inputs.a = value,
                            Kc::S => inputs.s = value,
                            Kc::D => inputs.d = value,
                            Kc::Up => inputs.up = value,
                            Kc::Down => inputs.down = value,
                            Kc::Left => inputs.left = value,
                            Kc::Right => inputs.right = value,
                            Kc::Space => inputs.space = value,
                            Kc::LShift => inputs.shift = value,
                            _ => {}
                        }
                    }
                }
                _ => {}
            },
            _ => (),
        });

        t += 1.0;

        let pi = std::f32::consts::PI;

        let up = camera.up();
        let right = camera.front().cross(up).normalize();
        let front = up.cross(right).normalize();

        let walk_speed = 0.1;
        let sensitivity = 0.005;

        camera.pos += walk_speed
            * (front * (inputs.w - inputs.s) + right * (inputs.d - inputs.a)
                + up * (inputs.space - inputs.shift));
        camera.yaw += sensitivity * mouse_delta.0;
        camera.pitch = (camera.pitch - sensitivity * mouse_delta.1)
            .max(-pi / 2.001)
            .min(pi / 2.001);
        // camera.yaw += sensitivity * (inputs.right - inputs.left);
        // camera.pitch = (camera.pitch + sensitivity * (inputs.up - inputs.down))
        //     .max(-pi / 2.001)
        //     .min(pi / 2.001);

        let light_pos = V3::new(
            1.5 + -20.0 * (t / 10.0).sin(),
            1.0,
            -20.0 * (t / 20.0).sin(),
        );
        let light_pos2 = V3::new(
            1.5 + -50.0 * ((t + 23.0) / 14.0).sin(),
            2.0,
            -50.0 * (t / 90.0).sin(),
        );

        let view = camera.get_view();
        let projection = camera.get_projection();

        unsafe {
            gl::ClearColor(0.8, 0.8, 0.9, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
        }

        {
            let binding = hdr_fbo.bind();
            unsafe {
                gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
            }
            {
                let program = program.bind();
                program.bind_mat4("projection", &projection);
                program.bind_mat4("view", &view);
                program.bind_vec3("lightPos", &light_pos);
                program.bind_vec3("viewPos", &camera.pos);
                program.bind_float("time", t);

                let one = V3::new(1.0, 1.0, 1.0);

                let lights = vec![
                    Light::Directional(DirectionalLight {
                        diffuse: V3::new(0.0, 1.0, 1.0) * 0.1,
                        ambient: one * 0.0,
                        specular: one * 0.05,

                        direction: -light_pos,
                    }),
                    Light::Point(PointLight {
                        diffuse: V3::new(0.4, 0.0, 0.0),
                        ambient: one * 0.0,
                        specular: one * 0.2,

                        position: light_pos,
                        constant: 1.0,
                        linear: 0.07,
                        quadratic: 0.017,
                    }),
                    Light::Point(PointLight {
                        diffuse: V3::new(0.2, 0.2, 0.2),
                        ambient: one * 0.0,
                        specular: one * 0.2,

                        position: light_pos2,
                        constant: 1.0,
                        linear: 0.014,
                        quadratic: 0.0007,
                    }),
                    Light::Spot(SpotLight {
                        diffuse: V3::new(1.0, 1.0, 1.0),
                        ambient: one * 0.0,
                        specular: V3::new(0.0, 1.0, 0.0),

                        position: camera.pos,
                        direction: camera.front(),

                        cut_off: Rad(0.2181661565),
                        outer_cut_off: Rad(0.3054326191),
                    }),
                ];

                program.bind_lights(
                    ("numDirectionalLights", "directionalLights"),
                    ("numPointLights", "pointLights"),
                    ("numSpotLights", "spotLights"),
                    &lights,
                );

                // unsafe {
                //     skybox.texture.bind_to(TextureSlot::Six);
                // }
                program.bind_texture("skybox", &skybox.texture, TextureSlot::Six);
                // program.bind_int("skybox", TextureSlot::Six.into());
                {
                    let model =
                        Mat4::from_scale(1.0 / 4.0) * Mat4::from_translation(V3::new(0.0, 0.0, 4.0));
                    program.bind_mat4("model", &model);
                    nanosuit.draw_instanced(&program, &instances.bind());
                }
                {
                    let model = Mat4::from_angle_y(Rad(pi));
                    program.bind_mat4("model", &model);
                    cyborg.draw_instanced(&program, &instances.bind());
                }
                {
                    let model = Mat4::from_scale(1.0);
                    program.bind_mat4("model", &model);
                    let mut c = VertexBuffer::from_data(&vec![
                        Mat4::from_translation(light_pos),
                        Mat4::from_translation(light_pos2),
                    ]);
                    cube_mesh.bind().draw_instanced(&program, &c.bind());
                }
            }
            if draw_normals {
                let program = normal_program.bind();
                program
                    .bind_mat4("projection", &projection)
                    .bind_mat4("view", &view);
                nanosuit.draw_instanced(&program, &instances.bind());
                let mut c = VertexBuffer::from_data(&vec![
                    Mat4::from_translation(light_pos),
                    Mat4::from_translation(light_pos2),
                ]);
                cube_mesh.bind().draw_instanced(&program, &c.bind());
                cyborg.draw_instanced(&program, &instances.bind());
            }
            {
                unsafe {
                    gl::DepthFunc(gl::LEQUAL);
                    gl::Disable(gl::CULL_FACE);
                }
                let program = skybox_program.bind();
                let mut view = view.clone();
                view.w = V4::new(0.0, 0.0, 0.0, 0.0);
                program
                    .bind_mat4("projection", &projection)
                    .bind_mat4("view", &view)
                    .bind_texture("skybox", &skybox.texture, TextureSlot::Ten);
                cube_mesh.bind().draw(&program);
                unsafe {
                    gl::DepthFunc(gl::LESS);
                    gl::Enable(gl::CULL_FACE);
                }
            }
        }

        unsafe {
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
        }

        {
            let hdr = hdr_program.bind();
            hdr.bind_texture("hdrBuffer", &color_buffer, TextureSlot::Zero);
            let mut c = VertexBuffer::from_data(&vec![
                Mat4::from_scale(1.0),
            ]);
            rect_mesh.bind().draw_instanced(&hdr, &c.bind());
        }

        gl_window.swap_buffers().unwrap();
    }
}

macro_rules! v {
    ($pos:expr, $norm:expr, $tex:expr, $tangent:expr) => {{
        let tangent = $tangent.into();
        Vertex {
            pos: $pos.into(),
            tex: $tex.into(),
            norm: $norm.into(),
            tangent: tangent,
            bitangent: tangent.cross($norm.into()),
        }
    }};
}

fn rect_verticies() -> Vec<Vertex> {
    vec![
        v!([-1.0,  1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0], [0.0, 1.0, 0.0]),
        v!([-1.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0], [0.0, 1.0, 0.0]),
        v!([ 1.0,  1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0], [0.0, 1.0, 0.0]),
        v!([-1.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0], [0.0, 1.0, 0.0]),
        v!([ 1.0, -1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0], [0.0, 1.0, 0.0]),
        v!([ 1.0,  1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0], [0.0, 1.0, 0.0]),
    ]
}

fn cube_vertices() -> Vec<Vertex> {
    vec![
        // Back face
        v!(
            [-0.5, -0.5, -0.5],
            [0.0, 0.0, -1.0],
            [0.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [0.5, 0.5, -0.5],
            [0.0, 0.0, -1.0],
            [1.0, 1.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [0.5, -0.5, -0.5],
            [0.0, 0.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [0.5, 0.5, -0.5],
            [0.0, 0.0, -1.0],
            [1.0, 1.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [-0.5, -0.5, -0.5],
            [0.0, 0.0, -1.0],
            [0.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [-0.5, 0.5, -0.5],
            [0.0, 0.0, -1.0],
            [0.0, 1.0],
            [0.0, 1.0, 0.0]
        ),
        // Front face
        v!(
            [-0.5, -0.5, 0.5],
            [0.0, 0.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [0.5, -0.5, 0.5],
            [0.0, 0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 1.0],
            [1.0, 1.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 1.0],
            [1.0, 1.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [-0.5, 0.5, 0.5],
            [0.0, 0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [-0.5, -0.5, 0.5],
            [0.0, 0.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        // Left face
        v!(
            [-0.5, 0.5, 0.5],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [-0.5, 0.5, -0.5],
            [-1.0, 0.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [-0.5, -0.5, -0.5],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [-0.5, -0.5, -0.5],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [-0.5, -0.5, 0.5],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [-0.5, 0.5, 0.5],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        // Right face
        v!(
            [0.5, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [0.5, -0.5, -0.5],
            [1.0, 0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [0.5, 0.5, -0.5],
            [1.0, 0.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [0.5, -0.5, -0.5],
            [1.0, 0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [0.5, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [0.5, -0.5, 0.5],
            [1.0, 0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        // Bottom face
        v!(
            [-0.5, -0.5, -0.5],
            [0.0, -1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0, 0.0]
        ),
        v!(
            [0.5, -0.5, -0.5],
            [0.0, -1.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0, 0.0]
        ),
        v!(
            [0.5, -0.5, 0.5],
            [0.0, -1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0, 0.0]
        ),
        v!(
            [0.5, -0.5, 0.5],
            [0.0, -1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0, 0.0]
        ),
        v!(
            [-0.5, -0.5, 0.5],
            [0.0, -1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0, 0.0]
        ),
        v!(
            [-0.5, -0.5, -0.5],
            [0.0, -1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0, 0.0]
        ),
        // Top face
        v!(
            [-0.5, 0.5, -0.5],
            [0.0, 1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0, 0.0]
        ),
        v!(
            [0.5, 0.5, 0.5],
            [0.0, 1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0, 0.0]
        ),
        v!(
            [0.5, 0.5, -0.5],
            [0.0, 1.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0, 0.0]
        ),
        v!(
            [0.5, 0.5, 0.5],
            [0.0, 1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0, 0.0]
        ),
        v!(
            [-0.5, 0.5, -0.5],
            [0.0, 1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0, 0.0]
        ),
        v!(
            [-0.5, 0.5, 0.5],
            [0.0, 1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0, 0.0]
        ),
    ]
}
