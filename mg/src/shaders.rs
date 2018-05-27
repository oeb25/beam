use gl;
use std::{ffi, fs, mem, path::Path, ptr};

use textures::{Texture, TextureSlot};

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
            let source = ffi::CString::new(src).unwrap();
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
                let error_msg =
                    String::from_utf8(buffer).expect("error message could not be turned into utf8");
                println!("Error while compiling shader of type {:?}", kind);
                for line in error_msg.lines() {
                    println!("{}", line);
                }
                return Err(());
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

pub struct UniformLocation(gl::types::GLint);
impl UniformLocation {
    fn new(loc: gl::types::GLint) -> UniformLocation {
        UniformLocation(loc)
    }
}

pub struct UniformBlockIndex(gl::types::GLuint);
impl UniformBlockIndex {
    fn new(loc: gl::types::GLuint) -> UniformBlockIndex {
        UniformBlockIndex(loc)
    }
}

#[derive(Debug)]
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
    pub fn new_from_src(vs_src: &str, gs_src: Option<&str>, fs_src: &str) -> Result<Program, ()> {
        let vs = VertexShader::new(&vs_src).map_err(|e| {
            println!("unable to create vertex shader with src:");
            println!(
                "{}",
                vs_src
                    .lines()
                    .enumerate()
                    .map(|(i, line)| format!("{:?} | {}\n", i + 1, line))
                    .collect::<String>()
            );
            e
        })?; //.expect("unable to create vertex shader");
        let gs = gs_src
            .map(|gs_src| GeometryShader::new(&gs_src))
            .transpose()?;
        let fs = FragmentShader::new(&fs_src).map_err(|e| {
            println!("unable to create fragment shader with src:");
            println!(
                "{}",
                fs_src
                    .lines()
                    .enumerate()
                    .map(|(i, line)| format!("{:?} | {}\n", i + 1, line))
                    .collect::<String>()
            );
            e
        })?; //.expect("unable to create fragment shader");

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
type Mat4 = [[f32; 4]; 4];
pub trait Uloc: Sized {
    fn loc(self, program: &impl ProgramBind) -> UniformLocation;
    fn id(self, program: &impl ProgramBind) -> i32 {
        self.loc(program).0
    }
}
impl<'a> Uloc for &'a str {
    fn loc(self, program: &impl ProgramBind) -> UniformLocation {
        let loc = unsafe {
            gl::GetUniformLocation(
                program.id(),
                ffi::CString::new(self)
                    .expect("unable to create a CString from passed str")
                    .as_ptr(),
            )
        };
        UniformLocation::new(loc)
    }
}
impl<'a> Uloc for &'a String {
    fn loc(self, program: &impl ProgramBind) -> UniformLocation {
        let a: &str = self;
        a.loc(program)
    }
}
impl Uloc for UniformLocation {
    fn loc(self, _program: &impl ProgramBind) -> UniformLocation {
        self
    }
}
use std::cell::Cell;
pub struct ProgramBinding<'a> {
    program: &'a mut Program,
    next_texture_slot: Cell<TextureSlot>,
}
impl<'a> ProgramBinding<'a> {
    fn new(program: &mut Program) -> ProgramBinding {
        unsafe {
            gl::UseProgram(program.id);
        }
        let next_texture_slot = Cell::new(TextureSlot::Zero);
        ProgramBinding {
            program,
            next_texture_slot,
        }
    }
}
impl<'a> ProgramBind for ProgramBinding<'a> {
    fn set_next_texture_slot(&self, slot: TextureSlot) -> &Self {
        self.next_texture_slot.set(slot);
        self
    }
    fn next_texture_slot(&self) -> TextureSlot {
        self.next_texture_slot.get().clone()
    }
    fn id(&self) -> u32 {
        self.program.id
    }
}
use std::cell::RefMut;
pub struct ProgramBindingRefMut<'a> {
    program: RefMut<'a, Program>,
    next_texture_slot: Cell<TextureSlot>,
}
impl<'a> ProgramBindingRefMut<'a> {
    pub fn new(program: RefMut<Program>) -> ProgramBindingRefMut {
        unsafe {
            gl::UseProgram(program.id);
        }
        let next_texture_slot = Cell::new(TextureSlot::Zero);
        ProgramBindingRefMut {
            program,
            next_texture_slot,
        }
    }
}
impl<'a> ProgramBind for ProgramBindingRefMut<'a> {
    fn set_next_texture_slot(&self, slot: TextureSlot) -> &Self {
        self.next_texture_slot.set(slot);
        self
    }
    fn next_texture_slot(&self) -> TextureSlot {
        self.next_texture_slot.get().clone()
    }
    fn id(&self) -> u32 {
        self.program.id
    }
}

pub trait ProgramBind: Sized {
    fn set_next_texture_slot(&self, slot: TextureSlot) -> &Self;
    fn next_texture_slot(&self) -> TextureSlot;
    fn id(&self) -> u32;
    fn bind_mat4(&self, loc: impl Uloc, mat: impl Into<Mat4>) -> &Self {
        self.bind_mat4s(loc, &[mat.into()])
    }
    fn bind_mat4s(&self, loc: impl Uloc, mats: &[Mat4]) -> &Self {
        unsafe {
            gl::UniformMatrix4fv(
                loc.id(self),
                mats.len() as i32,
                gl::FALSE,
                mats.as_ptr() as *const _,
            );
        }
        self
    }
    fn bind_int(&self, loc: impl Uloc, i: i32) -> &Self {
        unsafe {
            gl::Uniform1i(loc.id(self), i);
        }
        self
    }
    fn bind_ints(&self, loc: impl Uloc, i: &[i32]) -> &Self {
        unsafe {
            gl::Uniform1iv(loc.id(self), i.len() as i32, i.as_ptr() as *const _);
        }
        self
    }
    fn bind_uint(&self, loc: impl Uloc, i: u32) -> &Self {
        unsafe {
            gl::Uniform1ui(loc.id(self), i);
        }
        self
    }
    fn bind_bool(&self, loc: impl Uloc, i: bool) -> &Self {
        self.bind_uint(loc, if i { 1 } else { 0 })
    }
    fn bind_texture_to(
        &self,
        loc: impl Uloc,
        texture: &Texture,
        slot: TextureSlot,
    ) -> &Self {
        texture.bind_to(slot);
        self.bind_int(loc, slot.into())
    }
    fn bind_texture_returning_slot(&self, loc: impl Uloc, texture: &Texture) -> TextureSlot {
        let slot = self.next_texture_slot();
        texture.bind_to(slot);
        self.set_next_texture_slot(slot.next());
        self.bind_int(loc, slot.into());
        slot
    }
    fn bind_texture(&self, loc: impl Uloc, texture: &Texture) -> &Self {
        self.bind_texture_returning_slot(loc, texture);
        self
    }
    fn bind_textures<'b>(
        &self,
        loc: impl Uloc,
        textures: impl Iterator<Item = &'b Texture>,
    ) -> &Self {
        let mut cur_slot = self.next_texture_slot();
        let slots = textures
            .map(|tex| {
                let slot = cur_slot;
                tex.bind_to(slot);
                cur_slot = slot.next();
                slot.into()
            })
            .collect::<Vec<_>>();
        self.set_next_texture_slot(cur_slot.next());
        self.bind_ints(loc, &slots)
    }
    fn bind_float(&self, loc: impl Uloc, f: f32) -> &Self {
        unsafe {
            gl::Uniform1f(loc.id(self), f);
        }
        self
    }
    fn bind_float2(&self, loc: impl Uloc, f: &[f32]) -> &Self {
        unsafe {
            gl::Uniform1fv(loc.id(self), f.len() as i32, f.as_ptr() as *const _);
        }
        self
    }
    fn bind_vec3<T: Into<[f32; 3]>>(&self, loc: impl Uloc, v: T) -> &Self {
        let v = v.into();
        unsafe {
            gl::Uniform3f(loc.id(self), v[0], v[1], v[2]);
        }
        self
    }
    fn bind_vec3s(&self, loc: impl Uloc, v: &[[f32; 3]]) -> &Self {
        unsafe {
            gl::Uniform3fv(loc.id(self), v.len() as i32, v.as_ptr() as *const _);
        }
        self
    }
    #[allow(unused)]
    fn get_uniform_block_index(&self, name: &str) -> UniformBlockIndex {
        let loc = unsafe {
            gl::GetUniformBlockIndex(self.id(), ffi::CString::new(name).unwrap().as_ptr())
        };
        UniformBlockIndex::new(loc)
    }
    #[allow(unused)]
    fn uniform_block_binding(&self, name: &str, index: usize) -> &Self {
        let block_index = self.get_uniform_block_index(name);
        unsafe {
            gl::UniformBlockBinding(self.id(), block_index.0, index as u32);
        }
        self
    }
}
