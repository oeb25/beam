use gl;
use std::{borrow::Cow,
          ffi,
          fs,
          marker::PhantomData,
          mem,
          ptr,
          path::Path,
      };

use mg::types::GlError;
use mg::textures::{Texture, TextureSlot};

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
        // GlError::check().expect(&format!("unable to get uniform location for name: '{}'", name));
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
