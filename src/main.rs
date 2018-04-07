#![feature(fs_read_write, stmt_expr_attributes)]

extern crate cgmath;
extern crate gl;
extern crate glutin;
extern crate image;
extern crate obj;

use cgmath::{InnerSpace, Rad};

use glutin::GlContext;

use image::GenericImage;

use std::{ffi, fs, mem, os, ptr};

#[derive(Debug, Clone, Copy)]
enum ShaderType {
    Vertex,
    Fragment,
}
impl Into<u32> for ShaderType {
    fn into(self) -> u32 {
        match self {
            ShaderType::Vertex => gl::VERTEX_SHADER,
            ShaderType::Fragment => gl::FRAGMENT_SHADER,
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

struct UniformLocation<'a>(gl::types::GLint, std::marker::PhantomData<&'a u8>);
impl<'a> UniformLocation<'a> {
    fn new(loc: gl::types::GLint) -> UniformLocation<'a> {
        UniformLocation(loc, std::marker::PhantomData)
    }
}

struct Program(gl::types::GLuint);
impl Program {
    fn new_from_disk(vs_path: &str, fs_path: &str) -> Result<Program, ()> {
        let vs_src = fs::read_to_string(vs_path).expect("unable to load vertex shader");
        let fs_src = fs::read_to_string(fs_path).expect("unable to load fragment shader");

        Program::new(&vs_src, &fs_src)
    }
    fn new(vs_src: &str, fs_src: &str) -> Result<Program, ()> {
        let vs = VertexShader::new(vs_src)?;
        let fs = FragmentShader::new(fs_src)?;

        let program_id = unsafe {
            let program_id = gl::CreateProgram();

            gl::AttachShader(program_id, (vs.0).1);
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

        Ok(Program(program_id))
    }
    fn use_program(&self) {
        unsafe {
            gl::UseProgram(self.0);
        }
    }
    fn get_uniform_location<'a>(&'a self, name: &str) -> UniformLocation<'a> {
        let loc =
            unsafe { gl::GetUniformLocation(self.0, ffi::CString::new(name).unwrap().as_ptr()) };
        UniformLocation::new(loc)
    }
    fn bind_mat4_<'a>(&'a self, loc: UniformLocation<'a>, mat: &Mat4) {
        unsafe {
            gl::UniformMatrix4fv(loc.0, 1, gl::FALSE, mat as *const _ as *const _);
        }
    }
    fn bind_mat4(&self, name: &str, mat: &Mat4) {
        let loc = self.get_uniform_location(name);
        self.bind_mat4_(loc, mat);
    }
    fn bind_int(&self, name: &str, i: i32) {
        let loc = self.get_uniform_location(name);
        unsafe {
            gl::Uniform1i(loc.0, i);
        }
    }
    fn bind_vec3(&self, name: &str, v: &V3) {
        let loc = self.get_uniform_location(name);
        unsafe {
            gl::Uniform3f(loc.0, v.x, v.y, v.z);
        }
    }
    fn bind_texture(&self, name: &str, tex: &Texture) {
        if let Some(slot) = tex.binding {
            self.bind_int(name, slot.into());
        } else {
            unimplemented!();
        }
    }
}
impl Drop for Program {
    fn drop(&mut self) {
        unsafe { gl::DeleteProgram(self.0) }
    }
}

type V2 = cgmath::Vector2<f32>;
type V3 = cgmath::Vector3<f32>;
type V4 = cgmath::Vector4<f32>;
type Mat4 = cgmath::Matrix4<f32>;

struct Vertex {
    pos: V3,
    norm: V3,
    tex: V2,
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
    fn get_view(&self) -> Mat4 {
        let origo = cgmath::Point3::new(0.0, 0.0, 0.0);
        Mat4::look_at(origo + self.pos, origo + self.pos + self.front(), self.up())
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
}
impl Into<u32> for TextureSlot {
    fn into(self) -> u32 {
        use TextureSlot::*;
        match self {
            Zero => gl::TEXTURE0,
            One => gl::TEXTURE1,
            Two => gl::TEXTURE2,
            Three => gl::TEXTURE3,
            Four => gl::TEXTURE4,
            Five => gl::TEXTURE5,
            Six => gl::TEXTURE6,
        }
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
        }
    }
}

struct Texture {
    id: gl::types::GLuint,
    binding: Option<TextureSlot>,
}
impl Texture {
    fn new(path: &str) -> Texture {
        let img = image::open(path).expect("unable to read container2.png");
        let tex_id = unsafe {
            let mut tex_id = mem::uninitialized();
            gl::GenTextures(1, &mut tex_id);
            gl::BindTexture(gl::TEXTURE_2D, tex_id);

            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_S, gl::REPEAT as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_WRAP_T, gl::REPEAT as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
            gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);

            let (w, h) = img.dimensions();

            gl::TexImage2D(
                gl::TEXTURE_2D,
                0,
                gl::RGB as i32,
                w as i32,
                h as i32,
                0,
                gl::RGB,
                gl::UNSIGNED_BYTE,
                &img.to_rgb().into_raw()[0] as *const _ as *const _,
            );

            tex_id
        };

        Texture {
            id: tex_id,
            binding: None,
        }
    }

    unsafe fn bind_to(&mut self, slot: TextureSlot) -> TextureSlot {
        if let Some(slot) = self.binding {
            return slot;
        }

        gl::ActiveTexture(slot.into());
        gl::BindTexture(gl::TEXTURE_2D, self.id);

        self.binding = Some(slot);

        slot
    }
}

macro_rules! offset_of {
    ($ty:ty, $field:ident) => {
        #[allow(unused_unsafe)]
        unsafe {
            &(*(0 as *const $ty)).$field as *const _ as usize
        }
    };
}

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

struct Vao(gl::types::GLuint);
struct Vbo(gl::types::GLuint);
struct Ebo(gl::types::GLuint);

impl Vao {
    fn new() -> Vao {
        unsafe {
            let mut vao = mem::uninitialized();
            gl::GenVertexArrays(1, &mut vao);
            Vao(vao)
        }
    }
    fn bind(&mut self) -> VaoBinder {
        unsafe { gl::BindVertexArray(self.0); }
        VaoBinder(self)
    }
}

impl Vbo {
    fn new() -> Vbo {
        unsafe {
            let mut vbo = mem::uninitialized();
            gl::GenBuffers(1, &mut vbo);
            Vbo(vbo)
        }
    }
}

impl Ebo {
    fn new() -> Ebo {
        unsafe {
            let mut ebo = mem::uninitialized();
            gl::GenBuffers(1, &mut ebo);
            Ebo(ebo)
        }
    }
}

struct VaoBinder<'a>(&'a mut Vao);

impl<'a> VaoBinder<'a> {
    fn bind_vbo<'b>(&'b mut self, vbo: &'b mut Vbo) -> VboBinder<'a, 'b> {
        unsafe { gl::BindBuffer(gl::ARRAY_BUFFER, vbo.0); }
        VboBinder(self, vbo)
    }
    fn bind_ebo<'b>(&'b mut self, ebo: &'b mut Ebo) -> EboBinder<'a, 'b> {
        unsafe { gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo.0) }
        EboBinder(self, ebo)
    }
    fn draw_arrays(&mut self, typ: u32, u: usize, amt: usize) {
        unsafe { gl::DrawArrays(gl::TRIANGLES, 0, 36); }
    }
}
impl<'a> Drop for VaoBinder<'a> {
    fn drop(&mut self) {
        unsafe { gl::BindVertexArray(0); }
    }
}

struct VboBinder<'a: 'b, 'b>(&'b mut VaoBinder<'a>, &'b mut Vbo);
impl<'a, 'b> VboBinder<'a, 'b> {
    fn buffer_data<T>(&self, data: &[T]) {
        unsafe {
            gl::BufferData(
                gl::ARRAY_BUFFER,
                (data.len() * mem::size_of::<T>()) as isize,
                &data[0] as *const _ as *const _,
                gl::STATIC_DRAW,
            );
        }
    }
    fn vertex_attrib(&mut self, n: usize, size: usize, offset: usize) {
        unsafe {
            gl::VertexAttribPointer(
                n as u32,
                size as i32,
                gl::FLOAT,
                gl::FALSE,
                mem::size_of::<Vertex>() as i32,
                ptr::null::<os::raw::c_void>().add(offset) as *const _,
            );
            gl::EnableVertexAttribArray(n as u32);
        }
    }
}
impl<'a, 'b> Drop for VboBinder<'a, 'b> {
    fn drop(&mut self) {
        unsafe { gl::BindBuffer(gl::ARRAY_BUFFER, 0); }
    }
}

struct EboBinder<'a: 'b, 'b>(&'b mut VaoBinder<'a>, &'b mut Ebo);
impl<'a, 'b> EboBinder<'a, 'b> {
    fn buffer_data<T>(&self, data: &[T]) {
        unsafe {
            gl::BufferData(
                gl::ELEMENT_ARRAY_BUFFER,
                (data.len() * mem::size_of::<T>()) as isize,
                &data[0] as *const _ as *const _,
                gl::STATIC_DRAW,
            );
        }
    } 
}
impl<'a, 'b> Drop for EboBinder<'a, 'b> {
    fn drop(&mut self) {
        unsafe { gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, 0); }
    }
}

#[allow(unused)]
struct Mesh {
    vertices: Vec<Vertex>,
    indecies: Vec<usize>,
    textures: (), // TODO

    vao: Vao,
    vbo: Vbo,
    ebo: Ebo,
}

impl Mesh {
    #[allow(unused)]
    fn new(vertices: Vec<Vertex>, indecies: Vec<usize>, textures: ()) -> Mesh {
        let mut vao = Vao::new();
        let mut vbo = Vbo::new();
        let mut ebo = Ebo::new();
        {
            let mut vao_binder = vao.bind();

            {
                let ebo_binder = vao_binder.bind_ebo(&mut ebo);
                ebo_binder.buffer_data(&indecies);
            }

            let mut vbo_binder = vao_binder.bind_vbo(&mut vbo);
            vbo_binder.buffer_data(&vertices);

            let n = 0;

            let float_size = mem::size_of::<f32>();
            vbo_binder.vertex_attrib(0, size_of!(Vertex, pos) / float_size, offset_of!(Vertex, pos));
            vbo_binder.vertex_attrib(1, size_of!(Vertex, norm) / float_size, offset_of!(Vertex, norm));
            vbo_binder.vertex_attrib(2, size_of!(Vertex, tex) / float_size, offset_of!(Vertex, tex));
        }

        Mesh {
            vertices, indecies, textures, vao, vbo, ebo
        }
    }

    fn bind(&mut self) -> MeshBinding {
        MeshBinding(self)
    }
}

struct MeshBinding<'a>(&'a mut Mesh);
impl<'a> MeshBinding<'a> {
    fn draw(&mut self) {
        self.0.vao.bind().draw_arrays(gl::TRIANGLES, 0, 36);
        // gl::BindVertexArray(self);

        // Ikke instanced
        // gl::DrawArrays(gl::TRIANGLES, 0, 36);

        // Instanced
        // gl::DrawArraysInstanced(gl::TRIANGLES, 0, 36, models.len() as i32);

        // gl::BindVertexArray(0);
    }
}

#[allow(unused)]
struct Model {
    meshes: Vec<Mesh>,
}
impl Model {
    fn new_from_disk(path: &str) -> Model {
        let raw_model: obj::Obj<obj::SimplePolygon> = obj::Obj::load(std::path::Path::new(path)).unwrap();
        println!("{:?}", raw_model.position);
        println!("{:?}", raw_model.material_libs);
        println!("{:?}", raw_model.objects.len());
        Model {
            meshes: vec![],
        }
    }
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

macro_rules! vertex_attribs {
    ($typ:ty, $data:expr, [$($field:ident,)*]) => {
        let float_size = mem::size_of::<f32>();

        let element_size = mem::size_of::<$typ>() as i32;
        let size = ($data.len() * element_size as usize) as gl::types::GLsizeiptr;

        gl::BufferData(
            gl::ARRAY_BUFFER,
            size,
            &$data[0] as *const _ as *const _,
            gl::STATIC_DRAW,
        );

        let i = 0;
        $(
            gl::VertexAttribPointer(
                i,
                (size_of!($typ, $field) / float_size) as i32,
                gl::FLOAT,
                gl::FALSE,
                element_size,
                offset_ptr!($typ, $field),
            );
            gl::EnableVertexAttribArray(i);
            #[allow(unused)]
            let i = i + 1;
        )*
    }
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

    let program = Program::new_from_disk("shader.vs", "shader.fs").unwrap();
    program.use_program();

    let vertices = cube_vertices();

    let models = {
        let nr_models = 1_000_000;
        let mut models = Vec::with_capacity(nr_models);
        
        for i in 0..nr_models {
            // let x = (i % 25) as f32;
            // let y = (i / 25) as f32;
            // let model = Mat4::from_translation(V3::new(2.0 * x, 2.0 * y, 0.0));
            models.push(i as f32);
        }

        models
    };

    let mut mesh = if true {
        Mesh::new(vertices, vec![0, 2, 1, 3, 5, 4], ())
    } else {
        let (vao, _vbo) = unsafe {
            let mut vao = mem::uninitialized();
            let mut vbo = mem::uninitialized();
            let mut instance_vbo = mem::uninitialized();

            gl::GenVertexArrays(1, &mut vao);
            gl::GenBuffers(1, &mut vbo);
            gl::GenBuffers(1, &mut instance_vbo);

            gl::BindVertexArray(vao);

            gl::BindBuffer(gl::ARRAY_BUFFER, vbo);

            if true {
                vertex_attribs!(Vertex, vertices, [pos, norm, tex,]);
            } else {
                let element_size = mem::size_of::<Vertex>() as i32;
                let size = (vertices.len() * element_size as usize) as gl::types::GLsizeiptr;

                gl::BufferData(
                    gl::ARRAY_BUFFER,
                    size,
                    &vertices[0] as *const _ as *const _,
                    gl::STATIC_DRAW,
                );

                gl::VertexAttribPointer(
                    0,
                    3,
                    gl::FLOAT,
                    gl::FALSE,
                    element_size,
                    offset_ptr!(Vertex, pos),
                );
                gl::EnableVertexAttribArray(0);

                gl::VertexAttribPointer(
                    1,
                    3,
                    gl::FLOAT,
                    gl::FALSE,
                    element_size,
                    offset_ptr!(Vertex, norm),
                );
                gl::EnableVertexAttribArray(1);

                gl::VertexAttribPointer(
                    2,
                    2,
                    gl::FLOAT,
                    gl::FALSE,
                    element_size,
                    offset_ptr!(Vertex, tex),
                );
                gl::EnableVertexAttribArray(2);
            }

            // Instance data

            gl::BindBuffer(gl::ARRAY_BUFFER, instance_vbo);
            gl::BufferData(
                gl::ARRAY_BUFFER,
                (models.len() * mem::size_of::<f32>()) as isize,
                &models[0] as *const _ as *const _,
                gl::STATIC_DRAW,
            );

            gl::VertexAttribPointer(
                3,
                1,
                gl::FLOAT,
                gl::FALSE,
                mem::size_of::<f32>() as i32,
                ptr::null(),
            );
            gl::EnableVertexAttribArray(3);

            gl::BindBuffer(gl::ARRAY_BUFFER, 0);

            gl::VertexAttribDivisor(3, 1);

            gl::BindVertexArray(0);

            (vao, vbo)
        };
        unimplemented!()
    };

    // let mut nanosuit = Model::new_from_disk("nanosuit/nanosuit.obj");

    let mut tex1 = Texture::new("container2.png");
    let mut tex2 = Texture::new("container2_specular.png");


    unsafe {
        tex1.bind_to(TextureSlot::Zero);
        tex2.bind_to(TextureSlot::One);
    }

    let mut t: f32 = 0.0;

    unsafe {
        gl::Enable(gl::DEPTH_TEST);
        gl::Enable(gl::CULL_FACE);
    }

    let mut camera = Camera::new(
        V3::new(t.sin(), (t / 10.0).sin(), 1.0),
        Rad(std::f32::consts::PI / 2.0),
        (screen_width as f32) / (screen_height as f32),
    );

    let mut inputs = Input::default();

    let mut running = true;
    let mut last_pos = None;
    while running {
        let mut mouse_delta = (0.0, 0.0);

        events_loop.poll_events(|event| match event {
            glutin::Event::WindowEvent { event, .. } => match event {
                glutin::WindowEvent::Closed => running = false,
                glutin::WindowEvent::Resized(w, h) => gl_window.resize(w, h),
                glutin::WindowEvent::CursorMoved { position, .. } => {
                    match last_pos {
                        None => {
                            last_pos = Some(position);
                        },
                        Some(lp) => {
                            last_pos = Some(position);
                            mouse_delta = (position.0 as f32 - lp.0 as f32, position.1 as f32 - lp.1 as f32);
                        }
                    }
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
                _ => (),
            },
            _ => (),
        });

        t += 0.1;

        let pi = std::f32::consts::PI;

        let up = camera.up();
        let right = camera.front().cross(up);
        let front = up.cross(right);

        let walk_speed = 0.1;
        let sensitivity = 0.005;

        camera.pos += walk_speed * (front * (inputs.w - inputs.s) + right * (inputs.d - inputs.a) + up * (inputs.space - inputs.shift));
        camera.yaw += sensitivity * mouse_delta.0;
        camera.pitch = (camera.pitch - sensitivity * mouse_delta.1)
            .max(-pi / 2.001)
            .min(pi / 2.001);
        // camera.yaw += sensitivity * (inputs.right - inputs.left);
        // camera.pitch = (camera.pitch + sensitivity * (inputs.up - inputs.down))
        //     .max(-pi / 2.001)
        //     .min(pi / 2.001);

        let model = Mat4::from_scale(1.0);
        let view = camera.get_view();
        let projection = camera.get_projection();

        unsafe {
            gl::ClearColor(0.1, 0.1, 0.1, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
        }
        program.use_program();
        program.bind_texture("diffuseTex", &tex1);
        program.bind_texture("specularTex", &tex2);
        program.bind_mat4("projection", &projection);
        program.bind_mat4("view", &view);
        program.bind_mat4("model", &model);
        program.bind_vec3("lightPos", &V3::new(1.5, 1.0, 20.0));
        program.bind_vec3("viewPos", &camera.pos);
        
        // for i in 0..100 {
        //     let x = (i % 25) as f32;
        //     let y = (i / 25) as f32;
        //     let model = model + Mat4::from_translation(V3::new(2.0 * x, 2.0 * y, 0.0));
        //     program.bind_mat4(&format!("models[{}]", i), &model);
        // }
        mesh.bind().draw();
        // unsafe {
        //     gl::BindVertexArray(vao);

        //     // Ikke instanced
        //     gl::DrawArrays(gl::TRIANGLES, 0, 36);

        //     // Instanced
        //     gl::DrawArraysInstanced(gl::TRIANGLES, 0, 36, models.len() as i32);

        //     gl::BindVertexArray(0);
        // }

        gl_window.swap_buffers().unwrap();
    }
}

macro_rules! v {
    ($pos:expr, $norm:expr, $tex:expr) => {
        Vertex {
            pos: $pos.into(),
            norm: $norm.into(),
            tex: $tex.into(),
        }
    };
}

fn cube_vertices() -> Vec<Vertex> {
    vec![
        // Back face
        v!([-0.5, -0.5, -0.5], [0.0, 0.0, -1.0], [0.0, 0.0]),
        v!([0.5, 0.5, -0.5], [0.0, 0.0, -1.0], [1.0, 1.0]),
        v!([0.5, -0.5, -0.5], [0.0, 0.0, -1.0], [1.0, 0.0]),
        v!([0.5, 0.5, -0.5], [0.0, 0.0, -1.0], [1.0, 1.0]),
        v!([-0.5, -0.5, -0.5], [0.0, 0.0, -1.0], [0.0, 0.0]),
        v!([-0.5, 0.5, -0.5], [0.0, 0.0, -1.0], [0.0, 1.0]),
        // Front face
        v!([-0.5, -0.5, 0.5], [0.0, 0.0, 1.0], [0.0, 0.0]),
        v!([0.5, -0.5, 0.5], [0.0, 0.0, 1.0], [1.0, 0.0]),
        v!([0.5, 0.5, 0.5], [0.0, 0.0, 1.0], [1.0, 1.0]),
        v!([0.5, 0.5, 0.5], [0.0, 0.0, 1.0], [1.0, 1.0]),
        v!([-0.5, 0.5, 0.5], [0.0, 0.0, 1.0], [0.0, 1.0]),
        v!([-0.5, -0.5, 0.5], [0.0, 0.0, 1.0], [0.0, 0.0]),
        // Left face
        v!([-0.5, 0.5, 0.5], [-1.0, 0.0, 0.0], [1.0, 0.0]),
        v!([-0.5, 0.5, -0.5], [-1.0, 0.0, 0.0], [1.0, 1.0]),
        v!([-0.5, -0.5, -0.5], [-1.0, 0.0, 0.0], [0.0, 1.0]),
        v!([-0.5, -0.5, -0.5], [-1.0, 0.0, 0.0], [0.0, 1.0]),
        v!([-0.5, -0.5, 0.5], [-1.0, 0.0, 0.0], [0.0, 0.0]),
        v!([-0.5, 0.5, 0.5], [-1.0, 0.0, 0.0], [1.0, 0.0]),
        // Right face
        v!([0.5, 0.5, 0.5], [1.0, 0.0, 0.0], [1.0, 0.0]),
        v!([0.5, -0.5, -0.5], [1.0, 0.0, 0.0], [0.0, 1.0]),
        v!([0.5, 0.5, -0.5], [1.0, 0.0, 0.0], [1.0, 1.0]),
        v!([0.5, -0.5, -0.5], [1.0, 0.0, 0.0], [0.0, 1.0]),
        v!([0.5, 0.5, 0.5], [1.0, 0.0, 0.0], [1.0, 0.0]),
        v!([0.5, -0.5, 0.5], [1.0, 0.0, 0.0], [0.0, 0.0]),
        // Bottom face
        v!([-0.5, -0.5, -0.5], [0.0, -1.0, 0.0], [0.0, 1.0]),
        v!([0.5, -0.5, -0.5], [0.0, -1.0, 0.0], [1.0, 1.0]),
        v!([0.5, -0.5, 0.5], [0.0, -1.0, 0.0], [1.0, 0.0]),
        v!([0.5, -0.5, 0.5], [0.0, -1.0, 0.0], [1.0, 0.0]),
        v!([-0.5, -0.5, 0.5], [0.0, -1.0, 0.0], [0.0, 0.0]),
        v!([-0.5, -0.5, -0.5], [0.0, -1.0, 0.0], [0.0, 1.0]),
        // Top face
        v!([-0.5, 0.5, -0.5], [0.0, 1.0, 0.0], [0.0, 1.0]),
        v!([0.5, 0.5, 0.5], [0.0, 1.0, 0.0], [1.0, 0.0]),
        v!([0.5, 0.5, -0.5], [0.0, 1.0, 0.0], [1.0, 1.0]),
        v!([0.5, 0.5, 0.5], [0.0, 1.0, 0.0], [1.0, 0.0]),
        v!([-0.5, 0.5, -0.5], [0.0, 1.0, 0.0], [0.0, 1.0]),
        v!([-0.5, 0.5, 0.5], [0.0, 1.0, 0.0], [0.0, 0.0]),
    ]
}
