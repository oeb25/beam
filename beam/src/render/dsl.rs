use gl;
use mg::{
    BufferKind, DrawMode, Framebuffer, FramebufferTarget, GlError, GlType, Mask, Program, Texture,
    TextureSlot, VertexArray, VertexBuffer,
};
use misc::{Cacher, Mat4, V3};
use std::{self, borrow::Cow};
use time::PreciseTime;

pub type Mat4g = [[f32; 4]; 4];

type Rect = (u32, u32, u32, u32);

#[derive(Debug)]
pub enum FramebufferCall<'a> {
    Clear(Mask),
    BlitTo(&'a Framebuffer, Rect, Rect, Mask, u32),
}

#[derive(Debug, Clone)]
pub enum UniformValue<'a> {
    Bool(bool),
    Float(f32),
    Int(i32),
    Vec3([f32; 3]),
    Mat4([[f32; 4]; 4]),
    Mat4s(Vec<[[f32; 4]; 4]>),
    Texture(&'a Texture),
    Textures(Vec<&'a Texture>),
}

impl<'a> UniformValue<'a> {
    fn bind(self, loc: i32, texture_slots: &mut TextureSlots<'a>) {
        use self::UniformValue::*;
        match self {
            Bool(b) => Int(if b { 1 } else { 0 }).bind(loc, texture_slots),
            Float(f) => unsafe { gl::Uniform1f(loc, f) },
            Int(i) => unsafe { gl::Uniform1i(loc, i) },
            Vec3(v) => unsafe { gl::Uniform3f(loc, v[0], v[1], v[2]) },
            Mat4(m) => Mat4s(vec![m]).bind(loc, texture_slots), // TODO: avoid allocation
            Mat4s(mats) => unsafe {
                gl::UniformMatrix4fv(loc, mats.len() as i32, gl::FALSE, mats.as_ptr() as *const _);
            },
            Texture(tex) => {
                let slot = texture_slots.get_slot(tex);
                tex.bind_to(slot);
                Int(slot.into()).bind(loc, texture_slots);
            }
            Textures(texs) => {
                for tex in texs.into_iter() {
                    Texture(tex).bind(loc, texture_slots);
                }
            }
        }
    }
}

macro_rules! impl_uniform {
    [$($t:ident($f:ty),)*] => {
        $(
            impl<'a> Into<UniformValue<'a>> for $f {
                fn into(self) -> UniformValue<'a> {
                    UniformValue::$t(self.into())
                }
            }
        )*
    };
}

impl_uniform![
    Bool(bool),
    Float(f32),
    Int(i32),
    Vec3([f32; 3]),
    Vec3(V3),
    Mat4(Mat4g),
    Mat4(Mat4),
    Mat4s(Vec<Mat4g>),
    Texture(&'a Texture),
];

impl<'a> Into<ProgramCall<'a>> for Vec<(Cow<'a, str>, UniformValue<'a>)> {
    fn into(self) -> ProgramCall<'a> {
        ProgramCall::Uniforms(self)
    }
}

impl<'a> Into<ProgramCall<'a>> for Vec<(&'a str, UniformValue<'a>)> {
    fn into(self) -> ProgramCall<'a> {
        self.into_iter()
            .map(|x| {
                let new: Cow<_> = x.0.into();
                (new, x.1)
            })
            .collect::<Vec<_>>()
            .into()
    }
}

#[derive(Debug)]
pub enum ProgramCall<'a> {
    Uniforms(Vec<(Cow<'a, str>, UniformValue<'a>)>),
}

#[derive(Debug)]
pub enum VertexBufferCall<'a> {
    Transforms(Cow<'a, Mat4g>),
}

#[derive(Debug)]
pub enum DrawCall<'a> {
    Arrays(&'a VertexArray, DrawMode, usize, usize),
    ArraysInstanced(
        &'a VertexArray,
        DrawMode,
        usize,
        usize,
        &'a VertexBuffer<Mat4g>,
    ),
}

#[derive(Debug)]
pub enum GlCall<'a, TProgram: 'a> {
    Marker(&'static str),
    Viewport(u32, u32, u32, u32),
    CullFace(u32),  // TODO: use enum instead of u32, and maybe move to draw call
    DepthFunc(u32), // TODO
    SaveTextureSlot,
    RestoreTextureSlot,
    Enable(u32),  // TODO
    Disable(u32), // TODO
    Framebuffer(&'a Framebuffer, FramebufferCall<'a>),
    Program(&'a TProgram, ProgramCall<'a>),
    VertexBuffer(&'a VertexBuffer<Mat4g>, VertexBufferCall<'a>),
    Draw(&'a TProgram, &'a Framebuffer, DrawCall<'a>),
}

pub trait ProgramLike {
    fn id(&self) -> u32;
}

#[derive(Debug)]
struct FramebufferState<'a> {
    read: Option<&'a Framebuffer>,
    draw: Option<&'a Framebuffer>,
}

impl<'a> FramebufferState<'a> {
    fn read(&mut self, fbo: &'a Framebuffer) {
        if self.read != Some(fbo) {
            unsafe {
                gl::BindFramebuffer(FramebufferTarget::Read.into(), fbo.id);
            }
            self.read = Some(fbo);
        }
    }
    fn draw(&mut self, fbo: &'a Framebuffer) {
        if self.draw != Some(fbo) {
            unsafe {
                gl::BindFramebuffer(FramebufferTarget::Draw.into(), fbo.id);
            }
            self.draw = Some(fbo);
        }
    }
}

#[derive(Debug, PartialEq)]
enum ViewportState {
    None,
    Requested(Rect),
    Set(Rect),
}

impl ViewportState {
    fn request(&mut self, rect: Rect) {
        if *self == ViewportState::Requested(rect) || *self == ViewportState::Set(rect) {
            return;
        }

        *self = ViewportState::Requested(rect);
    }
    fn set(&mut self) {
        match self {
            ViewportState::None => unimplemented!("set viewport when nothing was requested"),
            ViewportState::Requested(v) => {
                unsafe {
                    gl::Viewport(v.0 as i32, v.1 as i32, v.2 as i32, v.3 as i32);
                }
                *self = ViewportState::Set(*v);
            }
            ViewportState::Set(_) => {
                // noop
            }
        }
    }
}

#[derive(Debug, PartialEq)]
enum CullFaceState {
    None,
    Requested(u32),
    Set(u32),
}

impl CullFaceState {
    fn request(&mut self, cull_face: u32) {
        if *self == CullFaceState::Requested(cull_face) || *self == CullFaceState::Set(cull_face) {
            return;
        }

        *self = CullFaceState::Requested(cull_face);
    }
    fn set(&mut self) {
        match self {
            CullFaceState::None => unimplemented!("set CullFace when nothing was requested"),
            CullFaceState::Requested(v) => {
                unsafe {
                    gl::CullFace(*v);
                }
                *self = CullFaceState::Set(*v);
            }
            CullFaceState::Set(_) => {
                // noop
            }
        }
    }
}

#[derive(Debug, PartialEq)]
enum DepthFuncState {
    None,
    Requested(u32),
    Set(u32),
}

impl DepthFuncState {
    fn request(&mut self, cull_face: u32) {
        if *self == DepthFuncState::Requested(cull_face) || *self == DepthFuncState::Set(cull_face)
        {
            return;
        }

        *self = DepthFuncState::Requested(cull_face);
    }
    fn set(&mut self) {
        match self {
            DepthFuncState::None => unimplemented!("set DepthFunc when nothing was requested"),
            DepthFuncState::Requested(v) => {
                unsafe {
                    gl::DepthFunc(*v);
                }
                *self = DepthFuncState::Set(*v);
            }
            DepthFuncState::Set(_) => {
                // noop
            }
        }
    }
}

#[derive(Debug)]
struct ProgramState<'a, T: 'a + ProgramLike>(Option<&'a T>);

impl<'a, T: ProgramLike + std::fmt::Debug + std::cmp::PartialEq> ProgramState<'a, T> {
    fn bind(&mut self, program: &'a T) {
        if self.0 != Some(program) {
            unsafe { gl::UseProgram(program.id()) }
            self.0 = Some(program);
        }
    }
}

#[derive(Debug)]
struct VaoState<'a>(Option<&'a VertexArray>);

impl<'a> VaoState<'a> {
    fn bind(&mut self, vao: &'a VertexArray) {
        if self.0 != Some(vao) {
            unsafe {
                gl::BindVertexArray(vao.id);
            }
            self.0 = Some(vao);
        }
    }
    unsafe fn attrib_pointer(
        &self,
        index: usize,
        size: usize,
        typ: GlType,
        normalized: bool,
        stride: usize,
        offset: usize,
    ) {
        gl::VertexAttribPointer(
            index as u32,
            size as i32,
            typ.into(),
            if normalized { gl::TRUE } else { gl::FALSE },
            stride as i32,
            std::ptr::null::<std::os::raw::c_void>().add(offset) as *const _,
        );
    }
    unsafe fn enable_attrib_array(&self, index: usize) {
        gl::EnableVertexAttribArray(index as u32);
    }
    pub fn attrib<S>(
        &self,
        _vbo: &VertexBuffer<S>,
        index: usize,
        size: usize,
        typ: GlType,
        stride: usize,
        offset: usize,
    ) {
        unsafe {
            self.attrib_pointer(index, size, typ.into(), false, stride, offset);
            self.enable_attrib_array(index);
        }
    }
    pub fn vbo_attrib<S>(&self, vbo: &VertexBuffer<S>, index: usize, size: usize, offset: usize) {
        self.attrib(
            vbo,
            index,
            size,
            GlType::Float,
            std::mem::size_of::<S>(),
            offset,
        );
    }
    pub fn attrib_divisor(&self, index: usize, divisor: usize) {
        unsafe {
            gl::VertexAttribDivisor(index as u32, divisor as u32);
        }
    }
}

#[derive(Debug, PartialEq)]
struct VboState<'a, T: 'a>(Option<&'a VertexBuffer<T>>);

impl<'a, T: std::cmp::PartialEq + std::fmt::Debug> VboState<'a, T> {
    fn bind(&mut self, vbo: &'a VertexBuffer<T>) {
        if self.0 != Some(vbo) {
            unsafe {
                gl::BindBuffer(BufferKind::Array.into(), vbo.buffer.id);
            }
            self.0 = Some(vbo);
        }
    }
    fn unbind(&mut self) {
        if self.0.is_some() {
            unsafe {
                gl::BindBuffer(BufferKind::Array.into(), 0);
            }
            self.0 = None;
        }
    }
}

#[derive(Debug)]
struct TextureSlots<'a> {
    slots: [Option<&'a Texture>; 16],
    current_index: usize,
    previous_indices: Vec<usize>,
}

impl<'a> TextureSlots<'a> {
    fn new() -> TextureSlots<'a> {
        TextureSlots {
            slots: [None; 16],
            current_index: 0,
            previous_indices: vec![],
        }
    }
    fn save(&mut self) {
        self.previous_indices.push(self.current_index);
    }
    fn restore(&mut self) {
        self.current_index = self.previous_indices.pop().unwrap_or(0);
    }
    fn get_slot(&mut self, tex: &'a Texture) -> TextureSlot {
        for i in 0..self.current_index {
            if self.slots[i] == Some(tex) {
                return i.into();
            }
        }

        self.slots[self.current_index] = Some(tex);
        let slot = self.current_index.into();
        self.current_index += 1;
        slot
    }
}

#[derive(Debug)]
struct ExecutionState<'a, T: 'a + ProgramLike> {
    viewport: ViewportState,
    cull_face: CullFaceState,
    depth_func: DepthFuncState,

    framebuffer: FramebufferState<'a>,

    program: ProgramState<'a, T>,

    vao_state: VaoState<'a>,
    vbo_state: VboState<'a, Mat4g>,

    uniform_cache: Cacher<(u32, Cow<'a, str>), i32>,

    texture_slots: TextureSlots<'a>,
}

impl<'a, T: ProgramLike> ExecutionState<'a, T> {
    fn new() -> ExecutionState<'a, T> {
        ExecutionState {
            viewport: ViewportState::None,
            cull_face: CullFaceState::None,
            depth_func: DepthFuncState::None,

            framebuffer: FramebufferState {
                read: None,
                draw: None,
            },

            program: ProgramState(None),

            vao_state: VaoState(None),
            vbo_state: VboState(None),

            uniform_cache: Cacher::new(),

            texture_slots: TextureSlots::new(),
        }
    }
}

pub fn execute<'a, Calls, Program>(calls: Calls)
where
    Calls: Iterator<Item = GlCall<'a, Program>>,
    Program: 'a + std::fmt::Debug + std::cmp::PartialEq + ProgramLike,
{
    GlError::check().expect("before execute");

    let mut state = ExecutionState::new();

    let mut timings = vec![];

    // Defaults

    state.cull_face.request(gl::BACK);
    state.depth_func.request(gl::LESS);

    let mut uniform_cache_fetch = 0;
    let mut uniform_cache_total = 0;

    // Executor
    for call in calls {
        let call_str = format!("{:?}", call);

        match call {
            GlCall::Marker(name) => {
                unsafe {
                    gl::Finish();
                }
                timings.push((name, PreciseTime::now()));
            }
            GlCall::Viewport(x, y, w, h) => {
                state.viewport.request((x, y, w, h));
            }
            GlCall::CullFace(c) => {
                state.cull_face.request(c);
            }
            GlCall::DepthFunc(d) => {
                state.depth_func.request(d);
            }
            GlCall::SaveTextureSlot => {
                state.texture_slots.save();
            }
            GlCall::RestoreTextureSlot => {
                state.texture_slots.restore();
            }
            GlCall::Enable(x) => unsafe {
                gl::Enable(x);
            },
            GlCall::Disable(x) => unsafe {
                gl::Disable(x);
            },
            GlCall::Framebuffer(fbo, fbo_call) => match fbo_call {
                FramebufferCall::Clear(mask) => {
                    state.framebuffer.draw(fbo);
                    unsafe { gl::Clear(mask.into()) }
                }
                FramebufferCall::BlitTo(target_fbo, src, dst, mask, filter) => {
                    state.framebuffer.read(fbo);
                    state.framebuffer.draw(target_fbo);

                    unsafe {
                        gl::BlitFramebuffer(
                            src.0 as i32,
                            src.1 as i32,
                            src.2 as i32,
                            src.3 as i32,
                            dst.0 as i32,
                            dst.1 as i32,
                            dst.2 as i32,
                            dst.3 as i32,
                            mask.into(),
                            filter,
                        );
                    }
                }
            },
            GlCall::Program(p, p_call) => match p_call {
                ProgramCall::Uniforms(uniforms) => {
                    state.program.bind(p);
                    for (name, value) in uniforms.into_iter() {
                        uniform_cache_total += 1;
                        let loc = if false {
                            *state
                                .uniform_cache
                                .get_or_insert_with((p.id(), name.clone()), || {
                                    uniform_cache_fetch += 1;
                                    let loc =
                                        unsafe {
                                            gl::GetUniformLocation(
                                        p.id(),
                                        std::ffi::CString::new(&*name)
                                            .expect("unable to create a CString from passed str")
                                            .as_ptr(),
                                    )
                                        };
                                    loc
                                })
                        } else {
                            unsafe {
                                gl::GetUniformLocation(
                                    p.id(),
                                    std::ffi::CString::new(&*name)
                                        .expect("unable to create a CString from passed str")
                                        .as_ptr(),
                                )
                            }
                        };
                        value.bind(loc, &mut state.texture_slots);
                    }
                }
            },
            GlCall::Draw(p, fbo, draw_call) => match draw_call {
                DrawCall::ArraysInstanced(vao, mode, first, count, vbo) => {
                    state.vao_state.bind(vao);
                    state.vbo_state.bind(vbo);

                    let offset = 5;
                    let width = 4;
                    for i in 0..width {
                        let index = i + offset;
                        state.vao_state.vbo_attrib(
                            &vbo,
                            index,
                            width,
                            width * i * std::mem::size_of::<f32>(),
                        );
                        state.vao_state.attrib_divisor(index, 1);
                    }
                    state.vbo_state.unbind();

                    state.framebuffer.draw(fbo);
                    state.program.bind(p);
                    state.viewport.set();
                    state.cull_face.set();
                    state.depth_func.set();
                    unsafe {
                        gl::DrawArraysInstanced(
                            mode.into(),
                            first as i32,
                            count as i32,
                            vbo.len() as i32,
                        );
                    }
                }
                DrawCall::Arrays(vao, mode, first, count) => {
                    state.vao_state.bind(vao);
                    state.vbo_state.unbind();

                    state.framebuffer.draw(fbo);
                    state.program.bind(p);
                    state.viewport.set();
                    state.cull_face.set();
                    state.depth_func.set();
                    unsafe {
                        gl::DrawArrays(mode.into(), first as i32, count as i32);
                    }
                }
            },
            x => unimplemented!("{:?}", x),
        }
        if let Err(e) = GlError::check() {
            println!("State was:\n{:?}", state);
            panic!("error executing {} -> {:?}", call_str, e);
        }
    }

    println!(
        "unifom cache: {}/{}",
        uniform_cache_fetch, uniform_cache_total
    );

    println!("# Render timings");
    for ab in timings.windows(2) {
        let a = ab[0];
        let b = ab[1];

        println!(
            "{:30.}: {:3.5}ms",
            a.0,
            a.1.to(b.1).num_nanoseconds().unwrap() as f32 * 0.000001
        );
    }
    println!(
        "{:30.}: {:3.5}ms",
        "total",
        timings[0]
            .1
            .to(timings[timings.len() - 1].1)
            .num_nanoseconds()
            .unwrap() as f32 * 0.000001
    );
}
