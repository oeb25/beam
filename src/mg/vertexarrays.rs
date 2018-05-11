use gl;
use std::{ptr, os::{self}, mem};

use mg::types::{GlType, GlError};
use mg::framebuffers::FramebufferBinderDrawer;
use mg::buffers::{ElementKind, ElementBufferBinder, VertexBufferBinder};


pub struct VertexArray(gl::types::GLuint);

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
#[allow(unused)]
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
        GlError::check().expect(
            &format!(
                r#"VertexArrayBinder::draw_arrays_instanced: failed to draw instanced.
                Call looks like: gl::DrawArraysInstanced({:?}, {}, {}, {})"#,
                mode, first, count, instances,
            ),
        );
        self
    }
    pub fn draw_elements<T, S>(
        &mut self,
        _fbo: &T,
        mode: DrawMode,
        data: &ElementBufferBinder<S>,
    ) -> &VertexArrayBinder
    where
        T: FramebufferBinderDrawer,
        S: ElementKind
    {
        unsafe {
            gl::DrawElements(
                mode.into(),
                data.len() as i32,
                S::gl(),
                ptr::null::<os::raw::c_void>().add(0),
            );
        }
        self
    }
    pub fn draw_elements_instanced<T, S>(
        &mut self,
        _fbo: &T,
        mode: DrawMode,
        data: &ElementBufferBinder<S>,
        instances: usize,
    ) -> &VertexArrayBinder
    where
        T: FramebufferBinderDrawer,
        S: ElementKind
    {
        unsafe {
            gl::DrawElementsInstanced(
                mode.into(),
                data.len() as i32,
                S::gl(),
                ptr::null::<os::raw::c_void>().add(0),
                instances as i32,
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
    pub fn attrib<T>(
        &self,
        _vbo: &VertexBufferBinder<T>,
        index: usize,
        size: usize,
        typ: GlType,
        stride: usize,
        offset: usize,
    ) -> &VertexArrayBinder {
        unsafe {
            self.attrib_pointer(index, size, typ.into(), false, stride, offset);
            self.enable_attrib_array(index);
        }
        self
    }
    pub fn vbo_attrib<T>(
        &self,
        vbo: &VertexBufferBinder<T>,
        index: usize,
        size: usize,
        offset: usize,
    ) -> &VertexArrayBinder {
        self.attrib(vbo, index, size, GlType::Float, mem::size_of::<T>(), offset);
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
            // gl::BindVertexArray(0);
        }
    }
}
