use gl;

use types::GlError;

use std::{mem, os, ptr, marker::PhantomData};

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

pub struct Buffer {
    kind: BufferKind,
    id: gl::types::GLuint,
    buffer_size: Option<usize>,
}

impl Buffer {
    pub fn new(kind: BufferKind) -> Buffer {
        let id = unsafe {
            let mut id = mem::uninitialized();
            gl::GenBuffers(1, &mut id);
            id
        };
        Buffer {
            kind,
            id,
            buffer_size: None,
        }
    }
    pub fn bind(&mut self) -> BufferBinder {
        BufferBinder::new(self)
    }
    pub fn size(&self) -> usize {
        self.buffer_size.unwrap_or(0)
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            gl::DeleteBuffers(1, &self.id as *const _);
        }
    }
}

pub struct BufferBinder<'a>(&'a mut Buffer);
impl<'a> BufferBinder<'a> {
    pub fn new(buffer: &mut Buffer) -> BufferBinder {
        unsafe { gl::BindBuffer(buffer.kind.into(), buffer.id) }
        GlError::check().expect("error binding buffer BufferBinder::new");
        BufferBinder(buffer)
    }
    pub fn size(&self) -> usize {
        self.0.size()
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
    pub fn alloc_elements<T>(&mut self, num_elements: usize) {
        self.alloc(num_elements * mem::size_of::<T>());
    }
    pub fn buffer_sub_data<T>(&mut self, offset: usize, data: &[T]) -> &mut Self {
        let data_ptr = &data[0] as *const _ as *const _;
        unsafe {
            self.buffer_raw_sub_data(offset, data.len() * mem::size_of::<T>(), data_ptr);
        }
        self
    }
    pub fn buffer_data<T>(&mut self, data: &[T]) {
        let size = data.len() * mem::size_of::<T>();
        if size > 0 {
            let data_ptr = &data[0] as *const _ as *const _;
            unsafe {
                self.buffer_raw_data(size, data_ptr);
            }
        }
    }
}
impl<'a> Drop for BufferBinder<'a> {
    fn drop(&mut self) {
        unsafe {
            gl::BindBuffer(self.0.kind.into(), 0);
        }
    }
}

pub struct VertexBuffer<T> {
    buffer: Buffer,
    phantom: PhantomData<T>,
}
pub trait ElementKind {
    fn gl() -> u32;
}
impl ElementKind for u8 {
    fn gl() -> u32 {
        gl::UNSIGNED_BYTE
    }
}
impl ElementKind for u16 {
    fn gl() -> u32 {
        gl::UNSIGNED_SHORT
    }
}
impl ElementKind for u32 {
    fn gl() -> u32 {
        gl::UNSIGNED_INT
    }
}
pub struct ElementBuffer<T: ElementKind> {
    buffer: Buffer,
    phantom: PhantomData<T>,
}

impl<T> VertexBuffer<T> {
    pub fn new() -> VertexBuffer<T> {
        VertexBuffer {
            buffer: Buffer::new(BufferKind::Array),
            phantom: PhantomData,
        }
    }
    pub fn from_size(size: usize) -> VertexBuffer<T> {
        let mut vbo = VertexBuffer::new();
        vbo.bind().alloc(size);
        vbo
    }
    pub fn from_data(data: &[T]) -> VertexBuffer<T> {
        let mut vbo = VertexBuffer::new();
        vbo.bind().buffer_data(data);
        vbo
    }
    pub fn bind(&mut self) -> VertexBufferBinder<T> {
        VertexBufferBinder::new(self)
    }
}

pub struct VertexBufferBinder<'a, T>(BufferBinder<'a>, PhantomData<T>);
impl<'a, T> VertexBufferBinder<'a, T> {
    pub fn new(vbo: &'a mut VertexBuffer<T>) -> VertexBufferBinder<'a, T> {
        VertexBufferBinder(vbo.buffer.bind(), PhantomData)
    }
    pub fn len(&self) -> usize {
        self.0.size() / mem::size_of::<T>()
    }
    pub fn alloc(&mut self, size: usize) {
        self.0.alloc(size)
    }
    pub fn buffer_sub_data<V>(&mut self, offset: usize, data: &[V]) -> &mut Self {
        self.0.buffer_sub_data(offset, data);
        self
    }
    pub fn buffer_data<V>(&mut self, data: &[V]) {
        self.0.buffer_data(data)
    }
}

impl<T: ElementKind> ElementBuffer<T> {
    pub fn new() -> ElementBuffer<T> {
        ElementBuffer {
            buffer: Buffer::new(BufferKind::ElementArray),
            phantom: PhantomData,
        }
    }
    pub fn from_data(data: &[T]) -> ElementBuffer<T> {
        let mut vbo = ElementBuffer::new();
        vbo.bind().buffer_data(data);
        vbo
    }
    pub fn bind(&mut self) -> ElementBufferBinder<T> {
        ElementBufferBinder::new(self)
    }
}

pub struct ElementBufferBinder<'a, T: ElementKind>(BufferBinder<'a>, PhantomData<T>);
#[allow(unused)]
impl<'a, T: ElementKind> ElementBufferBinder<'a, T> {
    pub fn new(vbo: &'a mut ElementBuffer<T>) -> ElementBufferBinder<'a, T> {
        ElementBufferBinder(vbo.buffer.bind(), PhantomData)
    }
    pub fn len(&self) -> usize {
        self.0.size() / mem::size_of::<T>()
    }
    pub fn alloc(&mut self, size: usize) {
        self.0.alloc(size)
    }
    pub fn buffer_sub_data<V>(&mut self, offset: usize, data: &[V]) -> &mut Self {
        self.0.buffer_sub_data(offset, data);
        self
    }
    pub fn buffer_data<V>(&mut self, data: &[V]) {
        self.0.buffer_data(data)
    }
}
