use gl;

use std::mem;

use textures::{Texture, TextureTarget, TextureInternalFormat};

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
pub enum Mask {
    Color,
    Depth,
    Stencil,

    ColorDepth,
    ColorStencil,
    DepthStencil,
}
impl Into<u32> for Mask {
    fn into(self) -> u32 {
        use Mask::*;
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
    fn clear(&self, mask: Mask) -> &Self {
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
        mask: Mask,
        filter: u32,
    ) -> &Self {
        unsafe {
            gl::BlitFramebuffer(
                src.0, src.1, src.2, src.3, dst.0, dst.1, dst.2, dst.3, mask.into(), filter,
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
