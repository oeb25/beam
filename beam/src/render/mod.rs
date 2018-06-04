pub mod lights;
pub mod materials;
pub mod mesh;
mod primitives;
pub mod store;

use cgmath::{self, InnerSpace, Rad};
use collada;
use gl;
use image;
pub use render::materials::*;

use failure::{Error, ResultExt};

use std::{
    self, cell::RefCell, collections::{BTreeMap, HashMap}, path::Path,
};

use mg::{
    Attachment, Framebuffer, FramebufferBinderBase, FramebufferBinderDraw, FramebufferBinderDrawer,
    FramebufferBinderRead, FramebufferBinderReadDraw, GlError, GlType, Mask, Program, ProgramBind,
    ProgramPin, Renderbuffer, Texture, TextureFormat, TextureInternalFormat, TextureKind,
    TextureParameter, TextureTarget, VertexArrayPin,
};

use mesh::{calculate_tangent_and_bitangent, Mesh};
use misc::{v3, v4, Cacher, Mat4, P3, V3, V4, Vertex};
use render::store::MeshRef;

#[allow(unused)]
#[derive(Debug, Clone)]
pub enum RenderObjectChild {
    Mesh(MeshRef),
    RenderObjects(Vec<RenderObject>),
}

#[derive(Debug, Clone)]
pub struct RenderObject {
    pub transform: Mat4,
    pub material: Option<Material>,
    pub child: RenderObjectChild,
}

impl RenderObject {
    pub fn mesh(mesh_ref: MeshRef) -> RenderObject {
        RenderObject {
            transform: Mat4::from_scale(1.0),
            material: None,
            child: RenderObjectChild::Mesh(mesh_ref),
        }
    }
    pub fn with_children(children: Vec<RenderObject>) -> RenderObject {
        RenderObject {
            transform: Mat4::from_scale(1.0),
            material: None,
            child: RenderObjectChild::RenderObjects(children),
        }
    }

    pub fn transform(&self, transform: Mat4) -> RenderObject {
        let mut new = self.clone();

        new.transform = transform * new.transform;

        new
    }

    pub fn translate(&self, v: V3) -> RenderObject {
        self.transform(Mat4::from_translation(v))
    }

    pub fn scale(&self, s: f32) -> RenderObject {
        self.transform(Mat4::from_scale(s))
    }

    pub fn scale_nonuniformly(&self, s: V3) -> RenderObject {
        self.transform(Mat4::from_nonuniform_scale(s.x, s.y, s.z))
    }

    #[allow(unused)]
    pub fn with_transform(&self, transform: Mat4) -> RenderObject {
        let mut new = self.clone();

        new.transform = transform;

        new
    }

    pub fn with_material(&self, material: Material) -> RenderObject {
        let mut new = self.clone();

        new.material = Some(material);

        new
    }

    // pub fn combined_transformed_verts(&self, meshes: &MeshStore, transform: &Mat4) -> Vec<V3> {
    //     let apply = |v: &V3, t: &Mat4| {
    //         let w = t * v4(v.x, v.y, v.z, 1.0);
    //         v3(w.x, w.y, w.z)
    //     };

    //     let transform = self.transform * transform;
    //     match &self.child {
    //         RenderObjectChild::Mesh(mesh_ref) => {
    //             let mesh = meshes.get_mesh(mesh_ref);
    //             mesh.simple_verts
    //                 .iter()
    //                 .map(|v| apply(v, &transform))
    //                 .collect()
    //         }
    //         RenderObjectChild::RenderObjects(children) => children
    //             .iter()
    //             .flat_map(|child| child.combined_transformed_verts(meshes, &transform))
    //             .collect(),
    //     }
    // }
    // pub fn raymarch_many<'a, I>(objs: I, meshes: &MeshStore, p: V3, r: V3) -> (usize, f32)
    // where
    //     I: Iterator<Item = &'a RenderObject>,
    // {
    //     let verts: Vec<(usize, V3)> = objs
    //         .enumerate()
    //         .flat_map::<Vec<_>, _>(|(i, obj)| {
    //             obj.combined_transformed_verts(meshes, &Mat4::from_scale(1.0))
    //                 .into_iter()
    //                 .map(|v| (i, v))
    //                 .collect()
    //         })
    //         .collect();

    //     verts.iter().fold((0, -1.0), |(j, d), (i, v)| {
    //         let vd = (v - p).normalize().dot(r);
    //         if vd > d {
    //             (*i, vd)
    //         } else {
    //             (j, d)
    //         }
    //     })
    // }
}

pub struct GRenderPass {
    pub fbo: Framebuffer,
    pub position: Texture,
    pub normal: Texture,
    pub albedo: Texture,
    pub emission: Texture,
    pub mrao: Texture,

    pub width: u32,
    pub height: u32,
}
impl GRenderPass {
    pub fn new(w: u32, h: u32) -> GRenderPass {
        let mut fbo = Framebuffer::new();
        let mut depth = Renderbuffer::new();
        depth
            .bind()
            .storage(TextureInternalFormat::DepthComponent, w, h);

        let (position, normal, albedo, emission, mrao) = {
            let buffer = fbo.bind();

            let create_texture = |internal, format, typ, attachment| {
                let tex = Texture::new(TextureKind::Texture2d);
                tex.bind()
                    .empty(TextureTarget::Texture2d, 0, internal, w, h, format, typ)
                    .parameter_int(TextureParameter::MinFilter, gl::LINEAR as i32)
                    .parameter_int(TextureParameter::MagFilter, gl::LINEAR as i32);
                buffer.texture_2d(attachment, TextureTarget::Texture2d, &tex, 0);
                tex
            };

            let position = create_texture(
                TextureInternalFormat::Rgb16f,
                TextureFormat::Rgb,
                GlType::Float,
                Attachment::Color0,
            );
            let normal = create_texture(
                TextureInternalFormat::Rgb16f,
                TextureFormat::Rgb,
                GlType::Float,
                Attachment::Color1,
            );
            let albedo = create_texture(
                TextureInternalFormat::Srgb8,
                TextureFormat::Rgb,
                GlType::UnsignedByte,
                Attachment::Color2,
            );
            let emission = create_texture(
                TextureInternalFormat::Srgb8,
                TextureFormat::Rgb,
                GlType::UnsignedByte,
                Attachment::Color3,
            );
            let mrao = create_texture(
                TextureInternalFormat::Rgba8,
                TextureFormat::Rgba,
                GlType::UnsignedByte,
                Attachment::Color4,
            );

            buffer
                .draw_buffers(&[
                    Attachment::Color0,
                    Attachment::Color1,
                    Attachment::Color2,
                    Attachment::Color3,
                    Attachment::Color4,
                ])
                .renderbuffer(Attachment::Depth, &depth);

            (position, normal, albedo, emission, mrao)
        };

        GRenderPass {
            fbo,
            // depth,
            position,
            normal,
            albedo,
            emission,
            mrao,

            width: w,
            height: h,
        }
    }
}

pub struct RenderTarget {
    pub width: u32,
    pub height: u32,
    pub framebuffer: Framebuffer,
    pub texture: Texture,
}

impl RenderTarget {
    pub fn new_with_format(
        width: u32,
        height: u32,
        internal_format: TextureInternalFormat,
        format: TextureFormat,
        typ: GlType,
    ) -> RenderTarget {
        let mut framebuffer = Framebuffer::new();
        let mut depth = Renderbuffer::new();
        let texture = Texture::new(TextureKind::Texture2d);
        texture
            .bind()
            .empty(
                TextureTarget::Texture2d,
                0,
                internal_format,
                width,
                height,
                format,
                typ,
            )
            .parameter_int(TextureParameter::MinFilter, gl::LINEAR as i32)
            .parameter_int(TextureParameter::MagFilter, gl::LINEAR as i32)
            .parameter_int(TextureParameter::WrapS, gl::CLAMP_TO_EDGE as i32)
            .parameter_int(TextureParameter::WrapT, gl::CLAMP_TO_EDGE as i32)
            .parameter_int(TextureParameter::WrapR, gl::CLAMP_TO_EDGE as i32);

        depth
            .bind()
            .storage(TextureInternalFormat::DepthComponent, width, height);

        framebuffer
            .bind()
            .texture_2d(Attachment::Color0, TextureTarget::Texture2d, &texture, 0)
            .draw_buffers(&[Attachment::Color0])
            .renderbuffer(Attachment::Depth, &depth);

        RenderTarget {
            width,
            height,
            framebuffer,
            texture,
        }
    }
    pub fn new(width: u32, height: u32) -> RenderTarget {
        RenderTarget::new_with_format(
            width,
            height,
            TextureInternalFormat::Srgb,
            TextureFormat::Rgb,
            GlType::UnsignedByte,
        )
    }

    pub fn set_viewport(&self) {
        unsafe {
            gl::Viewport(0, 0, self.width as i32, self.height as i32);
        }
    }

    pub fn bind(&mut self) -> FramebufferBinderReadDraw {
        self.framebuffer.bind()
    }

    #[allow(unused)]
    pub fn read(&mut self) -> FramebufferBinderRead {
        self.framebuffer.read()
    }

    #[allow(unused)]
    pub fn draw(&mut self) -> FramebufferBinderDraw {
        self.framebuffer.draw()
    }
}

pub trait Renderable {
    fn framebuffer(&self) -> &Framebuffer;
    fn framebuffer_mut(&mut self) -> &mut Framebuffer;
    fn size(&self) -> (u32, u32);
    fn framebuffer_read(&mut self) -> FramebufferBinderRead {
        self.framebuffer_mut().read()
    }
    fn framebuffer_draw(&mut self) -> FramebufferBinderDraw {
        self.framebuffer_mut().draw()
    }
    fn bounds(&self) -> (u32, u32, u32, u32) {
        let (w, h) = self.size();

        (0, 0, w, h)
    }
    fn blit_to<T>(&mut self, target: &mut T, mask: Mask, filter: u32)
    where
        T: Renderable,
    {
        let sb = self.bounds();
        let tb = target.bounds();

        target
            .framebuffer_draw()
            .blit_framebuffer(&self.framebuffer_read(), sb, tb, mask, filter);
    }
}

impl Renderable for GRenderPass {
    fn framebuffer(&self) -> &Framebuffer {
        &self.fbo
    }
    fn framebuffer_mut(&mut self) -> &mut Framebuffer {
        &mut self.fbo
    }
    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

impl Renderable for RenderTarget {
    fn framebuffer(&self) -> &Framebuffer {
        &self.framebuffer
    }
    fn framebuffer_mut(&mut self) -> &mut Framebuffer {
        &mut self.framebuffer
    }
    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

#[allow(unused)]
pub struct CubeMapBuilder<T> {
    pub back: T,
    pub front: T,
    pub right: T,
    pub bottom: T,
    pub left: T,
    pub top: T,
}

#[allow(unused)]
impl<'a> CubeMapBuilder<&'a str> {
    pub fn build(self) -> Texture {
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

            for (target, path) in &faces {
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

            tex.parameter_int(TextureParameter::MinFilter, gl::LINEAR as i32)
                .parameter_int(TextureParameter::MagFilter, gl::LINEAR as i32)
                .parameter_int(TextureParameter::WrapS, gl::CLAMP_TO_EDGE as i32)
                .parameter_int(TextureParameter::WrapT, gl::CLAMP_TO_EDGE as i32)
                .parameter_int(TextureParameter::WrapR, gl::CLAMP_TO_EDGE as i32);
        }

        texture
    }
}

pub fn map_cubemap<P>(
    size: u32,
    mip_levels: usize,
    program: &P,
    mut render_cube: impl FnMut(&FramebufferBinderReadDraw, &P),
) -> Texture
where
    P: ProgramBind,
{
    // Prepare
    let mut fbo = Framebuffer::new();
    let mut rbo = Renderbuffer::new();
    rbo.bind()
        .storage(TextureInternalFormat::DepthComponent24, size, size);
    let map = Texture::new(TextureKind::CubeMap);
    let faces = TextureTarget::cubemap_faces();
    {
        let tex = map.bind();
        for face in &faces {
            tex.empty(
                *face,
                0,
                TextureInternalFormat::Rgb16f,
                size,
                size,
                TextureFormat::Rgb,
                GlType::Float,
            );
        }

        if mip_levels > 1 {
            tex.parameter_int(TextureParameter::MinFilter, gl::LINEAR_MIPMAP_LINEAR as i32)
        } else {
            tex.parameter_int(TextureParameter::MinFilter, gl::LINEAR as i32)
        }.parameter_int(TextureParameter::MagFilter, gl::LINEAR as i32)
            .parameter_int(TextureParameter::WrapS, gl::CLAMP_TO_EDGE as i32)
            .parameter_int(TextureParameter::WrapT, gl::CLAMP_TO_EDGE as i32)
            .parameter_int(TextureParameter::WrapR, gl::CLAMP_TO_EDGE as i32);

        if mip_levels > 1 {
            unsafe {
                gl::GenerateMipmap(gl::TEXTURE_CUBE_MAP);
            }
        }
    }

    fbo.bind()
        .renderbuffer(Attachment::Depth, &rbo)
        .check_status()
        .expect("framebuffer for map_cubemap is not ready");

    // Render

    let capture_perspective: Mat4 = cgmath::PerspectiveFov {
        fovy: Rad(std::f32::consts::PI / 2.0),
        aspect: 1.0,
        near: 0.1,
        far: 10.0,
    }.into();

    let origo = P3::new(0.0, 0.0, 0.0);
    let lp = origo;
    let look_at = |p, up| Mat4::look_at(lp, lp + p, up);

    let transforms: [Mat4; 6] = [
        look_at(v3(1.0, 0.0, 0.0), v3(0.0, -1.0, 0.0)),
        look_at(v3(-1.0, 0.0, 0.0), v3(0.0, -1.0, 0.0)),
        look_at(v3(0.0, 1.0, 0.0), v3(0.0, 0.0, 1.0)),
        look_at(v3(0.0, -1.0, 0.0), v3(0.0, 0.0, -1.0)),
        look_at(v3(0.0, 0.0, 1.0), v3(0.0, -1.0, 0.0)),
        look_at(v3(0.0, 0.0, -1.0), v3(0.0, -1.0, 0.0)),
    ];

    program.bind_mat4("projection", capture_perspective);

    {
        unsafe {
            gl::Disable(gl::CULL_FACE);
        }
        let fbo = fbo.bind();
        for mip in 0..mip_levels {
            let mip_scale = (0.5 as f32).powf(mip as f32);
            let mip_size = (size as f32 * mip_scale) as u32;

            rbo.bind()
                .storage(TextureInternalFormat::DepthComponent24, mip_size, mip_size);

            unsafe {
                gl::Viewport(0, 0, mip_size as i32, mip_size as i32);
            }

            let roughness = if mip_levels > 1 {
                mip as f32 / (mip_levels - 1) as f32
            } else {
                0.0
            };
            program.bind_float("roughness", roughness);

            for (i, face) in faces.iter().enumerate() {
                program.bind_mat4("view", transforms[i]);
                fbo.texture_2d(Attachment::Color0, *face, &map, mip)
                    .clear(Mask::ColorDepth);
                GlError::check().expect("pre render cube");
                render_cube(&fbo, program);
                GlError::check().expect("post render cube");
            }
        }
        unsafe {
            gl::Enable(gl::CULL_FACE);
        }
    }

    GlError::check().expect("falied to make cubemap from equirectangular");

    map
}
pub fn cubemap_from_equirectangular<F, P>(size: u32, program: &P, render_cube: F) -> Texture
where
    F: FnMut(&FramebufferBinderReadDraw, &P),
    P: ProgramBind,
{
    let tex = map_cubemap(size, 1, program, render_cube);
    tex.bind()
        .parameter_int(TextureParameter::MinFilter, gl::LINEAR_MIPMAP_LINEAR as i32);
    tex
}
pub fn create_prefiler_map<F, P>(size: u32, program: &P, render_cube: F) -> Texture
where
    F: FnMut(&FramebufferBinderReadDraw, &P),
    P: ProgramBind,
{
    map_cubemap(size, 5, program, render_cube)
}
pub fn create_irradiance_map<F, P>(size: u32, program: &P, render_cube: F) -> Texture
where
    F: FnMut(&FramebufferBinderReadDraw, &P),
    P: ProgramBind,
{
    map_cubemap(size, 1, program, render_cube)
}

pub struct Camera {
    pub pos: V3,
    pub fov: Rad<f32>,
    pub aspect: f32,
    pub yaw: f32,
    pub pitch: f32,
}

impl Camera {
    pub fn new(pos: V3, fov: Rad<f32>, aspect: f32) -> Camera {
        Camera {
            pos,
            fov,
            aspect,
            yaw: 0.0,
            pitch: 0.0,
        }
    }

    pub fn up(&self) -> V3 {
        v3(0.0, 1.0, 0.0)
    }
    pub fn front(&self) -> V3 {
        let (ps, pc) = self.pitch.sin_cos();
        let (ys, yc) = self.yaw.sin_cos();

        v3(pc * yc, ps, pc * ys).normalize()
    }
    #[allow(unused)]
    pub fn front_look_at(&self, target: &V3) -> V3 {
        (target - self.pos).normalize()
    }
    pub fn get_view(&self) -> Mat4 {
        let origo = cgmath::Point3::new(0.0, 0.0, 0.0);
        Mat4::look_at(origo + self.pos, origo + self.pos + self.front(), self.up())
    }
    #[allow(unused)]
    pub fn get_view_look_at(&self, target: &V3) -> Mat4 {
        let origo = cgmath::Point3::new(0.0, 0.0, 0.0);
        Mat4::look_at(
            origo + self.pos,
            origo + self.pos + self.front_look_at(target),
            self.up(),
        )
    }
    pub fn get_projection(&self) -> Mat4 {
        cgmath::PerspectiveFov {
            fovy: self.fov,
            aspect: self.aspect,
            near: 0.1,
            far: 100.0,
        }.into()
    }
}

pub fn rgb(r: u8, g: u8, b: u8) -> V3 {
    v3(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0)
}

pub fn clamp<T: PartialOrd>(input: T, min: T, max: T) -> T {
    debug_assert!(min <= max, "min must be less than or equal to max");
    if input < min {
        min
    } else if input > max {
        max
    } else {
        input
    }
}

pub fn hsv(h: f32, s: f32, v: f32) -> V3 {
    let r = if (h % 1.0) < 0.5 {
        (clamp(-6.0 * (h % 1.0) + 2.0, 0.0, 1.0) * s + 1.0 - s) * v
    } else {
        (clamp(6.0 * (h % 1.0) - 4.0, 0.0, 1.0) * s + 1.0 - s) * v
    };

    let g = if (h % 1.0) < 1.0 / 3.0 {
        (clamp(6.0 * (h % 1.0), 0.0, 1.0) * s + 1.0 - s) * v
    } else {
        (clamp(-6.0 * (h % 1.0) + 4.0, 0.0, 1.0) * s + 1.0 - s) * v
    };

    let b = if (h % 1.0) < 2.0 / 0.3 {
        (clamp(6.0 * (h % 1.0) - 2.0, 0.0, 1.0) * s + 1.0 - s) * v
    } else {
        (clamp(-6.0 * (h % 1.0) + 6.0, 0.0, 1.0) * s + 1.0 - s) * v
    };

    v3(r, g, b)
}

pub fn hex(v: u32) -> V3 {
    rgb((v >> 16) as u8, (v >> 8) as u8, v as u8)
}
