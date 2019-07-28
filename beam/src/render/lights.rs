use crate::misc::{v3, Mat4, P3, V3};
use crate::render::dsl::UniformValue;
use cgmath::{self, Rad};
use gl;
use mg::{
    Attachment, BufferSlot, Framebuffer, FramebufferBinderBase, FramebufferBinderDrawer,
    FramebufferBinderReadDraw, GlError, GlType, ProgramBind, ProgramBinding, Texture,
    TextureFormat, TextureInternalFormat, TextureKind, TextureParameter, TextureTarget,
};
use std::{borrow::Cow, f32::consts::PI};

#[repr(C)]
#[derive(Debug)]
pub struct DirectionalLight {
    pub color: V3,

    pub direction: V3,

    pub shadow_map: ShadowMap,
}
impl DirectionalLight {
    pub fn space(&self, camera_pos: V3) -> Mat4 {
        let size = 50.0;
        let projection: Mat4 = cgmath::Ortho {
            left: -size,
            right: size,
            bottom: -size,
            top: size,
            near: 0.01,
            far: 300.0,
        }
        .into();
        let origo = P3::new(0.0, 0.0, 0.0);
        let o = origo + camera_pos + self.direction * 25.0;
        let view = Mat4::look_at(o, o - self.direction, v3(0.0, 1.0, 0.0));
        projection * view
    }
    pub fn bind_new<'a>(
        &'a self,
        camera_pos: V3,
        name: &str,
    ) -> Vec<(Cow<'a, str>, UniformValue<'a>)> {
        let space = self.space(camera_pos);
        let DirectionalLight {
            color,
            direction,
            shadow_map,
        } = self;
        vec![
            (format!("{}.{}", name, "color").into(), (*color).into()),
            (
                format!("{}.{}", name, "direction").into(),
                (*direction).into(),
            ),
            (format!("{}.{}", name, "space").into(), space.into()),
            ("directionalShadowMap".into(), (&shadow_map.map).into()),
        ]
    }
    pub fn bind_multiple_new<'a>(
        camera_pos: V3,
        lights: &'a [DirectionalLight],
        name_uniform: &str,
        amt_uniform: &'a str,
    ) -> Vec<(Cow<'a, str>, UniformValue<'a>)> {
        let mut uniforms = vec![(amt_uniform.into(), (lights.len() as i32).into())];
        for (i, light) in lights.iter().enumerate() {
            let mut new = light.bind_new(camera_pos, &format!("{}[{}]", name_uniform, i));
            uniforms.append(&mut new);
        }
        uniforms
    }
}
#[repr(C)]
#[derive(Debug)]
pub struct PointLight {
    pub color: V3,

    pub position: V3,
    pub last_shadow_map_position: V3,

    pub shadow_map: Option<PointShadowMap>,
}
impl PointLight {
    fn bind_new<'a>(&'a self, name: &str) -> Vec<(Cow<'a, str>, UniformValue<'a>)> {
        let ext = |e| format!("{}.{}", name, e).into();
        let PointLight {
            position,
            color,
            shadow_map,
            last_shadow_map_position,
        } = self;

        let mut uniforms = vec![
            (ext("color"), (*color).into()),
            (ext("position"), (*position).into()),
            (ext("lastPosition"), (*last_shadow_map_position).into()),
        ];

        let mut new = match shadow_map {
            Some(shadow_map) => vec![
                (ext("useShadowMap"), true.into()),
                ("pointShadowMap".into(), (&shadow_map.map).into()),
                (ext("farPlane"), shadow_map.far.into()),
            ],
            None => vec![
                (ext("lastPosition"), (*position).into()),
                (ext("useShadowMap"), false.into()),
            ],
        };
        uniforms.append(&mut new);
        uniforms
    }
    pub fn bind_multiple_new<'a>(
        lights: &'a [PointLight],
        name_uniform: &str,
        amt_uniform: &'a str,
    ) -> Vec<(Cow<'a, str>, UniformValue<'a>)> {
        let mut uniforms = vec![(amt_uniform.into(), (lights.len() as i32).into())];
        for (i, light) in lights.iter().enumerate() {
            let mut new = light.bind_new(&format!("{}[{}]", name_uniform, i));
            uniforms.append(&mut new);
        }
        uniforms
    }
    pub fn space(&self) -> Option<[[[f32; 4]; 4]; 6]> {
        let shadow_map = self.shadow_map.as_ref()?;

        let light_space: Mat4 = cgmath::PerspectiveFov {
            fovy: Rad(PI / 2.0),
            aspect: (shadow_map.width as f32) / (shadow_map.height as f32),
            near: shadow_map.near,
            far: shadow_map.far,
        }
        .into();

        let origo = P3::new(0.0, 0.0, 0.0);
        let lp = origo + self.last_shadow_map_position;
        let look_at = |p, up| (light_space * Mat4::look_at(lp, lp + p, up)).into();

        let shadow_transforms = [
            look_at(v3(1.0, 0.0, 0.0), v3(0.0, -1.0, 0.0)),
            look_at(v3(-1.0, 0.0, 0.0), v3(0.0, -1.0, 0.0)),
            look_at(v3(0.0, 1.0, 0.0), v3(0.0, 0.0, 1.0)),
            look_at(v3(0.0, -1.0, 0.0), v3(0.0, 0.0, -1.0)),
            look_at(v3(0.0, 0.0, 1.0), v3(0.0, -1.0, 0.0)),
            look_at(v3(0.0, 0.0, -1.0), v3(0.0, -1.0, 0.0)),
        ];

        Some(shadow_transforms)
    }
}
#[repr(C)]
#[derive(Debug, Clone)]
pub struct SpotLight {
    pub color: V3,

    pub position: V3,
    pub direction: V3,

    pub cut_off: Rad<f32>,
    pub outer_cut_off: Rad<f32>,
}
impl SpotLight {
    pub fn bind_new<'a>(&'a self, name: &str) -> Vec<(Cow<'a, str>, UniformValue<'a>)> {
        let ext = |e| format!("{}.{}", name, e).into();
        let SpotLight {
            position,
            color,
            direction,
            cut_off,
            outer_cut_off,
        } = self;
        vec![
            (ext("color"), (*color).into()),
            (ext("position"), (*position).into()),
            (ext("direction"), (*direction).into()),
            (ext("cutOff"), cut_off.0.cos().into()),
            (ext("outerCutOff"), outer_cut_off.0.cos().into()),
        ]
    }
    pub fn bind_multiple_new<'a>(
        lights: &'a [SpotLight],
        name_uniform: &str,
        amt_uniform: &'a str,
    ) -> Vec<(Cow<'a, str>, UniformValue<'a>)> {
        let mut uniforms = vec![(amt_uniform.into(), (lights.len() as i32).into())];
        for (i, light) in lights.iter().enumerate() {
            let mut new = light.bind_new(&format!("{}[{}]", name_uniform, i));
            uniforms.append(&mut new);
        }
        uniforms
    }
}

#[derive(Debug)]
pub struct ShadowMap {
    width: u32,
    height: u32,
    pub fbo: Framebuffer,
    map: Texture,
}

impl ShadowMap {
    pub fn new() -> ShadowMap {
        let (width, height) = ShadowMap::size();
        let mut fbo = Framebuffer::new();
        let map = Texture::new(TextureKind::Texture2d);
        map.bind()
            .empty(
                TextureTarget::Texture2d,
                0,
                TextureInternalFormat::DepthComponent,
                width,
                height,
                TextureFormat::DepthComponent,
                GlType::Float,
            )
            .parameter_int(TextureParameter::MinFilter, gl::LINEAR as i32)
            .parameter_int(TextureParameter::MagFilter, gl::LINEAR as i32)
            .parameter_int(TextureParameter::WrapS, gl::REPEAT as i32)
            .parameter_int(TextureParameter::WrapT, gl::REPEAT as i32);
        fbo.bind()
            .texture_2d(Attachment::Depth, TextureTarget::Texture2d, &map, 0)
            .draw_buffer(BufferSlot::None)
            .read_buffer(BufferSlot::None);
        ShadowMap {
            width,
            height,
            fbo,
            map,
        }
    }
    pub fn size() -> (u32, u32) {
        // (1024, 1024)
        (2048, 2048)
        // (4096, 4096)
        // (8192, 8192)
    }
}

#[derive(Debug)]
pub struct PointShadowMap {
    width: u32,
    height: u32,
    near: f32,
    pub far: f32,
    pub fbo: Framebuffer,
    map: Texture,
}

impl PointShadowMap {
    pub fn new() -> PointShadowMap {
        let (width, height) = PointShadowMap::size();
        let mut fbo = Framebuffer::new();
        let map = Texture::new(TextureKind::CubeMap);
        {
            let tex = map.bind();
            let faces = TextureTarget::cubemap_faces();

            for face in &faces {
                tex.empty(
                    *face,
                    0,
                    TextureInternalFormat::DepthComponent,
                    width,
                    height,
                    TextureFormat::DepthComponent,
                    GlType::UnsignedByte,
                );
            }

            tex.parameter_int(TextureParameter::MinFilter, gl::LINEAR as i32)
                .parameter_int(TextureParameter::MagFilter, gl::LINEAR as i32)
                .parameter_int(TextureParameter::WrapS, gl::CLAMP_TO_EDGE as i32)
                .parameter_int(TextureParameter::WrapT, gl::CLAMP_TO_EDGE as i32)
                .parameter_int(TextureParameter::WrapR, gl::CLAMP_TO_EDGE as i32);
        }
        fbo.bind()
            .texture(Attachment::Depth, &map, 0)
            .draw_buffer(BufferSlot::None)
            .read_buffer(BufferSlot::None)
            .check_status()
            .expect("point shadow map framebuffer not complete");

        let near = 0.6;
        let far = 100.0;

        PointShadowMap {
            width,
            height,
            near,
            far,
            fbo,
            map,
        }
    }
    pub fn size() -> (u32, u32) {
        // (128, 128)
        // (256, 256)
        (512, 512)
        // (1024, 1024)
        // (2048, 2048)
        // (4096, 4096)
        // (8192, 8192)
    }
}
