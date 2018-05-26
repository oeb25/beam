use cgmath::{self, Rad};
use gl;
use mg::{
    Attachment, BufferSlot, Framebuffer, FramebufferBinderBase, FramebufferBinderDrawer,
    FramebufferBinderReadDraw, GlError, GlType, ProgramBinding, Texture, TextureFormat,
    TextureInternalFormat, TextureKind, TextureParameter, TextureTarget,
};
use misc::{v3, Mat4, P3, V3};
use std::f32::consts::PI;

#[repr(C)]
#[derive(Debug)]
pub struct DirectionalLight {
    pub color: V3,

    pub direction: V3,

    pub shadow_map: ShadowMap,
}
impl DirectionalLight {
    fn space(&self, camera_pos: V3) -> Mat4 {
        let size = 50.0;
        let projection: Mat4 = cgmath::Ortho {
            left: -size,
            right: size,
            bottom: -size,
            top: size,
            near: 0.01,
            far: 300.0,
        }.into();
        let origo = P3::new(0.0, 0.0, 0.0);
        let o = origo + camera_pos + self.direction * 100.0;
        let view = Mat4::look_at(o, o - self.direction, v3(0.0, 1.0, 0.0));
        projection * view
    }
    fn bind(&self, camera_pos: V3, name: &str, program: &ProgramBinding) {
        let ext = |e| format!("{}.{}", name, e);
        let space = self.space(camera_pos);
        let DirectionalLight {
            color,
            direction,
            shadow_map,
        } = self;
        program
            .bind_vec3(&ext("color"), *color)
            .bind_vec3(&ext("direction"), *direction)
            .bind_mat4(&ext("space"), space)
            .bind_texture("directionalShadowMap", &shadow_map.map);
    }
    pub fn bind_multiple(
        camera_pos: V3,
        lights: &[DirectionalLight],
        name_uniform: &str,
        amt_uniform: &str,
        program: &ProgramBinding,
    ) {
        program.bind_int(amt_uniform, lights.len() as i32);
        for (i, light) in lights.iter().enumerate() {
            light.bind(camera_pos, &format!("{}[{}]", name_uniform, i), program);
        }
    }
    pub fn bind_shadow_map(&mut self, camera_pos: V3) -> (FramebufferBinderReadDraw, Mat4) {
        let light_space = self.space(camera_pos);
        (self.shadow_map.fbo.bind(), light_space)
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
    fn bind(&self, name: &str, program: &ProgramBinding) {
        let ext = |e| {
            let res = format!("{}.{}", name, e);
            res
        };
        let PointLight {
            position,
            color,
            shadow_map,
            last_shadow_map_position,
        } = self;
        program.bind_vec3(&ext("color"), *color);

        program.bind_vec3(&ext("position"), *position);
        program.bind_vec3(&ext("lastPosition"), *last_shadow_map_position);

        match shadow_map {
            Some(shadow_map) => {
                program.bind_bool(&ext("useShadowMap"), true);
                program.bind_texture("pointShadowMap", &shadow_map.map);
                program.bind_float(&ext("farPlane"), shadow_map.far);
            }
            None => {
                program.bind_vec3(&ext("lastPosition"), *position);
                program.bind_bool(&ext("useShadowMap"), false);
            }
        }
        GlError::check().expect(&format!("Failed to bind light: {:?}", self));
    }
    pub fn bind_multiple(
        lights: &[PointLight],
        name_uniform: &str,
        amt_uniform: &str,
        program: &ProgramBinding,
    ) {
        program.bind_int(amt_uniform, lights.len() as i32);
        GlError::check().expect("Failed to bind number of lights");
        for (i, light) in lights.iter().enumerate() {
            // println!("binding: {} into {:?}", format!("{}[{}]", name_uniform, i), slot);
            light.bind(&format!("{}[{}]", name_uniform, i), program);
        }
        GlError::check().expect("Failed to bind multiple lights");
    }
    pub fn bind_shadow_map(&mut self) -> Option<(FramebufferBinderReadDraw, [[[f32; 4]; 4]; 6])> {
        let shadow_map = self.shadow_map.as_mut()?;

        let light_space: Mat4 = cgmath::PerspectiveFov {
            fovy: Rad(PI / 2.0),
            aspect: (shadow_map.width as f32) / (shadow_map.height as f32),
            near: shadow_map.near,
            far: shadow_map.far,
        }.into();

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

        Some((shadow_map.fbo.bind(), shadow_transforms))
    }
}
#[repr(C)]
#[derive(Debug, Clone)]
struct SpotLight {
    ambient: V3,
    diffuse: V3,
    specular: V3,

    position: V3,
    direction: V3,

    cut_off: Rad<f32>,
    outer_cut_off: Rad<f32>,

    constant: f32,
    linear: f32,
    quadratic: f32,
}
impl SpotLight {
    #[allow(unused)]
    fn bind(&self, name: &str, program: &ProgramBinding) {
        let ext = |e| format!("{}.{}", name, e);
        let SpotLight {
            position,
            ambient,
            diffuse,
            specular,
            direction,
            cut_off,
            outer_cut_off,
            constant,
            linear,
            quadratic,
        } = self;
        program.bind_vec3(&ext("ambient"), *ambient);
        program.bind_vec3(&ext("diffuse"), *diffuse);
        program.bind_vec3(&ext("specular"), *specular);

        program.bind_vec3(&ext("position"), *position);
        program.bind_vec3(&ext("direction"), *direction);

        program.bind_float(&ext("cutOff"), cut_off.0.cos());
        program.bind_float(&ext("outerCutOff"), outer_cut_off.0.cos());

        program.bind_float(&ext("constant"), *constant);
        program.bind_float(&ext("linear"), *linear);
        program.bind_float(&ext("quadratic"), *quadratic);
    }
}

#[derive(Debug)]
pub struct ShadowMap {
    width: u32,
    height: u32,
    fbo: Framebuffer,
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
    fbo: Framebuffer,
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

            for face in faces.into_iter() {
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
