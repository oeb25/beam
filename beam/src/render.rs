use cgmath::{self, InnerSpace, Rad};
use collada;
use gl;
use image;

use std::{self, collections::HashMap, mem, path::Path};

use mg::*;

pub type V2 = cgmath::Vector2<f32>;
pub type V3 = cgmath::Vector3<f32>;
pub type V4 = cgmath::Vector4<f32>;
pub type P3 = cgmath::Point3<f32>;
pub type Mat3 = cgmath::Matrix3<f32>;
pub type Mat4 = cgmath::Matrix4<f32>;

macro_rules! offset_of {
    ($ty:ty, $field:ident) => {
        #[allow(unused_unsafe)]
        unsafe {
            &(*(0 as *const $ty)).$field as *const _ as usize
        }
    };
}

#[allow(unused_macros)]
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

pub fn v2(x: f32, y: f32) -> V2 {
    V2::new(x, y)
}

pub fn v3(x: f32, y: f32, z: f32) -> V3 {
    V3::new(x, y, z)
}

pub fn v4(x: f32, y: f32, z: f32, w: f32) -> V4 {
    V4::new(x, y, z, w)
}

pub trait Vertexable {
    // (pos, norm, tex, trangent, bitangent)
    fn sizes() -> (usize, usize, usize, usize);
    fn offsets() -> (usize, usize, usize, usize);
}

#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub pos: V3,
    pub norm: V3,
    pub tex: V2,
    pub tangent: V3,
    pub bitangent: V3,
}

impl Default for Vertex {
    fn default() -> Vertex {
        let zero3 = v3(0.0, 0.0, 0.0);
        Vertex {
            pos: zero3,
            norm: zero3,
            tex: v2(0.0, 0.0),
            tangent: zero3,
            bitangent: zero3,
        }
    }
}

pub struct Mesh {
    vcount: usize,
    vao: VertexArray,
}

impl Mesh {
    pub fn new(vertices: &[Vertex]) -> Mesh {
        let mut vao = VertexArray::new();
        let mut vbo = VertexBuffer::from_data(vertices);

        {
            let float_size = mem::size_of::<f32>();
            let vao_binder = vao.bind();
            let vbo_binder = vbo.bind();

            macro_rules! x {
                ($i:expr, $e:ident) => (
                    vao_binder.vbo_attrib(
                        &vbo_binder,
                        $i,
                        size_of!(Vertex, $e) / float_size,
                        offset_of!(Vertex, $e)
                    )
                )
            }

            x!(0, pos);
            x!(1, norm);
            x!(2, tex);
            x!(3, tangent);
            x!(4, bitangent);
        }

        Mesh {
            vcount: vertices.len(),
            vao: vao,
        }
    }
    pub fn bind(&mut self) -> MeshBinding {
        MeshBinding(self)
    }
}

pub struct MeshBinding<'a>(&'a mut Mesh);
impl<'a> MeshBinding<'a> {
    pub fn draw<F>(&mut self, fbo: &F, program: &ProgramBinding)
    where
        F: FramebufferBinderDrawer,
    {
        // self.bind_textures(program);

        self.0
            .vao
            .bind()
            .draw_arrays(fbo, program, DrawMode::Triangles, 0, self.0.vcount);
    }
    pub fn draw_geometry_instanced<F>(
        &mut self,
        fbo: &F,
        _program: &ProgramBinding,
        transforms: &VertexBufferBinder<Mat4>,
    ) where
        F: FramebufferBinderDrawer,
    {
        let mut vao = self.0.vao.bind();
        let offset = 5;
        let width = 4;
        for i in 0..width {
            let index = i + offset;
            vao.vbo_attrib(&transforms, index, width, width * i * mem::size_of::<f32>())
                .attrib_divisor(index, 1);
        }

        vao.draw_arrays_instanced(fbo, DrawMode::Triangles, 0, self.0.vcount, transforms.len());
    }
    pub fn draw_instanced<F>(
        &mut self,
        fbo: &F,
        program: &ProgramBinding,
        transforms: &VertexBufferBinder<Mat4>,
    ) where
        F: FramebufferBinderDrawer,
    {
        // self.bind_textures(program);
        self.draw_geometry_instanced(fbo, program, transforms);
    }
}

pub fn calculate_tangent_and_bitangent(va: &mut Vertex, vb: &mut Vertex, vc: &mut Vertex) {
    let v1 = va.pos;
    let v2 = vb.pos;
    let v3 = vc.pos;

    let w1 = va.tex;
    let w2 = vb.tex;
    let w3 = vc.tex;

    let d1 = v2 - v1;
    let d2 = v3 - v1;
    let t1 = w2 - w1;
    let t2 = w3 - w1;

    let r = 1.0 / (t1.x * t2.y - t2.x * t1.y);
    let sdir = r * (t2.y * d1 - t1.y * d2);
    let tdir = r * (t1.x * d2 - t2.x * d1);

    va.tangent = sdir;
    va.bitangent = tdir;
    vb.tangent = sdir;
    vb.bitangent = tdir;
    vc.tangent = sdir;
    vc.bitangent = tdir;
}

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
            fovy: Rad(std::f32::consts::PI / 2.0),
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


#[derive(Debug, Clone, Copy)]
pub struct Material {
    pub normal: TextureRef,
    pub albedo: TextureRef,
    pub metallic: TextureRef,
    pub roughness: TextureRef,
    pub ao: TextureRef,
}

impl Material {
    pub fn bind(&self, meshes: &mut MeshStore, program: &ProgramBinding) {
        program
            .bind_texture("tex_albedo", meshes.get_texture(&self.albedo))
            .bind_texture("tex_metallic", meshes.get_texture(&self.metallic))
            .bind_texture("tex_roughness", meshes.get_texture(&self.roughness))
            .bind_texture("tex_normal", meshes.get_texture(&self.normal))
            .bind_texture("tex_ao", meshes.get_texture(&self.ao));
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TextureRef(usize);

#[derive(Default)]
pub struct MeshStore {
    meshes: Vec<Mesh>,
    textures: Vec<Texture>,

    fs_textures: HashMap<String, TextureRef>,

    rgb_textures: Vec<(V3, TextureRef)>,
    rgba_textures: Vec<(V4, TextureRef)>,

    // primitive cache
    cube: Option<MeshRef>,
    spheres: Vec<(f32, MeshRef)>,
}

#[derive(Debug, Clone, Copy)]
pub struct MeshRef(usize);

impl MeshStore {
    pub fn load_collada(
        &mut self,
        path: impl AsRef<Path>,
    ) -> RenderObject {
        let path = path.as_ref();

        let src = std::fs::read_to_string(path).expect("collada src file not found");
        let data: collada::Collada = collada::Collada::parse(&src).expect("failed to parse collada");

        let mut normal_src = None;
        let mut albedo_src = None;
        let mut metallic_src = None;
        let mut roughness_src = None;
        let mut ao_src = None;

        for image in data.images.iter() {
            match image.name.as_str() {
                "DIFF" => albedo_src = Some(path.with_file_name(&image.source)),
                "ROUGH" => roughness_src = Some(path.with_file_name(&image.source)),
                "MET" => metallic_src = Some(path.with_file_name(&image.source)),
                "NRM" => normal_src = Some(path.with_file_name(&image.source)),
                "AO" => ao_src = Some(path.with_file_name(&image.source)),
                _ => {},
            }
        }

        let custom_material = Material {
            normal: self.load_rgb(normal_src.expect("normal map not found :(")),
            albedo: self.load_rgb(albedo_src.expect("albedo map not found :(")),
            metallic: self.load_rgb(metallic_src.expect("metallic map not found :(")),
            roughness: self.load_rgb(roughness_src.expect("roughness map not found :(")),
            ao: self.rgb_texture(v3(1.0, 1.0, 1.0)),
            // ao: self.load_rgb(ao_src.expect("ao map not found :(")),
        };

        let mut mesh_ids: HashMap<usize, Vec<(MeshRef, Material)>> = HashMap::new();

        let mut render_objects = vec![];
        for node in data.visual_scenes[0].nodes.iter() {
            use cgmath::Matrix;
            let mut transform = node.transformations.iter().fold(Mat4::from_scale(1.0), |acc, t| {
                let t: Mat4 = match t {
                    collada::Transform::Matrix(a) => unsafe {
                        std::mem::transmute::<[f32; 16], [[f32; 4]; 4]>(*a).into()
                    },
                    _ => unimplemented!(),
                };

                t * acc
            });
            std::mem::swap(&mut transform.y, &mut transform.z);
            transform = transform.transpose();
            std::mem::swap(&mut transform.y, &mut transform.z);
            assert_eq!(node.geometry.len(), 1);
            for geom_instance in node.geometry.iter() {
                let mesh_refs = if let Some(mesh_cache) = mesh_ids.get(&geom_instance.geometry.0) {
                    mesh_cache.clone()
                } else {
                    let geom = &data.geometry[geom_instance.geometry.0];
                    let meshes = match geom {
                        collada::Geometry::Mesh { triangles }  => {
                            let mut meshes = vec![];

                            for triangles in triangles.iter() {
                                let collada::MeshTriangles {
                                    vertices,
                                    material,
                                } = triangles;

                                let mut verts: Vec<_> = vertices.iter().map(|v|
                                    {
                                        let pos = v4(v.pos[0], v.pos[1], v.pos[2], 1.0);
                                        Vertex {
                                            // orient y up instead of z, which is default in blender
                                            pos: v3(pos[0], pos[2], pos[1]),
                                            norm: v3(v.nor[0], v.nor[2], v.nor[1]),
                                            tex: v.tex.into(),
                                            tangent: v3(0.0, 0.0, 0.0),
                                            bitangent: v3(0.0, 0.0, 0.0),
                                        }
                                    }).collect();

                                for vs in verts.chunks_mut(3) {
                                    // flip vertex clockwise direction
                                    vs.swap(1, 2);

                                    let x: *mut _ = &mut vs[0];
                                    let y: *mut _ = &mut vs[1];
                                    let z: *mut _ = &mut vs[2];
                                    unsafe {
                                        calculate_tangent_and_bitangent(&mut *x, &mut *y, &mut *z);
                                    }
                                }

                                let material = self.convert_collada_material(&path, &data, material);
                                let mesh = Mesh::new(&verts);
                                let mesh_ref = self.insert_mesh(mesh);

                                meshes.push((mesh_ref, material));
                            }
                            mesh_ids.insert(geom_instance.geometry.0, meshes.clone());
                            meshes
                        }
                        _ => unimplemented!()
                    };
                    meshes
                };

                // let material = geom_instance.material
                //     .map(|material_ref| self.convert_collada_material(&path, &data, material_ref));

                let children = mesh_refs.into_iter().map(|(mesh_ref, _material)| {
                    RenderObject::mesh(mesh_ref).with_material(custom_material)
                }).collect();

                render_objects.push(RenderObject {
                    transform,
                    material: None,
                    child: RenderObjectChild::RenderObjects(children)
                });
            }
        }

        RenderObject::with_children(render_objects)
    }

    fn convert_collada_material(
        &mut self,
        path: &Path,
        data: &collada::Collada,
        material_ref: &collada::MaterialRef,
    ) -> Material {
        let collada::Material::Effect(effect_ref) = data.materials[material_ref.0];
        let collada::Effect::Phong {
            emission,
            ambient,
            diffuse,
            specular,
            shininess,
            index_of_refraction,
        } = &data.effects[effect_ref.0];

        let white = self.rgb_texture(v3(1.0, 1.0, 1.0));

        let mut  convert = |c: &collada::PhongProperty| {
            use collada::PhongProperty::*;
            match c {
                Color(color) => self.rgba_texture((*color).into()),
                Float(f) => self.rgb_texture(v3(*f, *f, *f)),
                Texture(image_ref) => {
                    let img = &data.images[image_ref.0];
                    // TODO: How do we determin what kind of file it is?
                    self.load_srgb(path.with_file_name(&img.source))
                },
            }
        };

        let albedo = diffuse.as_ref().map(convert).unwrap_or_else(|| white);

        Material {
            normal: self.rgb_texture(v3(0.5, 0.5, 1.0)),
            albedo,
            metallic: white,
            roughness: self.rgb_texture(v3(0.5, 0.5, 0.5)),
            ao: white,
        }
    }

    pub fn insert_mesh(&mut self, mesh: Mesh) -> MeshRef {
        let mesh_ref = MeshRef(self.meshes.len());
        self.meshes.push(mesh);
        mesh_ref
    }

    pub fn get_mesh_mut(&mut self, mesh_ref: &MeshRef) -> &mut Mesh {
        &mut self.meshes[mesh_ref.0]
    }

    pub fn insert_texture(&mut self, texture: Texture) -> TextureRef {
        let texture_ref = TextureRef(self.textures.len());
        self.textures.push(texture);
        texture_ref
    }

    pub fn get_texture(&mut self, texture_ref: &TextureRef) -> &Texture {
        &self.textures[texture_ref.0]
    }

    fn color_texture(&mut self, color: &[f32]) -> TextureRef {
        let texture = Texture::new(TextureKind::Texture2d);
        let is_rgb = color.len() == 3;
        unsafe {
            texture.bind()
                .parameter_int(TextureParameter::WrapS, gl::REPEAT as i32)
                .parameter_int(TextureParameter::WrapT, gl::REPEAT as i32)
                .parameter_int(TextureParameter::MinFilter, gl::LINEAR as i32)
                .parameter_int(TextureParameter::MagFilter, gl::LINEAR as i32)
                .image_2d(
                    TextureTarget::Texture2d,
                    0,
                    if is_rgb { TextureInternalFormat::Rgb } else { TextureInternalFormat::Rgba },
                    1,
                    1,
                    if is_rgb { TextureFormat::Rgb } else { TextureFormat::Rgba },
                    color,
                );
        }
        let id = TextureRef(self.textures.len());
        self.textures.push(texture);
        id
    }

    pub fn rgb_texture(&mut self, color: V3) -> TextureRef {
        self.rgb_textures.iter().find(|(value, _)| value == &color).map(|(_, id)| *id).unwrap_or_else(||{
            let id = self.color_texture(&[color.x, color.y, color.z]);
            self.rgb_textures.push((color, id));
            id
        })
    }

    pub fn rgba_texture(&mut self, color: V4) -> TextureRef {
        self.rgba_textures.iter().find(|(value, _)| value == &color).map(|(_, id)| *id).unwrap_or_else(||{
            let id = self.color_texture(&[color.x, color.y, color.z, color.w]);
            self.rgba_textures.push((color, id));
            id
        })
    }

    pub fn get_cube(&mut self) -> MeshRef {
        if let Some(mesh_ref) = self.cube {
            return mesh_ref;
        }

        let verts = cube_vertices();
        let mesh = Mesh::new(&verts);
        let mesh_ref = self.insert_mesh(mesh);
        self.cube = Some(mesh_ref);
        mesh_ref
    }

    pub fn get_sphere(&mut self, radius: f32) -> MeshRef {
        if let Some(cached) = self.spheres.iter().find(|(cache_level, _)| *cache_level == radius) {
            return cached.1;
        }

        let verts = sphere_verticies(radius, 24, 16);
        let mesh = Mesh::new(&verts);
        let mesh_ref = self.insert_mesh(mesh);
        self.spheres.push((radius, mesh_ref));
        mesh_ref
    }

    pub fn load_srgb(&mut self, path: impl AsRef<Path>) -> TextureRef {
        self.load(path, TextureInternalFormat::Srgb, TextureFormat::Rgb)
    }
    pub fn load_rgb(&mut self, path: impl AsRef<Path>) -> TextureRef {
        self.load(path, TextureInternalFormat::Rgb, TextureFormat::Rgb)
    }
    pub fn load_hdr(&mut self, path: impl AsRef<Path>) -> TextureRef {
        let path = path.as_ref();

        self.cache_or_load(path, || {
            use std::{fs::File, io::BufReader};
            let decoder = image::hdr::HDRDecoder::new(BufReader::new(File::open(path)?))?;
            let metadata = decoder.metadata();
            let data = decoder.read_image_hdr()?;
            let texture = Texture::new(TextureKind::Texture2d);
            unsafe {
                texture
                    .bind()
                    .parameter_int(TextureParameter::WrapS, gl::REPEAT as i32)
                    .parameter_int(TextureParameter::WrapT, gl::REPEAT as i32)
                    .parameter_int(TextureParameter::MinFilter, gl::LINEAR as i32)
                    .parameter_int(TextureParameter::MagFilter, gl::LINEAR as i32)
                    .image_2d(
                        TextureTarget::Texture2d,
                        0,
                        TextureInternalFormat::Rgba16,
                        metadata.width,
                        metadata.height,
                        TextureFormat::Rgb,
                        &data,
                    );
            }
            Ok(texture)
        })
    }
    pub fn load(
        &mut self,
        path: impl AsRef<Path>,
        internal_format: TextureInternalFormat,
        format: TextureFormat,
    ) -> TextureRef {
        let path = path.as_ref();

        self.cache_or_load(path, move || {
            let img = image::open(&path).expect(&format!("unable to read {:?}", path));
            let texture = Texture::new(TextureKind::Texture2d);
            texture
                .bind()
                .parameter_int(TextureParameter::WrapS, gl::REPEAT as i32)
                .parameter_int(TextureParameter::WrapT, gl::REPEAT as i32)
                .parameter_int(TextureParameter::MinFilter, gl::LINEAR as i32)
                .parameter_int(TextureParameter::MagFilter, gl::LINEAR as i32)
                .load_image(TextureTarget::Texture2d, internal_format, format, &img);
            Ok(texture)
        })
    }
    fn cache_or_load(
        &mut self,
        path: impl AsRef<Path>,
        f: impl FnOnce() -> Result<Texture, Box<std::error::Error>>,
    ) -> TextureRef {
        let path: &Path = path.as_ref();
        let path_str = path.to_str().unwrap();

        self.fs_textures
            .get(path_str)
            .map(|t| *t)
            .unwrap_or_else(|| {
                let texture = f().unwrap();
                let texture_ref = self.insert_texture(texture);
                self.fs_textures.insert(path_str.to_owned(), texture_ref);
                texture_ref
            })
    }

    pub fn load_pbr_with_default_filenames(&mut self, path: impl AsRef<Path>, extension: &str) -> Material {
        let path = path.as_ref();

        let mut load_srgb = |map| {
            let p = path.join(map).with_extension(extension);
            let tex = self.load_srgb(p);
            tex
        };
        let albedo = load_srgb("albedo");
        let mut load_rgb = |map| {
            let p = path.join(map).with_extension(extension);
            let tex = self.load_rgb(p);
            tex
        };

        Material {
            albedo,
            ao: load_rgb("ao"),
            metallic: load_rgb("metallic"),
            roughness: load_rgb("roughness"),
            normal: load_rgb("normal"),
        }
    }
}

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
}

pub struct GRenderPass {
    pub fbo: Framebuffer,
    pub position: Texture,
    pub normal: Texture,
    pub albedo: Texture,
    pub metallic_roughness_ao: Texture,

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

        let (position, normal, albedo, metallic_roughness_ao) = {
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
            let metallic_roughness_ao = create_texture(
                TextureInternalFormat::Rgb8,
                TextureFormat::Rgb,
                GlType::UnsignedByte,
                Attachment::Color3,
            );

            buffer
                .draw_buffers(&[
                    Attachment::Color0,
                    Attachment::Color1,
                    Attachment::Color2,
                    Attachment::Color3,
                ])
                .renderbuffer(Attachment::Depth, &depth);

            (position, normal, albedo, metallic_roughness_ao)
        };

        GRenderPass {
            fbo,
            // depth,
            position,
            normal,
            albedo,
            metallic_roughness_ao,

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
    pub fn new(width: u32, height: u32) -> RenderTarget {
        let mut framebuffer = Framebuffer::new();
        let mut depth = Renderbuffer::new();
        let texture = Texture::new(TextureKind::Texture2d);
        texture
            .bind()
            .empty(
                TextureTarget::Texture2d,
                0,
                TextureInternalFormat::Srgb,
                width,
                height,
                TextureFormat::Rgb,
                GlType::UnsignedByte,
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

pub struct CubeMapBuilder<T> {
    pub back: T,
    pub front: T,
    pub right: T,
    pub bottom: T,
    pub left: T,
    pub top: T,
}

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

            for (target, path) in faces.into_iter() {
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

pub fn map_cubemap(
    size: u32,
    mip_levels: usize,
    program: &ProgramBinding,
    mut render_cube: impl FnMut(&FramebufferBinderReadDraw, &ProgramBinding),
) -> Texture {
    // Prepare
    let mut fbo = Framebuffer::new();
    let mut rbo = Renderbuffer::new();
    rbo.bind()
        .storage(TextureInternalFormat::DepthComponent24, size, size);
    let map = Texture::new(TextureKind::CubeMap);
    let faces = TextureTarget::cubemap_faces();
    {
        let tex = map.bind();
        for face in faces.into_iter() {
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
    let look_at = |p, up| (Mat4::look_at(lp, lp + p, up)).into();

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
pub fn cubemap_from_equirectangular(
    size: u32,
    program: &ProgramBinding,
    render_cube: impl FnMut(&FramebufferBinderReadDraw, &ProgramBinding),
) -> Texture {
    map_cubemap(size, 1, program, render_cube)
}
pub fn cubemap_from_importance(
    size: u32,
    program: &ProgramBinding,
    render_cube: impl FnMut(&FramebufferBinderReadDraw, &ProgramBinding),
) -> Texture {
    map_cubemap(size, 5, program, render_cube)
}
pub fn convolute_cubemap(
    size: u32,
    program: &ProgramBinding,
    render_cube: impl FnMut(&FramebufferBinderReadDraw, &ProgramBinding),
) -> Texture {
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

macro_rules! v {
    ($pos:expr, $norm:expr, $tex:expr, $tangent:expr) => {{
        let tangent = $tangent.into();
        let norm = $norm.into();
        Vertex {
            pos: $pos.into(),
            tex: $tex.into(),
            norm: norm,
            tangent: tangent,
            bitangent: tangent.cross(norm),
        }
    }};
}

fn cube_vertices() -> Vec<Vertex> {
    vec![
        // Back face
        v!(
            [-0.5, -0.5, -0.5],
            [0.0, 0.0, -1.0],
            [0.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [0.5, 0.5, -0.5],
            [0.0, 0.0, -1.0],
            [1.0, 1.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [0.5, -0.5, -0.5],
            [0.0, 0.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [0.5, 0.5, -0.5],
            [0.0, 0.0, -1.0],
            [1.0, 1.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [-0.5, -0.5, -0.5],
            [0.0, 0.0, -1.0],
            [0.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [-0.5, 0.5, -0.5],
            [0.0, 0.0, -1.0],
            [0.0, 1.0],
            [0.0, 1.0, 0.0]
        ),
        // Front face
        v!(
            [-0.5, -0.5, 0.5],
            [0.0, 0.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [0.5, -0.5, 0.5],
            [0.0, 0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 1.0],
            [1.0, 1.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 1.0],
            [1.0, 1.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [-0.5, 0.5, 0.5],
            [0.0, 0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [-0.5, -0.5, 0.5],
            [0.0, 0.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        // Left face
        v!(
            [-0.5, 0.5, 0.5],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [-0.5, 0.5, -0.5],
            [-1.0, 0.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [-0.5, -0.5, -0.5],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [-0.5, -0.5, -0.5],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [-0.5, -0.5, 0.5],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [-0.5, 0.5, 0.5],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        // Right face
        v!(
            [0.5, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [0.5, -0.5, -0.5],
            [1.0, 0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [0.5, 0.5, -0.5],
            [1.0, 0.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [0.5, -0.5, -0.5],
            [1.0, 0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [0.5, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [0.5, -0.5, 0.5],
            [1.0, 0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        // Bottom face
        v!(
            [-0.5, -0.5, -0.5],
            [0.0, -1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0, 0.0]
        ),
        v!(
            [0.5, -0.5, -0.5],
            [0.0, -1.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0, 0.0]
        ),
        v!(
            [0.5, -0.5, 0.5],
            [0.0, -1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0, 0.0]
        ),
        v!(
            [0.5, -0.5, 0.5],
            [0.0, -1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0, 0.0]
        ),
        v!(
            [-0.5, -0.5, 0.5],
            [0.0, -1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0, 0.0]
        ),
        v!(
            [-0.5, -0.5, -0.5],
            [0.0, -1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0, 0.0]
        ),
        // Top face
        v!(
            [-0.5, 0.5, -0.5],
            [0.0, 1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0, 0.0]
        ),
        v!(
            [0.5, 0.5, 0.5],
            [0.0, 1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0, 0.0]
        ),
        v!(
            [0.5, 0.5, -0.5],
            [0.0, 1.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0, 0.0]
        ),
        v!(
            [0.5, 0.5, 0.5],
            [0.0, 1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0, 0.0]
        ),
        v!(
            [-0.5, 0.5, -0.5],
            [0.0, 1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0, 0.0]
        ),
        v!(
            [-0.5, 0.5, 0.5],
            [0.0, 1.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0, 0.0]
        ),
    ]
}
fn sphere_verticies(radius: f32, nb_long: usize, nb_lat: usize) -> Vec<Vertex> {
    use std::f32::consts::PI;

    let mut verts: Vec<Vertex> = vec![Vertex::default(); (nb_long + 1) * nb_lat + 2];

    let up = v3(0.0, 1.0, 0.0) * radius;
    verts[0] = Vertex {
        pos: up,
        norm: up,
        tex: v2(0.0, 0.0),
        tangent: v3(0.0, 0.0, 0.0),
        bitangent: v3(0.0, 0.0, 0.0),
    };
    for lat in 0..nb_lat {
        let a1 = PI * (lat as f32 + 1.0) / (nb_lat as f32 + 1.0);
        let (sin1, cos1) = a1.sin_cos();

        for lon in 0..=nb_long {
            let a2 = PI * 2.0 * (if lon == nb_long { 0.0 } else { lon as f32 }) / nb_long as f32;
            let (sin2, cos2) = a2.sin_cos();

            let pos = v3(
                sin1 * cos2,
                cos1,
                sin1 * sin2,
            );
            let norm = pos;
            let tex = v2(
                lon as f32 / nb_long as f32,
                1.0 - (lat as f32 + 1.0) / (nb_lat as f32 + 1.0)
            );

            verts[lon + lat * (nb_long + 1) + 1] = Vertex {
                pos: pos * radius,
                norm,
                tex,
                tangent: v3(0.0, 0.0, 0.0),
                bitangent: v3(0.0, 0.0, 0.0),
            };
        }
    }
    let len = verts.len();
    verts[len - 1] = Vertex {
        pos: -up,
        norm: -up,
        tex: v2(0.0, 0.0),
        tangent: v3(0.0, 0.0, 0.0),
        bitangent: v3(0.0, 0.0, 0.0),
    };

    let nb_faces = verts.len();
    let nb_triangles = nb_faces * 2;
    let nb_indices = nb_triangles * 3;

    let mut new_verts: Vec<Vertex> = Vec::with_capacity(nb_indices);

    let mut v = |i: usize| new_verts.push(verts[i].clone());

    for lon in 0..nb_long {
        v(lon + 2);
        v(lon + 1);
        v(0);
    }

    for lat in 0..(nb_lat - 1) {
        for lon in 0..nb_long {
            let current = lon + lat * (nb_long + 1) + 1;
            let next = current + nb_long + 1;

            v(current);
            v(current + 1);
            v(next + 1);

            v(current);
            v(next + 1);
            v(next);
        }
    }

    for lon in 0..nb_long {
        v(len - 1);
        v(len - (lon + 2) - 1);
        v(len - (lon + 1) - 1);
    }

    for vs in new_verts.chunks_mut(3) {
        if vs.len() < 3 {
            continue;
        }

        let x: *mut _ = &mut vs[0];
        let y: *mut _ = &mut vs[1];
        let z: *mut _ = &mut vs[2];
        unsafe {
            calculate_tangent_and_bitangent(&mut *x, &mut *y, &mut *z);
        }
    }
    new_verts
}


pub fn rgb(r: u8, g: u8, b: u8) -> V3 {
    v3( r as f32 / 255.0, 
        g as f32 / 255.0, 
        b as f32 / 255.0
    )
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
    
    let r: f32 = 
        if (h % 1.0) < 0.5 {
            (clamp(-6.0 * (h % 1.0)  + 2.0, 0.0, 1.0) * s + 1.0 - s)  * v
        } 
        else {
            (clamp(6.0 * (h % 1.0)  - 4.0, 0.0, 1.0) * s + 1.0 - s)  * v
    };

    let g: f32 = 
        if (h % 1.0) < 1.0 / 3.0 {
            (clamp(6.0 * (h % 1.0), 0.0, 1.0) * s + 1.0 - s)  * v
        } 
        else {
            (clamp(-6.0 * (h % 1.0) + 4.0, 0.0, 1.0) * s + 1.0 - s)  * v
    };

    let b: f32 = 
        if (h % 1.0) < 2.0 / 0.3 {
            (clamp(6.0 * (h % 1.0)  - 2.0, 0.0, 1.0) * s + 1.0 - s)  * v
        } 
        else {
            (clamp(-6.0 * (h % 1.0)  + 6.0, 0.0, 1.0) * s + 1.0 - s)  * v
    };

    v3(r, g, b)
 }

 pub fn hex(v: u32) -> V3 {
    rgb((v >> 16 & 0xff) as u8, (v >> 8 & 0xff) as u8, (v & 0xff) as u8)
}