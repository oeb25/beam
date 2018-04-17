use cgmath::{self, InnerSpace, Rad};
use genmesh;
use gl;
use image;
use obj;

use std::{self, collections::HashMap, mem, path::Path, rc::Rc};

use mg::*;

pub type V2 = cgmath::Vector2<f32>;
pub type V3 = cgmath::Vector3<f32>;
pub type V4 = cgmath::Vector4<f32>;
pub type P3 = cgmath::Point3<f32>;
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

pub fn v3(x: f32, y: f32, z: f32) -> V3 {
    V3::new(x, y, z)
}

pub trait Vertexable {
    // (pos, norm, tex, trangent, bitangent)
    fn sizes() -> (usize, usize, usize, usize);
    fn offsets() -> (usize, usize, usize, usize);
}

pub struct Vertex {
    pub pos: V3,
    pub norm: V3,
    pub tex: V2,
    pub tangent: V3,
}
impl Vertex {
    pub fn soa(vs: &[Vertex]) -> (Vec<V3>, Vec<V3>, Vec<V2>, Vec<V3>) {
        let mut pos = vec![];
        let mut norm = vec![];
        let mut tex = vec![];
        let mut tangent = vec![];
        for v in vs.into_iter() {
            pos.push(v.pos);
            norm.push(v.norm);
            tex.push(v.tex);
            tangent.push(v.tangent);
        }
        (pos, norm, tex, tangent)
    }
}

impl Vertexable for Vertex {
    fn sizes() -> (usize, usize, usize, usize) {
        let float_size = mem::size_of::<f32>();
        (
            size_of!(Vertex, pos) / float_size,
            size_of!(Vertex, norm) / float_size,
            size_of!(Vertex, tex) / float_size,
            size_of!(Vertex, tangent) / float_size,
        )
    }
    fn offsets() -> (usize, usize, usize, usize) {
        (
            offset_of!(Vertex, pos),
            offset_of!(Vertex, norm),
            offset_of!(Vertex, tex),
            offset_of!(Vertex, tangent),
        )
    }
}
#[allow(unused)]
#[derive(Debug, Clone, Copy)]
pub enum ImageKind {
    Diffuse,
    Ambient,
    Specular,
    Reflection,
    CubeMap,
    NormalMap,
}

#[derive(Debug)]
pub struct Image {
    pub texture: Texture,
    pub path: String,
    pub kind: ImageKind,
}
impl Image {
    pub fn new_from_disk(path: &str, kind: ImageKind) -> Image {
        let img = image::open(path).expect(&format!("unable to read {}", path));
        let tex_kind = match kind {
            ImageKind::Diffuse
            | ImageKind::Ambient
            | ImageKind::Specular
            | ImageKind::NormalMap
            | ImageKind::Reflection => TextureKind::Texture2d,
            ImageKind::CubeMap => unimplemented!(),
        };
        let texture = Texture::new(tex_kind);
        texture
            .bind()
            .parameter_int(TextureParameter::WrapS, gl::REPEAT as i32)
            .parameter_int(TextureParameter::WrapT, gl::REPEAT as i32)
            .parameter_int(TextureParameter::MinFilter, gl::LINEAR as i32)
            .parameter_int(TextureParameter::MagFilter, gl::LINEAR as i32)
            .load_image(
                TextureTarget::Texture2d,
                TextureInternalFormat::Srgb,
                TextureFormat::Rgb,
                &img,
            );

        Image {
            texture,
            path: path.to_string(),
            kind,
        }
    }
}

#[allow(unused)]
pub struct Mesh {
    positions: Vec<V3>,
    normals: Vec<V3>,
    tex: Vec<V2>,
    tangents: Vec<V3>,

    indecies: Option<Vec<usize>>,
    textures: Vec<Rc<Image>>,

    vao: VertexArray,
    vbo: VertexBuffer<()>,
    ebo: Option<Ebo>,
}

impl Mesh {
    pub fn new(
        positions: Vec<V3>,
        normals: Vec<V3>,
        tex: Vec<V2>,
        tangents: Vec<V3>,

        indecies: Option<Vec<usize>>,
        textures: Vec<Rc<Image>>,
    ) -> Mesh {
        let positions_size = positions.len() * mem::size_of::<V3>();
        let normals_size = normals.len() * mem::size_of::<V3>();
        let tex_size = tex.len() * mem::size_of::<V2>();
        let tangents_size = tangents.len() * mem::size_of::<V3>();

        let positions_offset = 0;
        let normals_offset = positions_offset + positions_size;
        let tex_offset = normals_offset + normals_size;
        let tangents_offset = tex_offset + tex_size;

        let mut vao = VertexArray::new();
        let mut vbo =
            VertexBuffer::from_size(positions_size + normals_size + tex_size + tangents_size);

        let ebo = if indecies.is_some() {
            Some(Ebo::new())
        } else {
            None
        };
        {
            // if let Some(ref indecies) = &indecies {
            //     if let Some(ref mut ebo) = &mut ebo {
            //         let ebo_binder = vao_binder.bind_ebo(ebo);
            //         ebo_binder.buffer_data(&indecies);
            //     }
            // }

            let mut vbo_binder = vbo.bind();

            vbo_binder
                .buffer_sub_data(positions_offset, &positions)
                .buffer_sub_data(normals_offset, &normals)
                .buffer_sub_data(tex_offset, &tex)
                .buffer_sub_data(tangents_offset, &tangents);

            let float_size = mem::size_of::<f32>();

            vao.bind()
                .attrib(0, 3, 3 * float_size, positions_offset)
                .attrib(1, 3, 3 * float_size, normals_offset)
                .attrib(2, 2, 2 * float_size, tex_offset)
                .attrib(3, 3, 3 * float_size, tangents_offset);
        }

        Mesh {
            positions,
            normals,
            tex,
            tangents,

            indecies,
            textures,
            vao,
            vbo,
            ebo,
        }
    }

    fn bind(&mut self) -> MeshBinding {
        MeshBinding(self)
    }
}

struct MeshBinding<'a>(&'a mut Mesh);
impl<'a> MeshBinding<'a> {
    fn bind_textures(&self, program: &ProgramBinding) {
        let mut diffuse_n = 0;
        let mut ambient_n = 0;
        let mut specular_n = 0;
        let mut reflection_n = 0;
        let mut normal_n = 0;
        for (i, tex) in self.0.textures.iter().enumerate() {
            let (name, number) = match tex.kind {
                ImageKind::Diffuse => {
                    diffuse_n += 1;
                    ("diffuse", diffuse_n)
                }
                ImageKind::Ambient => {
                    ambient_n += 1;
                    ("ambient", ambient_n)
                }
                ImageKind::Specular => {
                    specular_n += 1;
                    ("specular", specular_n)
                }
                ImageKind::Reflection => {
                    reflection_n += 1;
                    ("reflection", reflection_n)
                }
                ImageKind::NormalMap => {
                    normal_n += 1;
                    ("normal", normal_n)
                }
                ImageKind::CubeMap => unimplemented!(),
            };

            assert_eq!(number, 1);

            program.bind_texture(&format!("tex_{}{}", name, number), &tex.texture, i.into());
        }
        program.bind_bool("useNormalMap", normal_n > 0);
    }
    fn draw<T>(&mut self, fbo: &T, program: &ProgramBinding)
    where
        T: FramebufferBinderDrawer,
    {
        self.bind_textures(program);

        self.0
            .vao
            .bind()
            .draw_arrays(fbo, DrawMode::Triangles, 0, self.0.positions.len());
    }
    fn draw_instanced<T>(&mut self, fbo: &T, program: &ProgramBinding, transforms: &VboBinder<Mat4>)
    where
        T: FramebufferBinderDrawer,
    {
        self.bind_textures(program);

        let mut vao = self.0.vao.bind();
        let offset = 4;
        let width = 4;
        for i in 0..width {
            let index = i + offset;
            vao.vbo_attrib(&transforms, index, width, width * i * mem::size_of::<f32>())
                .attrib_divisor(index, 1);
        }

        vao.draw_arrays_instanced(
            fbo,
            DrawMode::Triangles,
            0,
            self.0.positions.len(),
            transforms.len(),
        );
    }
}

#[allow(unused)]
pub struct Model {
    meshes: Vec<Mesh>,
    texture_cache: HashMap<String, Rc<Image>>,
}
impl Model {
    pub fn new_from_disk(path: &str) -> Model {
        let path = Path::new(path);
        let mut raw_model = obj::Obj::load(path).unwrap();
        let _ = raw_model.load_mtls().unwrap();
        let obj::Obj {
            position,
            texture,
            normal,
            material_libs,
            objects,
            ..
        } = raw_model;

        println!("{:?}", position.len());
        println!("{:?}", texture.len());
        println!("{:?}", normal.len());
        println!("{:?}", material_libs);
        println!("{:?}", objects.len());

        let mut meshes = vec![];
        let mut texture_cache: HashMap<String, Rc<Image>> = HashMap::new();

        for mut o in objects.into_iter() {
            let mut vertices = vec![];
            let mut materials = vec![];
            macro_rules! add_tex {
                ($name:expr, $tex_kind:expr) => {{
                    let path = path.with_file_name($name);
                    let path_string = path.to_str().unwrap().to_string();
                    let tex: Rc<Image> = if texture_cache.contains_key(&path_string) {
                        texture_cache.get(&path_string).unwrap().clone()
                    } else {
                        let tex = Image::new_from_disk(path.to_str().unwrap(), $tex_kind);
                        let tex = Rc::new(tex);
                        texture_cache.insert(path_string, tex.clone());
                        tex
                    };
                    materials.push(tex);
                }};
            }
            for group in o.groups {
                if let Some(mat) = group.material {
                    println!("{:?}", mat);
                    mat.map_kd
                        .as_ref()
                        .map(|diff| add_tex!(diff, ImageKind::Diffuse));
                    mat.map_ks
                        .as_ref()
                        .map(|spec| add_tex!(spec, ImageKind::Specular));
                    // mat.map_ka.as_ref().map(|ambient| add_tex!(ambient, ImageKind::Ambient));
                    mat.map_ka
                        .as_ref()
                        .map(|refl| add_tex!(refl, ImageKind::Reflection));
                    mat.map_refl
                        .as_ref()
                        .map(|_| unimplemented!("REFLECTION MAP!"));
                    mat.map_bump
                        .as_ref()
                        .map(|bump| add_tex!(bump, ImageKind::NormalMap));
                }
                for ps in group.polys {
                    match ps {
                        genmesh::Polygon::PolyTri(genmesh::Triangle {
                            x: va,
                            y: vb,
                            z: vc,
                        }) => {
                            let res = [va, vb, vc]
                                .into_iter()
                                .map(|v| {
                                    let obj::IndexTuple(vert, tex, norm) = v;
                                    let vert = position[*vert].into();
                                    let norm =
                                        norm.map(|i| normal[i].into()).unwrap_or(v3(0.0, 0.0, 0.0));
                                    let tex =
                                        tex.map(|i| texture[i].into()).unwrap_or(V2::new(0.0, 0.0));
                                    (vert, tex, norm)
                                })
                                .collect::<Vec<_>>();
                            let va = res[0];
                            let vb = res[1];
                            let vc = res[2];
                            let meh = [(va, vb, vc), (vb, va, vc), (vc, va, vb)];
                            for ((vert, tex, norm), (avert, atex, _anorm), (bvert, btex, _bnorm)) in
                                meh.iter()
                            {
                                let pos = *vert;
                                let norm = *norm;
                                let tex = *tex;

                                // Tangets and bitangents

                                let delta_pos1 = avert - pos;
                                let delta_pos2 = bvert - pos;

                                let delta_uv1 = atex - tex;
                                let delta_uv2 = btex - tex;

                                let r =
                                    1.0 / (delta_uv1.x * delta_uv2.y - delta_uv1.y * delta_uv2.x);
                                let tangent =
                                    (delta_pos2 * delta_uv1.x - delta_pos1 * delta_uv2.x) * r;

                                let v = Vertex {
                                    pos,
                                    norm,
                                    tex,
                                    tangent,
                                };
                                vertices.push(v);
                            }
                        }
                        _ => unimplemented!(),
                    }
                }
            }

            let (a, b, c, d) = Vertex::soa(&vertices);
            let mesh = Mesh::new(a, b, c, d, None, materials);

            meshes.push(mesh);
        }

        Model {
            meshes,
            texture_cache,
        }
    }
    #[allow(unused)]
    fn draw<T>(&mut self, fbo: &T, program: &ProgramBinding)
    where
        T: FramebufferBinderDrawer,
    {
        for mut mesh in self.meshes.iter_mut() {
            mesh.bind().draw(fbo, &program);
        }
    }
    fn draw_instanced<T>(&mut self, fbo: &T, program: &ProgramBinding, offsets: &VboBinder<Mat4>)
    where
        T: FramebufferBinderDrawer,
    {
        for mut mesh in self.meshes.iter_mut() {
            mesh.bind().draw_instanced(fbo, &program, offsets);
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct DirectionalLight {
    pub ambient: V3,
    pub diffuse: V3,
    pub specular: V3,

    pub direction: V3,

    pub shadow_map: ShadowMap,
}
impl DirectionalLight {
    fn space(&self) -> Mat4 {
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
        let o = origo;
        let view = Mat4::look_at(o, o - self.direction, v3(0.0, 1.0, 0.0));
        projection * view
    }
    fn bind(&self, name: &str, texture_slot: TextureSlot, program: &ProgramBinding) {
        let ext = |e| format!("{}.{}", name, e);
        let space = self.space();
        let DirectionalLight {
            ambient,
            diffuse,
            specular,
            direction,
            shadow_map,
        } = self;
        program.bind_vec3(&ext("ambient"), *ambient);
        program.bind_vec3(&ext("diffuse"), *diffuse);
        program.bind_vec3(&ext("specular"), *specular);

        program.bind_vec3(&ext("direction"), *direction);

        program.bind_mat4(&ext("space"), space);

        program.bind_texture(&ext("shadowMap"), &shadow_map.map, texture_slot);
    }
    fn bind_multiple(
        lights: &[&mut DirectionalLight],
        initial_slot: TextureSlot,
        name_uniform: &str,
        amt_uniform: &str,
        program: &ProgramBinding,
    ) {
        program.bind_int(amt_uniform, lights.len() as i32);
        for (i, light) in lights.iter().enumerate() {
            let slot: i32 = initial_slot.into();
            let slot = (slot as usize + i).into();
            light.bind(&format!("{}[{}]", name_uniform, i), slot, program);
        }
    }
    fn bind_shadow_map(&mut self) -> (FramebufferBinderReadDraw, Mat4) {
        let light_space = self.space();
        (self.shadow_map.fbo.bind(), light_space)
    }
}
#[repr(C)]
#[derive(Debug)]
pub struct PointLight {
    pub ambient: V3,
    pub diffuse: V3,
    pub specular: V3,

    pub position: V3,
    pub last_shadow_map_position: V3,

    pub constant: f32,
    pub linear: f32,
    pub quadratic: f32,

    pub shadow_map: PointShadowMap,
}
impl PointLight {
    fn bind(&self, name: &str, texture_slot: TextureSlot, program: &ProgramBinding) {
        let ext = |e| format!("{}.{}", name, e);
        let PointLight {
            position,
            ambient,
            diffuse,
            specular,
            constant,
            linear,
            quadratic,
            shadow_map,
            last_shadow_map_position,
        } = self;
        program.bind_vec3(&ext("ambient"), *ambient);
        program.bind_vec3(&ext("diffuse"), *diffuse);
        program.bind_vec3(&ext("specular"), *specular);

        program.bind_vec3(&ext("position"), *position);
        program.bind_vec3(&ext("lastPosition"), *last_shadow_map_position);

        program.bind_float(&ext("constant"), *constant);
        program.bind_float(&ext("linear"), *linear);
        program.bind_float(&ext("quadratic"), *quadratic);

        program.bind_texture(&ext("shadowMap"), &shadow_map.map, texture_slot);
        program.bind_float(&ext("farPlane"), shadow_map.far);
    }
    fn bind_multiple(
        lights: &[PointLight],
        initial_slot: TextureSlot,
        name_uniform: &str,
        amt_uniform: &str,
        program: &ProgramBinding,
    ) {
        program.bind_int(amt_uniform, lights.len() as i32);
        for (i, light) in lights.iter().enumerate() {
            let slot: i32 = initial_slot.into();
            let slot = (slot as usize + i).into();
            light.bind(&format!("{}[{}]", name_uniform, i), slot, program);
        }
    }
    fn bind_shadow_map(&mut self) -> (FramebufferBinderReadDraw, [Mat4; 6]) {
        let light_space: Mat4 = cgmath::PerspectiveFov {
            fovy: Rad(std::f32::consts::PI / 2.0),
            aspect: (self.shadow_map.width as f32) / (self.shadow_map.height as f32),
            near: self.shadow_map.near,
            far: self.shadow_map.far,
        }.into();

        let origo = P3::new(0.0, 0.0, 0.0);
        let lp = origo + self.last_shadow_map_position;
        let look_at = |p, up| light_space * Mat4::look_at(lp, lp + p, up);

        let shadow_transforms = [
            look_at(v3(1.0, 0.0, 0.0), v3(0.0, -1.0, 0.0)),
            look_at(v3(-1.0, 0.0, 0.0), v3(0.0, -1.0, 0.0)),
            look_at(v3(0.0, 1.0, 0.0), v3(0.0, 0.0, 1.0)),
            look_at(v3(0.0, -1.0, 0.0), v3(0.0, 0.0, -1.0)),
            look_at(v3(0.0, 0.0, 1.0), v3(0.0, -1.0, 0.0)),
            look_at(v3(0.0, 0.0, -1.0), v3(0.0, -1.0, 0.0)),
        ];

        (self.shadow_map.fbo.bind(), shadow_transforms)
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

#[allow(unused)]
#[derive(Debug)]
pub struct ShadowMap {
    width: u32,
    height: u32,
    fbo: Framebuffer,
    map: Texture,
}

#[allow(unused)]
impl ShadowMap {
    pub fn new() -> ShadowMap {
        let (width, height) = (1024, 1024);
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
            .parameter_int(TextureParameter::MinFilter, gl::NEAREST as i32)
            .parameter_int(TextureParameter::MagFilter, gl::NEAREST as i32)
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
}

#[derive(Debug)]
pub struct PointShadowMap {
    width: u32,
    height: u32,
    near: f32,
    far: f32,
    fbo: Framebuffer,
    map: Texture,
}

impl PointShadowMap {
    pub fn new() -> PointShadowMap {
        let (width, height) = (1024, 1024);
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
                    GlType::Float,
                );
            }

            tex.parameter_int(TextureParameter::MinFilter, gl::NEAREST as i32)
                .parameter_int(TextureParameter::MagFilter, gl::NEAREST as i32)
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

        let near = 1.0;
        let far = 25.0;

        PointShadowMap {
            width,
            height,
            near,
            far,
            fbo,
            map,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ObjectKind {
    Cube,
    Nanosuit,
    Cyborg,
}

#[derive(Debug, Clone)]
pub struct Object {
    pub kind: ObjectKind,
    pub transform: Mat4,
}

pub struct GRenderPass {
    fbo: Framebuffer,
    #[allow(unused)]
    depth: Renderbuffer,
    position: Texture,
    normal: Texture,
    albedo_spec: Texture,
}
impl GRenderPass {
    pub fn new(w: u32, h: u32) -> GRenderPass {
        let mut fbo = Framebuffer::new();
        let mut depth = Renderbuffer::new();
        depth
            .bind()
            .storage(TextureInternalFormat::DepthComponent, w, h);

        let (position, normal, albedo_spec) = {
            let buffer = fbo.bind();

            let position = Texture::new(TextureKind::Texture2d);
            position
                .bind()
                .empty(
                    TextureTarget::Texture2d,
                    0,
                    TextureInternalFormat::Rgb16f,
                    w,
                    h,
                    TextureFormat::Rgb,
                    GlType::Float,
                )
                .parameter_int(TextureParameter::MinFilter, gl::NEAREST as i32)
                .parameter_int(TextureParameter::MagFilter, gl::NEAREST as i32);
            buffer.texture_2d(Attachment::Color0, TextureTarget::Texture2d, &position, 0);

            let normal = Texture::new(TextureKind::Texture2d);
            normal
                .bind()
                .empty(
                    TextureTarget::Texture2d,
                    0,
                    TextureInternalFormat::Rgb16f,
                    w,
                    h,
                    TextureFormat::Rgb,
                    GlType::Float,
                )
                .parameter_int(TextureParameter::MinFilter, gl::NEAREST as i32)
                .parameter_int(TextureParameter::MagFilter, gl::NEAREST as i32);
            buffer.texture_2d(Attachment::Color1, TextureTarget::Texture2d, &normal, 0);

            let albedo_spec = Texture::new(TextureKind::Texture2d);
            albedo_spec
                .bind()
                .empty(
                    TextureTarget::Texture2d,
                    0,
                    TextureInternalFormat::Rgba,
                    w,
                    h,
                    TextureFormat::Rgba,
                    GlType::UnsignedByte,
                )
                .parameter_int(TextureParameter::MinFilter, gl::NEAREST as i32)
                .parameter_int(TextureParameter::MagFilter, gl::NEAREST as i32);
            buffer.texture_2d(
                Attachment::Color2,
                TextureTarget::Texture2d,
                &albedo_spec,
                0,
            );

            buffer
                .draw_buffers(&[Attachment::Color0, Attachment::Color1, Attachment::Color2])
                .renderbuffer(Attachment::Depth, &depth);

            (position, normal, albedo_spec)
        };

        GRenderPass {
            fbo,
            depth,
            position,
            normal,
            albedo_spec,
        }
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
    fn build(self) -> Image {
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

        Image {
            texture,
            path: sum_path,
            kind: ImageKind::CubeMap,
        }
    }
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
            near: 0.01,
            far: 100.0,
        }.into()
    }
}

pub struct RenderProps<'a> {
    pub objects: &'a [Object],
    pub camera: &'a Camera,
    pub time: f32,

    pub directional_lights: &'a mut [&'a mut DirectionalLight],
    pub point_lights: &'a mut [PointLight],
}

pub struct Pipeline {
    pub vertex_program: Program,
    pub directional_shadow_program: Program,
    pub point_shadow_program: Program,
    pub directional_lighting_program: Program,
    pub point_lighting_program: Program,
    pub skybox_program: Program,
    pub hdr_program: Program,

    pub nanosuit: Model,
    pub cyborg: Model,
    pub cube: Mesh,
    pub rect: Mesh,

    pub screen_width: u32,
    pub screen_height: u32,

    pub g: GRenderPass,
    pub color_a_fbo: Framebuffer,
    pub color_a_tex: Texture,
    pub color_b_fbo: Framebuffer,
    pub color_b_tex: Texture,

    pub window_fbo: Framebuffer,

    pub skybox: Texture,
}

impl Pipeline {
    pub fn new(w: u32, h: u32) -> Pipeline {
        unsafe {
            gl::Enable(gl::DEPTH_TEST);
            gl::Enable(gl::CULL_FACE);
            gl::Enable(gl::FRAMEBUFFER_SRGB);
        }

        let vertex_program =
            Program::new_from_disk("shaders/shader.vs", None, "shaders/shader.fs").unwrap();
        let skybox_program =
            Program::new_from_disk("shaders/skybox.vs", None, "shaders/skybox.fs").unwrap();
        let hdr_program =
            Program::new_from_disk("shaders/hdr.vs", None, "shaders/hdr.fs").unwrap();
        let directional_shadow_program =
            Program::new_from_disk("shaders/shadow.vs", None, "shaders/shadow.fs").unwrap();
        let point_shadow_program = Program::new_from_disk(
            "shaders/point_shadow.vs",
            Some("shaders/point_shadow.gs"),
            "shaders/point_shadow.fs",
        ).unwrap();
        let directional_lighting_program = Program::new_from_disk(
            "shaders/lighting.vs",
            None,
            "shaders/directional_lighting.fs",
        ).unwrap();
        let point_lighting_program =
            Program::new_from_disk("shaders/lighting.vs", None, "shaders/point_lighting.fs")
                .unwrap();

        let skybox = CubeMapBuilder {
            back: "assets/skybox/back.jpg",
            front: "assets/skybox/front.jpg",
            right: "assets/skybox/right.jpg",
            bottom: "assets/skybox/bottom.jpg",
            left: "assets/skybox/left.jpg",
            top: "assets/skybox/top.jpg",
        }.build();
        let nanosuit = Model::new_from_disk("assets/nanosuit_reflection/nanosuit.obj");
        let cyborg = Model::new_from_disk("assets/cyborg/cyborg.obj");

        let tex1 = Image::new_from_disk("assets/container2.png", ImageKind::Diffuse);
        let tex2 = Image::new_from_disk("assets/container2_specular.png", ImageKind::Specular);
        let (tex1, tex2) = (Rc::new(tex1), Rc::new(tex2));

        let (a, b, c, d) = Vertex::soa(&cube_vertices());
        let cube_mesh = Mesh::new(a, b, c, d, None, vec![tex1, tex2]);
        let (a, b, c, d) = Vertex::soa(&rect_vertices());
        let rect_mesh = Mesh::new(a, b, c, d, None, vec![]);

        let window_fbo = unsafe { Framebuffer::window() };

        let g = GRenderPass::new(w, h);

        let mut color_a_fbo = Framebuffer::new();
        let mut color_a_depth = Renderbuffer::new();
        color_a_depth
            .bind()
            .storage(TextureInternalFormat::DepthComponent, w, h);
        let color_a_tex = Texture::new(TextureKind::Texture2d);
        color_a_tex
            .bind()
            .empty(
                TextureTarget::Texture2d,
                0,
                TextureInternalFormat::Rgba16f,
                w,
                h,
                TextureFormat::Rgba,
                GlType::Float,
            )
            .parameter_int(TextureParameter::MinFilter, gl::LINEAR as i32)
            .parameter_int(TextureParameter::MagFilter, gl::LINEAR as i32);
        color_a_fbo
            .bind()
            .texture_2d(
                Attachment::Color0,
                TextureTarget::Texture2d,
                &color_a_tex,
                0,
            )
            .renderbuffer(Attachment::Depth, &color_a_depth)
            .check_status()
            .expect("framebuffer not complete");

        let mut color_b_fbo = Framebuffer::new();
        let mut color_b_depth = Renderbuffer::new();
        color_b_depth
            .bind()
            .storage(TextureInternalFormat::DepthComponent, w, h);
        let color_b_tex = Texture::new(TextureKind::Texture2d);
        color_b_tex
            .bind()
            .empty(
                TextureTarget::Texture2d,
                0,
                TextureInternalFormat::Rgba16f,
                w,
                h,
                TextureFormat::Rgba,
                GlType::Float,
            )
            .parameter_int(TextureParameter::MinFilter, gl::LINEAR as i32)
            .parameter_int(TextureParameter::MagFilter, gl::LINEAR as i32);
        color_b_fbo
            .bind()
            .texture_2d(
                Attachment::Color0,
                TextureTarget::Texture2d,
                &color_b_tex,
                0,
            )
            .renderbuffer(Attachment::Depth, &color_a_depth)
            .check_status()
            .expect("framebuffer not complete");

        Pipeline {
            vertex_program,
            directional_shadow_program,
            point_shadow_program,
            directional_lighting_program,
            point_lighting_program,
            skybox_program,
            hdr_program,

            nanosuit,
            cyborg,
            cube: cube_mesh,
            rect: rect_mesh,

            screen_width: w,
            screen_height: h,

            g,
            color_a_fbo,
            color_a_tex,
            color_b_fbo,
            color_b_tex,

            window_fbo,

            skybox: skybox.texture,
        }
    }
    pub fn render(&mut self, update_shadows: bool, props: RenderProps) {
        let view = props.camera.get_view();
        let view_pos = props.camera.pos;
        let projection = props.camera.get_projection();

        let mut nanosuit_transforms = vec![];
        let mut cyborg_transforms = vec![];
        let mut cube_transforms = vec![];

        for obj in props.objects.iter() {
            match obj.kind {
                ObjectKind::Cube => cube_transforms.push(obj.transform),
                ObjectKind::Nanosuit => nanosuit_transforms.push(obj.transform),
                ObjectKind::Cyborg => cyborg_transforms.push(obj.transform),
            }
        }
        let mut nanosuit_vbo = VertexBuffer::from_data(&nanosuit_transforms);
        let mut cyborg_vbo = VertexBuffer::from_data(&cyborg_transforms);
        let mut cube_vbo = VertexBuffer::from_data(&cube_transforms);

        let pi = std::f32::consts::PI;

        macro_rules! render_scene {
            ($fbo:expr, $program:expr) => {{
                {
                    let model = Mat4::from_scale(1.0 / 4.0) * Mat4::from_translation(v3(0.0, 0.0, 4.0));
                    $program.bind_mat4("model", model);
                    self.nanosuit
                        .draw_instanced(&$fbo, &$program, &nanosuit_vbo.bind());
                }
                {
                    let model = Mat4::from_angle_y(Rad(pi));
                    $program.bind_mat4("model", model);
                    self.cyborg
                        .draw_instanced(&$fbo, &$program, &cyborg_vbo.bind());
                }
                {
                    let model = Mat4::from_scale(1.0);
                    $program.bind_mat4("model", model);
                    self.cube
                        .bind()
                        .draw_instanced(&$fbo, &$program, &cube_vbo.bind());
                }
            }};
        };

        {
            // Render geometry
            let fbo = self.g.fbo.bind();
            fbo.clear(ClearMask::ColorDepth);
            let program = self.vertex_program.bind();
            program
                .bind_mat4("projection", projection)
                .bind_mat4("view", view)
                .bind_vec3("viewPos", view_pos)
                .bind_float("time", props.time)
                .bind_texture("skybox", &self.skybox, TextureSlot::Six);

            render_scene!(fbo, program);
        }

        // Clear backbuffer
        self.color_b_fbo.bind().clear(ClearMask::ColorDepth);
        if update_shadows {
            {
                // Render depth map for directional lights
                let p = self.directional_shadow_program.bind();
                unsafe {
                    gl::Viewport(
                        0,
                        0,
                        props.directional_lights[0].shadow_map.width as i32,
                        props.directional_lights[0].shadow_map.height as i32,
                    );
                    gl::CullFace(gl::FRONT);
                }
                for light in props.directional_lights.iter_mut() {
                    let (fbo, light_space) = light.bind_shadow_map();
                    fbo.clear(ClearMask::Depth);
                    p.bind_mat4("lightSpace", light_space);
                    render_scene!(fbo, p);
                }
            }
            {
                // Render depth map for point lights
                unsafe {
                    gl::Viewport(
                        0,
                        0,
                        props.point_lights[0].shadow_map.width as i32,
                        props.point_lights[0].shadow_map.height as i32,
                    );
                    gl::CullFace(gl::FRONT);
                }
                let p = self.point_shadow_program.bind();
                for light in props.point_lights.iter_mut() {
                    let far = light.shadow_map.far;
                    let position = light.position;
                    light.last_shadow_map_position = position;
                    let (fbo, light_spaces) = light.bind_shadow_map();
                    fbo.clear(ClearMask::Depth);
                    p.bind_mat4s("shadowMatrices", &light_spaces)
                        .bind_vec3("lightPos", position)
                        .bind_float("farPlane", far);
                    render_scene!(fbo, p);
                }
                unsafe {
                    gl::CullFace(gl::BACK);
                    gl::Viewport(
                        0,
                        0,
                        self.screen_width as i32,
                        self.screen_height as i32,
                    );
                }
            }
        }

        {
            // Render lighting

            {
                let fbo = self.color_a_fbo.bind();
                fbo.clear(ClearMask::ColorDepth);

                let g = self.directional_lighting_program.bind();
                g.bind_texture("aPosition", &self.g.position, TextureSlot::Zero)
                    .bind_texture("aNormal", &self.g.normal, TextureSlot::One)
                    .bind_texture("aAlbedoSpec", &self.g.albedo_spec, TextureSlot::Two)
                    .bind_texture("skybox", &self.skybox, TextureSlot::Three)
                    .bind_vec3("viewPos", view_pos);

                let lights: &[_] = &props.directional_lights;

                DirectionalLight::bind_multiple(
                    lights,
                    TextureSlot::Four,
                    "lights",
                    "nrLights",
                    &g,
                );

                let mut c = VertexBuffer::from_data(&vec![Mat4::from_scale(1.0)]);
                self.rect.bind().draw_instanced(&fbo, &g, &c.bind());
            }

            {
                let fbo = self.color_b_fbo.bind();
                fbo.clear(ClearMask::ColorDepth);

                let g = self.point_lighting_program.bind();
                g.bind_texture("aPosition", &self.g.position, TextureSlot::Zero)
                    .bind_texture("aNormal", &self.g.normal, TextureSlot::One)
                    .bind_texture("aAlbedoSpec", &self.g.albedo_spec, TextureSlot::Two)
                    .bind_texture("skybox", &self.skybox, TextureSlot::Three)
                    .bind_texture("accumulator", &self.color_a_tex, TextureSlot::Four)
                    .bind_vec3("viewPos", view_pos);

                PointLight::bind_multiple(
                    props.point_lights,
                    TextureSlot::Five,
                    "lights",
                    "nrLights",
                    &g,
                );

                let mut c = VertexBuffer::from_data(&vec![Mat4::from_scale(1.0)]);
                self.rect.bind().draw_instanced(&fbo, &g, &c.bind());
            }
        }

        let color_fbo = &mut self.color_b_fbo;
        let color_tex = &self.color_b_tex;

        // Skybox Pass
        {
            // Copy z-buffer over from geometry pass
            let g_binding = self.g.fbo.read();
            let hdr_binding = color_fbo.draw();
            let (w, h) = (self.screen_width as i32, self.screen_height as i32);
            hdr_binding.blit_framebuffer(
                &g_binding,
                (0, 0, w, h),
                (0, 0, w, h),
                gl::DEPTH_BUFFER_BIT,
                gl::NEAREST,
            );
        }
        {
            // Render skybox
            let fbo = color_fbo.bind();
            {
                unsafe {
                    gl::DepthFunc(gl::LEQUAL);
                    gl::Disable(gl::CULL_FACE);
                }
                let program = self.skybox_program.bind();
                let mut view = view.clone();
                view.w = V4::new(0.0, 0.0, 0.0, 0.0);
                program
                    .bind_mat4("projection", projection)
                    .bind_mat4("view", view)
                    .bind_texture("skybox", &self.skybox, TextureSlot::Ten);
                self.cube.bind().draw(&fbo, &program);
                unsafe {
                    gl::DepthFunc(gl::LESS);
                    gl::Enable(gl::CULL_FACE);
                }
            }
        }

        // HDR/Screen Pass
        {
            let fbo = self.window_fbo.bind();
            fbo.clear(ClearMask::ColorDepth);

            let hdr = self.hdr_program.bind();
            hdr.bind_texture("hdrBuffer", &color_tex, TextureSlot::Zero)
                .bind_float("time", props.time);
            let mut c = VertexBuffer::from_data(&vec![Mat4::from_scale(1.0)]);
            self.rect.bind().draw_instanced(&fbo, &hdr, &c.bind());
        }
    }
}

macro_rules! v {
    ($pos:expr, $norm:expr, $tex:expr, $tangent:expr) => {{
        let tangent = $tangent.into();
        Vertex {
            pos: $pos.into(),
            tex: $tex.into(),
            norm: $norm.into(),
            tangent: tangent,
        }
    }};
}

fn rect_vertices() -> Vec<Vertex> {
    vec![
        v!(
            [-1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [-1.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [-1.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [1.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0],
            [0.0, 1.0, 0.0]
        ),
    ]
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
