use cgmath::{self, InnerSpace, Rad};
use genmesh;
use gl;
use image;
use obj;

use std::{self, collections::HashMap, mem, path::Path, rc::{Rc, Weak}};

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
    pub bitangent: V3,
}
impl Vertex {
    pub fn soa(vs: &[Vertex]) -> (Vec<V3>, Vec<V3>, Vec<V2>, Vec<V3>, Vec<V3>) {
        let mut pos = vec![];
        let mut norm = vec![];
        let mut tex = vec![];
        let mut tangent = vec![];
        let mut bitangent = vec![];
        for v in vs.into_iter() {
            pos.push(v.pos);
            norm.push(v.norm);
            tex.push(v.tex);
            tangent.push(v.tangent);
            bitangent.push(v.bitangent);
        }
        (pos, norm, tex, tangent, bitangent)
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
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum ImageKind {
    Diffuse,
    Ambient,
    Specular,
    Reflection,
    Emissive,
    CubeMap,
    NormalMap,
}

#[derive(Debug)]
pub struct TextureCache {
    pub cache: HashMap<String, Weak<Texture>>,
}

impl TextureCache {
    pub fn new() -> TextureCache {
        let cache = HashMap::new();
        TextureCache { cache }
    }
    pub fn load(&mut self, path: impl AsRef<Path>) -> Rc<Texture> {
        let path: &Path = path.as_ref();
        let path_str = path.to_str().unwrap();

        self.cache.get(path_str).and_then(|c| c.upgrade()).unwrap_or_else(|| {
            let img = image::open(&path).expect(&format!("unable to read {:?}", path));
            let texture = Texture::new(TextureKind::Texture2d);
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
            let tex = Rc::new(texture);
            self.cache.insert(path_str.to_owned(), Rc::downgrade(&tex));
            tex
        })

    }
}

#[derive(Debug, Clone)]
pub struct Image {
    pub texture: Rc<Texture>,
    pub path: Option<String>,
    pub kind: ImageKind,
}
impl Image {
    pub fn new_from_disk<'a>(texture_cache: &mut TextureCache, path: impl AsRef<Path>, kind: ImageKind) -> Image {
        // let tex_kind = match kind {
        //     ImageKind::Diffuse
        //     | ImageKind::Ambient
        //     | ImageKind::Specular
        //     | ImageKind::NormalMap
        //     | ImageKind::Reflection
        //     | ImageKind::Emissive => TextureKind::Texture2d,
        //     ImageKind::CubeMap => unimplemented!(),
        // };
        let tex = texture_cache.load(&path);

        Image::new(tex, Some(path.as_ref().to_str().unwrap().to_owned()), kind)
    }
    pub fn new(texture: Rc<Texture>, path: Option<String>, kind: ImageKind) -> Image {
        Image { texture, path, kind }
    }
}

#[allow(unused)]
pub struct Mesh<T> {
    vcount: usize,

    indecies: Option<Vec<usize>>,
    textures: Vec<Image>,

    vao: VertexArray,
    vbo: VertexBuffer<T>,
    ebo: Option<ElementBuffer<u32>>,
}

impl Mesh<Vertex> {
    pub fn new(
        vertices: &[Vertex],
        textures: Vec<Image>,
    ) -> Mesh<Vertex> {
        let mut vao = VertexArray::new();
        let mut vbo = VertexBuffer::from_data(vertices);

        {
            let float_size = mem::size_of::<f32>();
            let vao_binder = vao.bind();
            let vbo_binder = vbo.bind();

            macro_rules! x {
                ($i:expr, $e:ident) => (vao_binder.vbo_attrib(&vbo_binder, $i, size_of!(Vertex, $e) / float_size, offset_of!(Vertex, $e)))
            }

            x!(0, pos);
            x!(1, norm);
            x!(2, tex);
            x!(3, tangent);
            x!(4, bitangent);
        }

        Mesh {
            vcount: vertices.len(),
            indecies: None,
            textures: textures,
            vao: vao,
            vbo: vbo,
            ebo: None,
        }
    }
}
impl Mesh<()> {
    pub fn new_soa(
        positions: &[V3],
        normals: &[V3],
        tex: &[V2],
        tangents: &[V3],
        bitangents: &[V3],

        indecies: Option<Vec<usize>>,
        textures: Vec<Image>,
    ) -> Mesh<()> {
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
            Some(ElementBuffer::new())
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
                .buffer_sub_data(positions_offset, positions)
                .buffer_sub_data(normals_offset, normals)
                .buffer_sub_data(tex_offset, tex)
                .buffer_sub_data(tangents_offset, tangents);

            vao.bind()
                .attrib(&vbo_binder, 0, 3, GlType::Float, 0, positions_offset)
                .attrib(&vbo_binder, 1, 3, GlType::Float, 0, normals_offset)
                .attrib(&vbo_binder, 2, 2, GlType::Float, 0, tex_offset)
                .attrib(&vbo_binder, 3, 3, GlType::Float, 0, tangents_offset);
        }

        Mesh {
            // positions,
            // normals,
            // tex,
            // tangents,
            vcount: positions.len(),

            indecies,
            textures,
            vao,
            vbo,
            ebo,
        }
    }
}
impl<T> Mesh<T> {
    pub fn bind(&mut self) -> MeshBinding<T> {
        MeshBinding(self)
    }
}

pub struct MeshBinding<'a, T: 'a>(&'a mut Mesh<T>);
impl<'a, T> MeshBinding<'a, T> {
    fn bind_textures(&self, program: &ProgramBinding) {
        let mut diffuse_n = 0;
        let mut ambient_n = 0;
        let mut specular_n = 0;
        let mut reflection_n = 0;
        let mut normal_n = 0;
        let mut emissive_n = 0;
        for tex in self.0.textures.iter() {
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
                ImageKind::Emissive => {
                    emissive_n += 1;
                    ("emissive", emissive_n)
                }
                ImageKind::CubeMap => unimplemented!(),
            };

            assert_eq!(number, 1);

            program.bind_texture(&format!("tex_{}{}", name, number), &tex.texture);
        }
        program.bind_bool("useNormalMap", normal_n > 0);
    }
    pub fn draw<F>(&mut self, fbo: &F, program: &ProgramBinding)
    where
        F: FramebufferBinderDrawer,
    {
        self.bind_textures(program);

        self.0
            .vao
            .bind()
            .draw_arrays(fbo, program, DrawMode::Triangles, 0, self.0.vcount);
    }
    pub fn draw_geometry_instanced<F>(
        &mut self,
        fbo: &F,
        program: &ProgramBinding,
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
        self.bind_textures(program);
        self.draw_geometry_instanced(fbo, program, transforms);
    }
}

#[allow(unused)]
pub struct Model<T> {
    meshes: Vec<Mesh<T>>,
    // texture_cache: HashMap<String, Rc<Image>>,
}
impl<T> Model<T> {
    fn new_vertex_data_from_disk(texture_cache: &mut TextureCache, path: &'static str) -> Vec<(Vec<Vertex>, Vec<Image>)> {
        let path = Path::new(path.clone());
        let mut raw_model = obj::Obj::load(path).unwrap();
        let _ = raw_model.load_mtls().unwrap();
        let obj::Obj {
            position,
            texture,
            normal,
            objects,
            ..
        } = raw_model;

        let vertex_groups = objects.into_iter().map(|o| {
            let mut vertices = vec![];
            let mut materials = vec![];
            macro_rules! add_tex {
                ($name:expr, $tex_kind:expr) => {{
                    let path = path.with_file_name($name);
                    // let path_string = path.to_str().unwrap().to_string();
                    // let img: Image = if texture_cache.contains_key(&path_string) {
                    //     Image::new(texture_cache.get(&path_string).unwrap().clone(), Some(path_string), $tex_kind)
                    // } else {
                    //     let img = Image::new_from_disk(path.to_str().unwrap(), $tex_kind);
                    //     texture_cache.insert(path_string, img.texture.clone());
                    //     img
                    // };
                    let img = Image::new_from_disk(texture_cache, path, $tex_kind);
                    materials.push(img);
                }};
            }
            for group in o.groups {
                if let Some(mat) = group.material {
                    // println!("{:?}", mat);
                    mat.map_kd
                        .as_ref()
                        .map(|diff| add_tex!(diff, ImageKind::Diffuse));
                    mat.map_ks
                        .as_ref()
                        .map(|spec| add_tex!(spec, ImageKind::Specular));
                    // mat.map_ka.as_ref().map(|ambient| add_tex!(ambient, ImageKind::Ambient));
                    mat.map_ka
                        .as_ref()
                        .map(|refl| add_tex!(refl, ImageKind::Emissive));
                    mat.map_refl
                        .as_ref()
                        .map(|_| unimplemented!("REFLECTION MAP!"));
                    mat.ke
                        .as_ref()
                        .map(|_| unimplemented!("EMISSIVE MAP!"));
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

                                let bitangent =
                                    (delta_pos1 * delta_uv2.y - delta_pos2 * delta_uv1.y) * r;

                                let v = Vertex {
                                    pos,
                                    norm,
                                    tex,
                                    tangent,
                                    bitangent,
                                };
                                vertices.push(v);
                            }
                        }
                        x => unimplemented!("{:?}", x),
                    }
                }
            }

            (vertices, materials)
        });

        vertex_groups.collect()
    }
    #[allow(unused)]
    fn draw<F>(&mut self, fbo: &F, program: &ProgramBinding)
    where
        F: FramebufferBinderDrawer,
    {
        for mut mesh in self.meshes.iter_mut() {
            mesh.bind().draw(fbo, program);
        }
    }
    pub fn draw_geometry_instanced<F>(
        &mut self,
        fbo: &F,
        program: &ProgramBinding,
        offsets: &VertexBufferBinder<Mat4>,
    ) where
        F: FramebufferBinderDrawer,
    {
        for mut mesh in self.meshes.iter_mut() {
            mesh.bind().draw_geometry_instanced(fbo, program, offsets);
        }
    }
    pub fn draw_instanced<F>(
        &mut self,
        fbo: &F,
        program: &ProgramBinding,
        offsets: &VertexBufferBinder<Mat4>,
    ) where
        F: FramebufferBinderDrawer,
    {
        let start_slot = program.next_texture_slot();
        for mut mesh in self.meshes.iter_mut() {
            mesh.bind().draw_instanced(fbo, program, offsets);
            program.set_next_texture_slot(start_slot);
        }
    }
}
impl Model<()> {
    pub fn new_from_disk(texture_cache: &mut TextureCache, path: &'static str) -> Model<()> {
        let vertex_groups = Model::<()>::new_vertex_data_from_disk(texture_cache, path);

        let meshes = vertex_groups.into_iter().map(|(vertices, materials)| {
            let (a, b, c, d, e) = Vertex::soa(&vertices);
            let mesh = Mesh::new_soa(&a, &b, &c, &d, &e, None, materials);

            mesh
        }).collect();

        Model {
            meshes,
            // texture_cache,
        }
    }
}
impl Model<Vertex> {
    pub fn new_from_disk(texture_cache: &mut TextureCache, path: &'static str) -> Model<Vertex> {
        let vertex_groups = Model::<()>::new_vertex_data_from_disk(texture_cache, path);

        let meshes = vertex_groups.into_iter().map(|(vertices, materials)| {
            let mesh = Mesh::new(&vertices, materials);

            mesh
        }).collect();

        Model {
            meshes,
            // texture_cache,
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
    fn bind(
        &self,
        camera_pos: V3,
        name: &str,
        texture_slot: TextureSlot,
        program: &ProgramBinding,
    ) {
        let ext = |e| format!("{}.{}", name, e);
        let space = self.space(camera_pos);
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

        program.bind_texture(&ext("shadowMap"), &shadow_map.map);
    }
    pub fn bind_multiple(
        camera_pos: V3,
        lights: &[DirectionalLight],
        initial_slot: TextureSlot,
        name_uniform: &str,
        amt_uniform: &str,
        program: &ProgramBinding,
    ) {
        program.bind_int(amt_uniform, lights.len() as i32);
        for (i, light) in lights.iter().enumerate() {
            let slot: i32 = initial_slot.into();
            let slot = (slot as usize + i).into();
            light.bind(
                camera_pos,
                &format!("{}[{}]", name_uniform, i),
                slot,
                program,
            );
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
    pub ambient: V3,
    pub diffuse: V3,
    pub specular: V3,

    pub position: V3,
    pub last_shadow_map_position: V3,

    pub constant: f32,
    pub linear: f32,
    pub quadratic: f32,

    pub shadow_map: Option<PointShadowMap>,
}
impl PointLight {
    fn bind(&self, name: &str, texture_slot: TextureSlot, program: &ProgramBinding) {
        let ext = |e| {
            let res = format!("{}.{}", name, e);
            // println!("{}", res);
            res
        };
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
        GlError::check().expect("Failed to bind light: light properties");

        program.bind_vec3(&ext("position"), *position);
        program.bind_vec3(&ext("lastPosition"), *last_shadow_map_position);
        GlError::check().expect("Failed to bind light: position");

        program.bind_float(&ext("constant"), *constant);
        program.bind_float(&ext("linear"), *linear);
        program.bind_float(&ext("quadratic"), *quadratic);
        GlError::check().expect("Failed to bind light: attenuation");

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
        initial_slot: TextureSlot,
        name_uniform: &str,
        amt_uniform: &str,
        program: &ProgramBinding,
    ) {
        program.bind_int(amt_uniform, lights.len() as i32);
        GlError::check().expect("Failed to bind number of lights");
        for (i, light) in lights.iter().enumerate() {
            let slot: i32 = initial_slot.into();
            let slot = (slot as usize + i).into();
            // println!("binding: {} into {:?}", format!("{}[{}]", name_uniform, i), slot);
            light.bind(&format!("{}[{}]", name_uniform, i), slot, program);
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
        (1024, 1024)
        // (2048, 2048)
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
    pub fbo: Framebuffer,
    // #[allow(unused)]
    // pub depth: Renderbuffer,
    pub position: Texture,
    pub normal: Texture,
    pub albedo_spec: Texture,
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
                TextureInternalFormat::Rgba16f,
                TextureFormat::Rgba,
                GlType::Float,
                Attachment::Color1,
            );
            let albedo_spec = create_texture(
                TextureInternalFormat::Srgb8Alpha8,
                TextureFormat::Rgba,
                GlType::UnsignedByte,
                Attachment::Color2,
            );

            buffer
                .draw_buffers(&[Attachment::Color0, Attachment::Color1, Attachment::Color2])
                .renderbuffer(Attachment::Depth, &depth);

            (position, normal, albedo_spec)
        };

        GRenderPass {
            fbo,
            // depth,
            position,
            normal,
            albedo_spec,
        }
    }
}

pub struct RenderTarget {
    pub size: (u32, u32),
    pub framebuffer: Framebuffer,
    pub texture: Texture,
}

impl RenderTarget {
    pub fn new(w: u32, h: u32) -> RenderTarget {
        let mut framebuffer = Framebuffer::new();
        let mut depth = Renderbuffer::new();
        let texture = Texture::new(TextureKind::Texture2d);
        texture
            .bind()
            .empty(
                TextureTarget::Texture2d,
                0,
                TextureInternalFormat::Srgb,
                w,
                h,
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
            .storage(TextureInternalFormat::DepthComponent, w, h);

        framebuffer
            .bind()
            .texture_2d(Attachment::Color0, TextureTarget::Texture2d, &texture, 0)
            .draw_buffers(&[Attachment::Color0])
            .renderbuffer(Attachment::Depth, &depth);

        RenderTarget {
            size: (w, h),
            framebuffer,
            texture,
        }
    }

    pub fn bind(&mut self) -> FramebufferBinderReadDraw {
        self.framebuffer.bind()
    }

    pub fn read(&mut self) -> FramebufferBinderRead {
        self.framebuffer.read()
    }

    pub fn draw(&mut self) -> FramebufferBinderDraw {
        self.framebuffer.draw()
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
    pub fn build(self) -> Image {
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

        Image::new(Rc::new(texture), Some(sum_path.into()), ImageKind::CubeMap)
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
            near: 0.1,
            far: 100.0,
        }.into()
    }
}
