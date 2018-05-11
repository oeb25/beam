use cgmath::{self, InnerSpace, Rad};
use genmesh;
use gl;
use image;
use obj;

use std::{self, collections::HashMap, mem, path::Path, rc::Rc};

use mg::*;
use timing::{Report, Timer, Timings};

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
    // positions: Vec<V3>,
    // normals: Vec<V3>,
    // tex: Vec<V2>,
    // tangents: Vec<V3>,
    vcount: usize,

    indecies: Option<Vec<usize>>,
    textures: Vec<Rc<Image>>,

    vao: VertexArray,
    vbo: VertexBuffer<()>,
    ebo: Option<ElementBuffer<u32>>,
}

impl Mesh {
    pub fn new(
        positions: &[V3],
        normals: &[V3],
        tex: &[V2],
        tangents: &[V3],

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

    pub fn bind(&mut self) -> MeshBinding {
        MeshBinding(self)
    }
}

pub struct MeshBinding<'a>(&'a mut Mesh);
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
    pub fn draw<T>(&mut self, fbo: &T, program: &ProgramBinding)
    where
        T: FramebufferBinderDrawer,
    {
        self.bind_textures(program);

        self.0
            .vao
            .bind()
            .draw_arrays(fbo, DrawMode::Triangles, 0, self.0.vcount);
    }
    pub fn draw_instanced<'b, T, S>(
        &mut self,
        timer: &mut S,
        fbo: &T,
        program: &ProgramBinding,
        transforms: &VertexBufferBinder<Mat4>,
    ) where
        T: FramebufferBinderDrawer,
        S: Timer<'b>,
    {
        timer.time("prepare textures");
        self.bind_textures(program);
        GlError::check().expect("Mesh::draw_instanced: failed to bind textures");

        timer.time("prepare instance vao");
        let mut vao = self.0.vao.bind();
        GlError::check().expect("Mesh::draw_instanced: failed to bind vao");
        let offset = 4;
        let width = 4;
        for i in 0..width {
            let index = i + offset;
            vao.vbo_attrib(&transforms, index, width, width * i * mem::size_of::<f32>())
                .attrib_divisor(index, 1);
            GlError::check().expect("Mesh::draw_instanced: failed to bind transforms");
        }

        timer.time("draw instanced");
        vao.draw_arrays_instanced(fbo, DrawMode::Triangles, 0, self.0.vcount, transforms.len());
        timer.time("draw instanced done");
        GlError::check().expect("Mesh::draw_instanced: failed to draw instanced");
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
                        x => unimplemented!("{:?}", x),
                    }
                }
            }

            let (a, b, c, d) = Vertex::soa(&vertices);
            let mesh = Mesh::new(&a, &b, &c, &d, None, materials);

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
    pub fn draw_instanced<'a, T, S>(
        &mut self,
        timer: &mut S,
        fbo: &T,
        program: &ProgramBinding,
        offsets: &VertexBufferBinder<Mat4>,
    ) where
        T: FramebufferBinderDrawer,
        S: Timer<'a>,
    {
        for mut mesh in self.meshes.iter_mut() {
            let block = timer.block("draw mesh...");
            mesh.bind().draw_instanced(block, fbo, &program, offsets);
            block.end_block();
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
    pub fn bind_multiple(
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
            light.bind(&format!("{}[{}]", name_uniform, i), slot, program);
        }
    }
    pub fn bind_shadow_map(&mut self) -> (FramebufferBinderReadDraw, Mat4) {
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
                GlError::check().expect("Failed to bind light: useShadowMap");
                program.bind_texture("shadowMap", &shadow_map.map, texture_slot);
                GlError::check().expect("Failed to bind light: shadowMap");
                program.bind_float(&ext("farPlane"), shadow_map.far);
                GlError::check().expect("Failed to bind light: farPlane");
            }
            None => {
                program.bind_vec3(&ext("lastPosition"), *position);
                program.bind_bool(&ext("useShadowMap"), false);
                GlError::check().expect("Failed to bind light: useShadowMap");
                // program.bind_int(&ext("shadowMap"), TextureSlot::Zero.into());
                // GlError::check().expect("Failed to not bind light: shadowMap");
                // program.bind_float(&ext("farPlane"), 0.0);
                // GlError::check().expect("Failed to bind light: farPlane");
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
        for (i, light) in lights.iter().enumerate() {
            let slot: i32 = initial_slot.into();
            let slot = (slot as usize + i).into();
            // println!("binding: {} into {:?}", format!("{}[{}]", name_uniform, i), slot);
            light.bind(&format!("{}[{}]", name_uniform, i), slot, program);
        }
        GlError::check().expect("Failed to bind multiple lights");
    }
    pub fn bind_shadow_map(&mut self) -> Option<(FramebufferBinderReadDraw, [Mat4; 6])> {
        let shadow_map = self.shadow_map.as_mut()?;

        let light_space: Mat4 = cgmath::PerspectiveFov {
            fovy: Rad(std::f32::consts::PI / 2.0),
            aspect: (shadow_map.width as f32) / (shadow_map.height as f32),
            near: shadow_map.near,
            far: shadow_map.far,
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
        // (1024, 1024)
        // (2048, 2048)
        // (4096, 4096)
        (8192, 8192)
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
                    GlType::Float,
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
        (1024, 1024)
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
                TextureInternalFormat::Rgba,
                w,
                h,
                TextureFormat::Rgba,
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
