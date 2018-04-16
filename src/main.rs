#![feature(fs_read_write, stmt_expr_attributes, transpose_result)]

extern crate cgmath;
extern crate genmesh;
extern crate gl;
extern crate glutin;
extern crate image;
extern crate obj;
extern crate time;

use cgmath::{InnerSpace, Rad};

use glutin::GlContext;

use time::{Duration, PreciseTime};

use std::{collections::HashMap, mem, path::Path, rc::Rc};

mod mg;

use mg::*;

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

type V2 = cgmath::Vector2<f32>;
type V3 = cgmath::Vector3<f32>;
type V4 = cgmath::Vector4<f32>;
type P3 = cgmath::Point3<f32>;
type Mat4 = cgmath::Matrix4<f32>;

fn v3(x: f32, y: f32, z: f32) -> V3 {
    V3::new(x, y, z)
}

trait Vertexable {
    // (pos, norm, tex, trangent, bitangent)
    fn sizes() -> (usize, usize, usize, usize, usize);
    fn offsets() -> (usize, usize, usize, usize, usize);
}

struct Vertex {
    pos: V3,
    norm: V3,
    tex: V2,
    tangent: V3,
    bitangent: V3,
}

impl Vertexable for Vertex {
    fn sizes() -> (usize, usize, usize, usize, usize) {
        let float_size = mem::size_of::<f32>();
        (
            size_of!(Vertex, pos) / float_size,
            size_of!(Vertex, norm) / float_size,
            size_of!(Vertex, tex) / float_size,
            size_of!(Vertex, tangent) / float_size,
            size_of!(Vertex, bitangent) / float_size,
        )
    }
    fn offsets() -> (usize, usize, usize, usize, usize) {
        (
            offset_of!(Vertex, pos),
            offset_of!(Vertex, norm),
            offset_of!(Vertex, tex),
            offset_of!(Vertex, tangent),
            offset_of!(Vertex, bitangent),
        )
    }
}

struct Camera {
    pos: V3,
    fov: Rad<f32>,
    aspect: f32,
    yaw: f32,
    pitch: f32,
}

impl Camera {
    fn new(pos: V3, fov: Rad<f32>, aspect: f32) -> Camera {
        Camera {
            pos,
            fov,
            aspect,
            yaw: 0.0,
            pitch: 0.0,
        }
    }

    fn up(&self) -> V3 {
        v3(0.0, 1.0, 0.0)
    }
    fn front(&self) -> V3 {
        let (ps, pc) = self.pitch.sin_cos();
        let (ys, yc) = self.yaw.sin_cos();

        v3(pc * yc, ps, pc * ys).normalize()
    }
    #[allow(unused)]
    fn front_look_at(&self, target: &V3) -> V3 {
        (target - self.pos).normalize()
    }
    fn get_view(&self) -> Mat4 {
        let origo = cgmath::Point3::new(0.0, 0.0, 0.0);
        Mat4::look_at(origo + self.pos, origo + self.pos + self.front(), self.up())
    }
    #[allow(unused)]
    fn get_view_look_at(&self, target: &V3) -> Mat4 {
        let origo = cgmath::Point3::new(0.0, 0.0, 0.0);
        Mat4::look_at(
            origo + self.pos,
            origo + self.pos + self.front_look_at(target),
            self.up(),
        )
    }
    fn get_projection(&self) -> Mat4 {
        cgmath::PerspectiveFov {
            fovy: self.fov,
            aspect: self.aspect,
            near: 0.01,
            far: 100.0,
        }.into()
    }
}

#[allow(unused)]
#[derive(Debug, Clone, Copy)]
enum ImageKind {
    Diffuse,
    Ambient,
    Specular,
    Reflection,
    CubeMap,
    NormalMap,
}

#[derive(Debug)]
struct Image {
    texture: Texture,
    path: String,
    kind: ImageKind,
}
impl Image {
    fn new_from_disk(path: &str, kind: ImageKind) -> Image {
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

// struct EboBinder<'a: 'b, 'b>(&'b VertexArrayBinder<'a>, &'b mut Ebo);
// impl<'a, 'b> EboBinder<'a, 'b> {
//     fn new(vao_binder: &'b VertexArrayBinder<'a>, ebo: &'b mut Ebo) -> EboBinder<'a, 'b> {
//         unsafe { gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ebo.0) }
//         EboBinder(vao_binder, ebo)
//     }
//     fn buffer_data<T>(&self, data: &[T]) {
//         unsafe {
//             gl::BufferData(
//                 gl::ELEMENT_ARRAY_BUFFER,
//                 (data.len() * mem::size_of::<T>()) as isize,
//                 &data[0] as *const _ as *const _,
//                 gl::STATIC_DRAW,
//             );
//         }
//     }
// }
// impl<'a, 'b> Drop for EboBinder<'a, 'b> {
//     fn drop(&mut self) {
//         unsafe { gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, 0); }
//     }
// }

#[allow(unused)]
struct Mesh<V> {
    vertices: Vec<V>,
    indecies: Option<Vec<usize>>,
    textures: Vec<Rc<Image>>,

    vao: VertexArray,
    vbo: VertexBuffer<V>,
    ebo: Option<Ebo<usize>>,
}

impl<V: Vertexable> Mesh<V> {
    fn new(vertices: Vec<V>, indecies: Option<Vec<usize>>, textures: Vec<Rc<Image>>) -> Mesh<V> {
        let mut vao = VertexArray::new();
        let mut vbo = VertexBuffer::from_data(&vertices);
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

            let (pos_s, norm_s, tex_s, tangent_s, _) = V::sizes();
            let (pos_o, norm_o, tex_o, tangent_o, _) = V::offsets();

            let vbo_binder = vbo.bind();

            vao.bind()
                .vbo_attrib(&vbo_binder, 0, pos_s, pos_o)
                .vbo_attrib(&vbo_binder, 1, norm_s, norm_o)
                .vbo_attrib(&vbo_binder, 2, tex_s, tex_o)
                .vbo_attrib(&vbo_binder, 3, tangent_s, tangent_o);
        }

        Mesh {
            vertices,
            indecies,
            textures,
            vao,
            vbo,
            ebo,
        }
    }

    fn bind(&mut self) -> MeshBinding<V> {
        MeshBinding(self)
    }
}

struct MeshBinding<'a, V: 'a>(&'a mut Mesh<V>);
impl<'a, V> MeshBinding<'a, V> {
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
            .draw_arrays(fbo, DrawMode::Triangles, 0, self.0.vertices.len());
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
            self.0.vertices.len(),
            transforms.len(),
        );
    }
}

#[allow(unused)]
struct Model {
    meshes: Vec<Mesh<Vertex>>,
    texture_cache: HashMap<String, Rc<Image>>,
}
impl Model {
    fn new_from_disk(path: &str) -> Model {
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
                                let bitangent =
                                    (delta_pos1 * delta_uv2.y - delta_pos2 * delta_uv1.y) * r;
                                let tangent =
                                    (delta_pos2 * delta_uv1.x - delta_pos1 * delta_uv2.x) * r;

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
                        _ => unimplemented!(),
                    }
                }
            }

            let mesh = Mesh::new(vertices, None, materials);

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

struct CubeMapBuilder<T> {
    back: T,
    front: T,
    right: T,
    bottom: T,
    left: T,
    top: T,
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

#[repr(C)]
#[derive(Debug, Clone)]
struct DirectionalLight {
    ambient: V3,
    diffuse: V3,
    specular: V3,

    direction: V3,
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
}
#[repr(C)]
#[derive(Debug, Clone)]
struct PointLight {
    ambient: V3,
    diffuse: V3,
    specular: V3,

    position: V3,

    constant: f32,
    linear: f32,
    quadratic: f32,
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

#[allow(unused)]
#[derive(Debug, Clone)]
enum Light {
    Directional(DirectionalLight),
    Point(PointLight),
    Spot(SpotLight),
}

impl Light {
    fn bind_to_program(&self, name: &str, program: &ProgramBinding) {
        let ext = |e| format!("{}.{}", name, e);

        match self {
            Light::Directional(light) => {
                let space = light.space();
                let DirectionalLight {
                    ambient,
                    diffuse,
                    specular,
                    direction,
                } = light;
                program.bind_vec3(&ext("ambient"), *ambient);
                program.bind_vec3(&ext("diffuse"), *diffuse);
                program.bind_vec3(&ext("specular"), *specular);

                program.bind_vec3(&ext("direction"), *direction);

                program.bind_mat4(&ext("space"), space);
            }
            Light::Point(PointLight {
                position,
                ambient,
                diffuse,
                specular,
                constant,
                linear,
                quadratic,
            }) => {
                program.bind_vec3(&ext("ambient"), *ambient);
                program.bind_vec3(&ext("diffuse"), *diffuse);
                program.bind_vec3(&ext("specular"), *specular);

                program.bind_vec3(&ext("position"), *position);

                program.bind_float(&ext("constant"), *constant);
                program.bind_float(&ext("linear"), *linear);
                program.bind_float(&ext("quadratic"), *quadratic);
            }
            Light::Spot(SpotLight {
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
            }) => {
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
    }
    fn bind_multiple_to_program(
        lights: &[Light],
        (directional_num, directional_array): (&str, &str),
        (point_num, point_array): (&str, &str),
        (spot_num, spot_array): (&str, &str),
        program: &ProgramBinding,
    ) {
        let mut n_directional = 0;
        let mut n_point = 0;
        let mut n_spot = 0;
        for light in lights {
            let name = match light {
                Light::Directional(_) => {
                    let name = format!("{}[{}]", directional_array, n_directional);
                    n_directional += 1;
                    name
                }
                Light::Point(_) => {
                    let name = format!("{}[{}]", point_array, n_point);
                    n_point += 1;
                    name
                }
                Light::Spot(_) => {
                    let name = format!("{}[{}]", spot_array, n_spot);
                    n_spot += 1;
                    name
                }
            };
            light.bind_to_program(&name, program);
        }
        program.bind_int(directional_num, n_directional);
        program.bind_int(point_num, n_point);
        program.bind_int(spot_num, n_spot);
    }
}

#[allow(unused)]
struct ShadowMap {
    width: u32,
    height: u32,
    fbo: Framebuffer,
    map: Texture,
}

#[allow(unused)]
impl ShadowMap {
    fn new() -> ShadowMap {
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
    fn bind_directional_light(
        &mut self,
        light: &DirectionalLight,
    ) -> (FramebufferBinderReadDraw, Mat4) {
        let light_space = light.space();
        (self.fbo.bind(), light_space)
    }
}

struct PointShadowMap {
    width: u32,
    height: u32,
    near: f32,
    far: f32,
    fbo: Framebuffer,
    map: Texture,
}

impl PointShadowMap {
    fn new() -> PointShadowMap {
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
    fn bind_point_light(&mut self, light: &PointLight) -> (FramebufferBinderReadDraw, [Mat4; 6]) {
        let light_space: Mat4 = cgmath::PerspectiveFov {
            fovy: Rad(std::f32::consts::PI / 2.0),
            aspect: (self.width as f32) / (self.height as f32),
            near: self.near,
            far: self.far,
        }.into();

        let origo = P3::new(0.0, 0.0, 0.0);
        let lp = origo + light.position;
        let look_at = |p, up| light_space * Mat4::look_at(lp, lp + p, up);

        let shadow_transforms = [
            look_at(v3(1.0, 0.0, 0.0), v3(0.0, -1.0, 0.0)),
            look_at(v3(-1.0, 0.0, 0.0), v3(0.0, -1.0, 0.0)),
            look_at(v3(0.0, 1.0, 0.0), v3(0.0, 0.0, 1.0)),
            look_at(v3(0.0, -1.0, 0.0), v3(0.0, 0.0, -1.0)),
            look_at(v3(0.0, 0.0, 1.0), v3(0.0, -1.0, 0.0)),
            look_at(v3(0.0, 0.0, -1.0), v3(0.0, -1.0, 0.0)),
        ];

        (self.fbo.bind(), shadow_transforms)
    }
}

#[derive(Debug, Clone, Copy)]
enum ObjectKind {
    Cube,
    Nanosuit,
    Cyborg,
}

#[derive(Debug, Clone)]
struct Object {
    kind: ObjectKind,
    transform: Mat4,
}

struct Scene<'a> {
    vertex_program: &'a mut Program,
    directional_shadow_program: &'a mut Program,
    point_shadow_program: &'a mut Program,
    directional_lighting_program: &'a mut Program,
    point_lighting_program: &'a mut Program,
    skybox_program: &'a mut Program,
    hdr_program: &'a mut Program,

    nanosuit: &'a mut Model,
    cyborg: &'a mut Model,
    cube: &'a mut Mesh<Vertex>,
    rect: &'a mut Mesh<Vertex>,

    objects: Vec<Object>,
    directional_lights: Vec<DirectionalLight>,
    point_lights: Vec<PointLight>,

    screen_width: u32,
    screen_height: u32,

    g: &'a mut GRenderPass,
    color_a_fbo: &'a mut Framebuffer,
    color_a_tex: &'a Texture,
    color_b_fbo: &'a mut Framebuffer,
    color_b_tex: &'a Texture,

    window_fbo: &'a mut Framebuffer,

    directional_shadow_map: &'a mut ShadowMap,
    point_shadow_map: &'a mut PointShadowMap,

    projection: Mat4,
    view: Mat4,
    view_pos: V3,
    time: f32,
    skybox: &'a Texture,
}

impl<'a> Scene<'a> {
    fn render(&mut self) {
        let mut nanosuit_transforms = vec![];
        let mut cyborg_transforms = vec![];
        let mut cube_transforms = vec![];

        for obj in self.objects.iter() {
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
                .bind_mat4("projection", self.projection)
                .bind_mat4("view", self.view)
                .bind_vec3("viewPos", self.view_pos)
                .bind_float("time", self.time)
                .bind_texture("skybox", self.skybox, TextureSlot::Six);

            render_scene!(fbo, program);
        }

        // Clear backbuffer
        self.color_b_fbo.bind().clear(ClearMask::ColorDepth);

        let mut use_a_buffer = true;

        for light in self.directional_lights.iter() {
            {
                // Render depth map
                unsafe {
                    gl::Viewport(
                        0,
                        0,
                        self.directional_shadow_map.width as i32,
                        self.directional_shadow_map.height as i32,
                    );
                    gl::CullFace(gl::FRONT);
                }
                let p = self.directional_shadow_program.bind();
                let (fbo, light_space) = self.directional_shadow_map.bind_directional_light(&light);
                fbo.clear(ClearMask::Depth);
                p.bind_mat4("lightSpace", light_space);
                render_scene!(fbo, p);
                unsafe {
                    gl::CullFace(gl::BACK);
                    gl::Viewport(
                        0,
                        0,
                        self.screen_width as i32 * 2,
                        self.screen_height as i32 * 2,
                    );
                }
            }
            {
                // Render lighting
                let (fbo, accumulator) = if use_a_buffer {
                    (self.color_a_fbo.bind(), &self.color_b_tex)
                } else {
                    (self.color_b_fbo.bind(), &self.color_a_tex)
                };
                use_a_buffer = !use_a_buffer;
                fbo.clear(ClearMask::ColorDepth);

                {
                    let g = self.directional_lighting_program.bind();
                    g.bind_texture("aPosition", &self.g.position, TextureSlot::Zero)
                        .bind_texture("aNormal", &self.g.normal, TextureSlot::One)
                        .bind_texture("aAlbedoSpec", &self.g.albedo_spec, TextureSlot::Two)
                        .bind_texture(
                            "shadowMap",
                            &self.directional_shadow_map.map,
                            TextureSlot::Three,
                        )
                        .bind_texture("skybox", &self.skybox, TextureSlot::Four)
                        .bind_texture("accumulator", &accumulator, TextureSlot::Five)
                        .bind_vec3("viewPos", self.view_pos);

                    Light::Directional(light.clone()).bind_to_program("light", &g);

                    let mut c = VertexBuffer::from_data(&vec![Mat4::from_scale(1.0)]);
                    self.rect.bind().draw_instanced(&fbo, &g, &c.bind());
                }
            }
        }

        for light in self.point_lights.iter() {
            {
                // Render depth map
                unsafe {
                    gl::Viewport(
                        0,
                        0,
                        self.point_shadow_map.width as i32,
                        self.point_shadow_map.height as i32,
                    );
                    gl::CullFace(gl::FRONT);
                }
                let p = self.point_shadow_program.bind();
                let far = self.point_shadow_map.far;
                let (fbo, light_spaces) = self.point_shadow_map.bind_point_light(&light);
                fbo.clear(ClearMask::Depth);
                p.bind_mat4s("shadowMatrices", &light_spaces)
                    .bind_vec3("lightPos", light.position)
                    .bind_float("farPlane", far);
                render_scene!(fbo, p);
                unsafe {
                    gl::CullFace(gl::BACK);
                    gl::Viewport(
                        0,
                        0,
                        self.screen_width as i32 * 2,
                        self.screen_height as i32 * 2,
                    );
                }
            }
            {
                // Render lighting
                let (fbo, accumulator) = if use_a_buffer {
                    (self.color_a_fbo.bind(), &self.color_b_tex)
                } else {
                    (self.color_b_fbo.bind(), &self.color_a_tex)
                };
                use_a_buffer = !use_a_buffer;
                fbo.clear(ClearMask::ColorDepth);

                {
                    let g = self.point_lighting_program.bind();
                    g.bind_texture("aPosition", &self.g.position, TextureSlot::Zero)
                        .bind_texture("aNormal", &self.g.normal, TextureSlot::One)
                        .bind_texture("aAlbedoSpec", &self.g.albedo_spec, TextureSlot::Two)
                        .bind_texture("shadowMap", &self.point_shadow_map.map, TextureSlot::Three)
                        .bind_texture("skybox", &self.skybox, TextureSlot::Four)
                        .bind_texture("accumulator", &accumulator, TextureSlot::Five)
                        .bind_vec3("viewPos", self.view_pos)
                        .bind_float("farPlane", self.point_shadow_map.far);

                    Light::Point(light.clone()).bind_to_program("light", &g);

                    let mut c = VertexBuffer::from_data(&vec![Mat4::from_scale(1.0)]);
                    self.rect.bind().draw_instanced(&fbo, &g, &c.bind());
                }
            }
        }

        let (color_fbo, color_tex) = if use_a_buffer {
            (&mut self.color_b_fbo, &self.color_b_tex)
        } else {
            (&mut self.color_a_fbo, &self.color_a_tex)
        };

        // Skybox Pass
        {
            // Copy z-buffer over from geometry pass
            let g_binding = self.g.fbo.read();
            let hdr_binding = color_fbo.draw();
            let (w, h) = (self.screen_width as i32 * 2, self.screen_height as i32 * 2);
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
                let mut view = self.view.clone();
                view.w = V4::new(0.0, 0.0, 0.0, 0.0);
                program
                    .bind_mat4("projection", self.projection)
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
                .bind_float("time", self.time);
            let mut c = VertexBuffer::from_data(&vec![Mat4::from_scale(1.0)]);
            self.rect.bind().draw_instanced(&fbo, &hdr, &c.bind());
        }
    }
}

struct GRenderPass {
    fbo: Framebuffer,
    depth: Renderbuffer,
    position: Texture,
    normal: Texture,
    albedo_spec: Texture,
}
impl GRenderPass {
    fn new(w: u32, h: u32) -> GRenderPass {
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

#[derive(Default)]
struct Input {
    w: f32,
    a: f32,
    s: f32,
    d: f32,

    up: f32,
    down: f32,
    left: f32,
    right: f32,

    space: f32,
    shift: f32,
}

fn main() {
    let (screen_width, screen_height) = (800, 600);
    let mut events_loop = glutin::EventsLoop::new();
    let window = glutin::WindowBuilder::new()
        .with_title("Hello, world!")
        .with_dimensions(screen_width, screen_height);
    let context = glutin::ContextBuilder::new().with_vsync(true);
    let gl_window = glutin::GlWindow::new(window, context, &events_loop).unwrap();

    unsafe {
        gl_window.make_current().unwrap();
    }

    gl::load_with(|symbol| gl_window.get_proc_address(symbol) as *const _);

    let mut program = Program::new_from_disk(
        "shaders/shader.vs",
        // Some("shaders/shader.gs"),
        None,
        "shaders/shader.fs",
    ).unwrap();
    let mut normal_program = Program::new_from_disk(
        "shaders/normal.vs",
        Some("shaders/normal.gs"),
        "shaders/normal.fs",
    ).unwrap();
    let mut skybox_program =
        Program::new_from_disk("shaders/skybox.vs", None, "shaders/skybox.fs").unwrap();
    let mut hdr_program = Program::new_from_disk("shaders/hdr.vs", None, "shaders/hdr.fs").unwrap();
    let mut directional_shadow_program =
        Program::new_from_disk("shaders/shadow.vs", None, "shaders/shadow.fs").unwrap();
    let mut point_shadow_program = Program::new_from_disk(
        "shaders/point_shadow.vs",
        Some("shaders/point_shadow.gs"),
        "shaders/point_shadow.fs",
    ).unwrap();
    let mut lighting_program =
        Program::new_from_disk("shaders/lighting.vs", None, "shaders/lighting.fs").unwrap();
    let mut directional_lighting_program = Program::new_from_disk(
        "shaders/lighting.vs",
        None,
        "shaders/directional_lighting.fs",
    ).unwrap();
    let mut point_lighting_program =
        Program::new_from_disk("shaders/lighting.vs", None, "shaders/point_lighting.fs").unwrap();

    let skybox = CubeMapBuilder {
        back: "assets/skybox/back.jpg",
        front: "assets/skybox/front.jpg",
        right: "assets/skybox/right.jpg",
        bottom: "assets/skybox/bottom.jpg",
        left: "assets/skybox/left.jpg",
        top: "assets/skybox/top.jpg",
    }.build();
    let mut nanosuit = Model::new_from_disk("assets/nanosuit_reflection/nanosuit.obj");
    let mut cyborg = Model::new_from_disk("assets/cyborg/cyborg.obj");
    // let mut teapot = Model::new_from_disk("teapot.obj");

    let tex1 = Image::new_from_disk("assets/container2.png", ImageKind::Diffuse);
    let tex2 = Image::new_from_disk("assets/container2_specular.png", ImageKind::Specular);
    let (tex1, tex2) = (Rc::new(tex1), Rc::new(tex2));

    let mut cube_mesh = Mesh::new(cube_vertices(), None, vec![tex1, tex2]);
    let mut rect_mesh = Mesh::new(rect_verticies(), None, vec![]);

    let mut t: f32 = 0.0;

    unsafe {
        gl::Enable(gl::DEPTH_TEST);
        gl::Enable(gl::CULL_FACE);
        gl::Enable(gl::FRAMEBUFFER_SRGB);
    }

    let mut camera = Camera::new(
        v3(t.sin(), (t / 10.0).sin(), 1.0),
        Rad(std::f32::consts::PI / 2.0),
        (screen_width as f32) / (screen_height as f32),
    );

    let mut inputs = Input::default();

    let draw_normals = false;

    let mut running = true;
    let mut last_pos = None;
    let mut is = vec![];
    for i in 0..5 {
        for n in 0..i {
            let x = i as f32 / 2.0;
            let v = v3(n as f32 - x, -i as f32 - 5.0, i as f32 / 2.0) * 2.0;
            let v = Mat4::from_translation(v) * Mat4::from_angle_y(Rad(i as f32 - 1.0));
            let obj = Object {
                kind: ObjectKind::Nanosuit,
                transform: v,
            };
            is.push(obj);
        }
    }
    println!("drawing {} nanosuits", is.len());
    let mut instances = VertexBuffer::from_data(&is);

    let (w, h) = gl_window.get_inner_size().unwrap();
    let (w, h) = (w * 2, h * 2);

    let mut window_fbo = unsafe { Framebuffer::window() };

    let mut g = GRenderPass::new(w, h);
    // let (w, h) = (w / 2, h / 2);

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

    let mut directional_shadow_map = ShadowMap::new();
    let mut point_shadow_map = PointShadowMap::new();

    let mut last_time = PreciseTime::now();
    let fps_step = Duration::seconds(1);
    let mut number_of_frames = 0;

    while running {
        let mut mouse_delta = (0.0, 0.0);

        number_of_frames += 1;
        let now = PreciseTime::now();
        let delta = last_time.to(now);
        if delta > fps_step {
            last_time = now;
            gl_window.set_title(&format!("{}", number_of_frames));
            number_of_frames = 0;
            hdr_program = Program::new_from_disk("shaders/hdr.vs", None, "shaders/hdr.fs").unwrap();

            lighting_program =
                Program::new_from_disk("shaders/lighting.vs", None, "shaders/lighting.fs").unwrap();
        }

        events_loop.poll_events(|event| match event {
            glutin::Event::WindowEvent { event, .. } => match event {
                glutin::WindowEvent::Closed => running = false,
                glutin::WindowEvent::Resized(w, h) => gl_window.resize(w, h),
                glutin::WindowEvent::CursorMoved { position, .. } => {
                    match last_pos {
                        None => {
                            last_pos = Some(position);
                        }
                        Some(lp) => {
                            last_pos = Some(position);
                            mouse_delta = (
                                position.0 as f32 - lp.0 as f32,
                                position.1 as f32 - lp.1 as f32,
                            );
                        }
                    }
                    // let (w, h) = gl_window.get_outer_size().unwrap();
                    // let (x, y) = gl_window.get_position().unwrap();
                    // ignore_next_mouse_move = true;
                    // gl_window.set_cursor_position(x + w as i32 / 2, y + h as i32 / 2).unwrap();
                }
                glutin::WindowEvent::KeyboardInput { input, .. } => {
                    if let Some(keycode) = input.virtual_keycode {
                        use glutin::VirtualKeyCode as Kc;

                        let value = if input.state == glutin::ElementState::Pressed {
                            1.0
                        } else {
                            0.0
                        };

                        match keycode {
                            Kc::Escape => running = false,
                            Kc::W => inputs.w = value,
                            Kc::A => inputs.a = value,
                            Kc::S => inputs.s = value,
                            Kc::D => inputs.d = value,
                            Kc::Up => inputs.up = value,
                            Kc::Down => inputs.down = value,
                            Kc::Left => inputs.left = value,
                            Kc::Right => inputs.right = value,
                            Kc::Space => inputs.space = value,
                            Kc::LShift => inputs.shift = value,
                            _ => {}
                        }
                    }
                }
                _ => {}
            },
            _ => (),
        });

        t += 1.0;

        let pi = std::f32::consts::PI;

        let up = camera.up();
        let right = camera.front().cross(up).normalize();
        let front = up.cross(right).normalize();

        let walk_speed = 0.1;
        let sensitivity = 0.005;

        camera.pos += walk_speed
            * (front * (inputs.w - inputs.s) + right * (inputs.d - inputs.a)
                + up * (inputs.space - inputs.shift));
        camera.yaw += sensitivity * mouse_delta.0;
        camera.pitch = (camera.pitch - sensitivity * mouse_delta.1)
            .max(-pi / 2.001)
            .min(pi / 2.001);
        // camera.yaw += sensitivity * (inputs.right - inputs.left);
        // camera.pitch = (camera.pitch + sensitivity * (inputs.up - inputs.down))
        //     .max(-pi / 2.001)
        //     .min(pi / 2.001);

        let light_pos = v3(
            1.5 + -20.0 * (t / 10.0).sin(),
            1.0,
            -20.0 * (t / 20.0).sin(),
        );
        let light_pos2 = v3(
            1.5 + -10.0 * ((t + 23.0) / 14.0).sin(),
            2.0,
            -10.0 * (t / 90.0).sin(),
        );
        // let light_pos2 = camera.pos + v3(
        //     (t / 10.0).sin() * 2.0,
        //     3.0,
        //     (t / 10.0).cos() * 2.0,
        // );

        let one = v3(1.0, 1.0, 1.0);

        let sun = DirectionalLight {
            diffuse: v3(0.8, 0.8, 0.8) * 1.0,
            ambient: one * 0.1,
            specular: v3(0.8, 0.8, 0.8) * 0.2,

            direction: v3(1.5, 1.0, 0.0).normalize(),
        };

        let point_light_1 = PointLight {
            diffuse: v3(0.2, 0.2, 0.2),
            ambient: one * 0.0,
            specular: one * 0.2,

            position: light_pos2,

            constant: 1.0,
            linear: 0.07,
            quadratic: 0.007,
        };
        let point_light_2 = PointLight {
            diffuse: v3(0.4, 0.3, 0.3),
            ambient: one * 0.0,
            specular: one * 0.2,

            position: light_pos,
            constant: 1.0,
            linear: 0.07,
            quadratic: 0.017,
        };

        let lights = vec![
            // Light::Directional(sun.clone()),
            Light::Point(point_light_1.clone()),
            Light::Point(point_light_2.clone()),
            // Light::Spot(SpotLight {
            //     diffuse: v3(1.0, 1.0, 1.0),
            //     ambient: one * 0.0,
            //     specular: v3(0.2, 0.2, 0.2),

            //     position: camera.pos,
            //     direction: camera.front(),

            //     cut_off: Rad(0.2181661565),
            //     outer_cut_off: Rad(0.3054326191),

            //     constant: 1.0,
            //     linear: 0.014,
            //     quadratic: 0.0007,
            // }),
        ];

        let view = camera.get_view();
        let projection = camera.get_projection();

        // {
        //     window_fbo.bind().clear(ClearMask::ColorDepth);
        // }

        macro_rules! render_scene {
            ($fbo:expr, $program:expr) => {{
                {
                    let model = Mat4::from_scale(1.0 / 4.0) * Mat4::from_translation(v3(0.0, 0.0, 4.0));
                    $program.bind_mat4("model", model);
                    // nanosuit.draw_instanced(&$fbo, &$program, &instances.bind());
                }
                {
                    let model = Mat4::from_angle_y(Rad(pi));
                    $program.bind_mat4("model", model);
                    // cyborg.draw_instanced(&$fbo, &$program, &instances.bind());
                }
                {
                    let model = Mat4::from_scale(1.0);
                    $program.bind_mat4("model", model);
                    let mut c = VertexBuffer::from_data(&vec![
                        Mat4::from_translation(light_pos),
                        Mat4::from_translation(light_pos2),
                        Mat4::from_nonuniform_scale(100.0, 0.1, 100.0)
                            * Mat4::from_translation(v3(0.0, -200.0, 0.0)),
                    ]);
                    cube_mesh.bind().draw_instanced(&$fbo, &$program, &c.bind());
                }
            }};
        };

        // Begin rendering!
        if false {
            {
                let fbo = g.fbo.bind();
                fbo.clear(ClearMask::ColorDepth);
                {
                    let program = program.bind();
                    program
                        .bind_mat4("projection", projection)
                        .bind_mat4("view", view)
                        .bind_vec3("lightPos", light_pos)
                        .bind_vec3("viewPos", camera.pos)
                        .bind_float("time", t)
                        .bind_texture("skybox", &skybox.texture, TextureSlot::Six);

                    // Light::bind_multiple_to_program(
                    //     &lights,
                    //     ("numDirectionalLights", "directionalLights"),
                    //     ("numPointLights", "pointLights"),
                    //     ("numSpotLights", "spotLights"),
                    //     &program,
                    // );

                    render_scene!(fbo, program);
                }
            }
            if false {
                // Draw shadow map for directional light
                unsafe {
                    gl::Viewport(
                        0,
                        0,
                        directional_shadow_map.width as i32,
                        directional_shadow_map.height as i32,
                    );
                    gl::CullFace(gl::FRONT);
                }
                let (fbo, light_space) = directional_shadow_map.bind_directional_light(&sun);
                fbo.clear(ClearMask::Depth);
                let program = directional_shadow_program.bind();
                program.bind_mat4("lightSpace", light_space);

                render_scene!(fbo, program);
                unsafe {
                    gl::CullFace(gl::BACK);
                    gl::Viewport(0, 0, screen_width as i32 * 2, screen_height as i32 * 2);
                }
            }
            if true {
                // Draw shadow map for point light
                unsafe {
                    gl::Viewport(
                        0,
                        0,
                        point_shadow_map.width as i32,
                        point_shadow_map.height as i32,
                    );
                    gl::CullFace(gl::FRONT);
                }
                let far = point_shadow_map.far;
                let (fbo, shadow_matrices) = point_shadow_map.bind_point_light(&point_light_1);
                fbo.clear(ClearMask::Depth);
                let program = point_shadow_program.bind();
                program
                    .bind_mat4s("shadowMatrices", &shadow_matrices)
                    .bind_vec3("lightPos", point_light_1.position)
                    .bind_float("farPlane", far);

                render_scene!(fbo, program);
                unsafe {
                    gl::CullFace(gl::BACK);
                    gl::Viewport(0, 0, screen_width as i32 * 2, screen_height as i32 * 2);
                }
            }
            {
                // Lighting pass
                let fbo = color_a_fbo.bind();
                fbo.clear(ClearMask::ColorDepth);

                {
                    let program = lighting_program.bind();
                    program.bind_texture("aPosition", &g.position, TextureSlot::Zero)
                    .bind_texture("aNormal", &g.normal, TextureSlot::One)
                    .bind_texture("aAlbedoSpec", &g.albedo_spec, TextureSlot::Two)
                    // .bind_texture("shadowMap", &directional_shadow_map.map, TextureSlot::Three)
                    .bind_texture("pointShadowMap", &point_shadow_map.map, TextureSlot::Three)
                    .bind_float("farPlane", point_shadow_map.far)
                    .bind_texture("skybox", &skybox.texture, TextureSlot::Four)
                    .bind_vec3("viewPos", camera.pos);

                    Light::bind_multiple_to_program(
                        &lights,
                        ("numDirectionalLights", "directionalLights"),
                        ("numPointLights", "pointLights"),
                        ("numSpotLights", "spotLights"),
                        &program,
                    );
                    let mut c = VertexBuffer::from_data(&vec![Mat4::from_scale(1.0)]);
                    rect_mesh.bind().draw_instanced(&fbo, &program, &c.bind());
                }
            }
            {
                let g_binding = g.fbo.read();
                let hdr_binding = color_a_fbo.draw();
                let (w, h) = (w as i32, h as i32);
                hdr_binding.blit_framebuffer(
                    &g_binding,
                    (0, 0, w, h),
                    (0, 0, w, h),
                    gl::DEPTH_BUFFER_BIT,
                    gl::NEAREST,
                );
            }
            {
                let fbo = color_a_fbo.bind();

                if draw_normals {
                    let program = normal_program.bind();
                    program
                        .bind_mat4("projection", projection)
                        .bind_mat4("view", view);
                    render_scene!(fbo, program);
                }
                {
                    unsafe {
                        gl::DepthFunc(gl::LEQUAL);
                        gl::Disable(gl::CULL_FACE);
                    }
                    let program = skybox_program.bind();
                    let mut view = view.clone();
                    view.w = V4::new(0.0, 0.0, 0.0, 0.0);
                    program
                        .bind_mat4("projection", projection)
                        .bind_mat4("view", view)
                        .bind_texture("skybox", &skybox.texture, TextureSlot::Ten);
                    cube_mesh.bind().draw(&fbo, &program);
                    unsafe {
                        gl::DepthFunc(gl::LESS);
                        gl::Enable(gl::CULL_FACE);
                    }
                }
            }
            {
                let fbo = window_fbo.bind();
                fbo.clear(ClearMask::ColorDepth);

                let hdr = hdr_program.bind();
                hdr.bind_texture("hdrBuffer", &color_a_tex, TextureSlot::Zero)
                    .bind_texture("shadowMap", &point_shadow_map.map, TextureSlot::One)
                    .bind_float("farPlane", point_shadow_map.far)
                    .bind_float("time", t);
                let mut c = VertexBuffer::from_data(&vec![Mat4::from_scale(1.0)]);
                rect_mesh.bind().draw_instanced(&fbo, &hdr, &c.bind());
            }
        } else {
            let mut objects = vec![
                Object {
                    kind: ObjectKind::Cube,
                    transform: Mat4::from_nonuniform_scale(100.0, 0.1, 100.0)
                        * Mat4::from_translation(v3(0.0, -200.0, 0.0)),
                },
                Object {
                    kind: ObjectKind::Cube,
                    transform: Mat4::from_translation(light_pos),
                },
                Object {
                    kind: ObjectKind::Cube,
                    transform: Mat4::from_translation(light_pos2),
                },
            ];

            objects.append(&mut is.clone());

            Scene {
                vertex_program: &mut program,
                directional_shadow_program: &mut directional_shadow_program,
                point_shadow_program: &mut point_shadow_program,
                directional_lighting_program: &mut directional_lighting_program,
                point_lighting_program: &mut point_lighting_program,
                skybox_program: &mut skybox_program,
                hdr_program: &mut hdr_program,

                nanosuit: &mut nanosuit,
                cyborg: &mut cyborg,
                cube: &mut cube_mesh,
                rect: &mut rect_mesh,

                objects: objects,
                directional_lights: vec![sun],
                point_lights: vec![point_light_1, point_light_2],

                screen_width: screen_width,
                screen_height: screen_height,

                g: &mut g,
                color_a_fbo: &mut color_a_fbo,
                color_a_tex: &color_a_tex,
                color_b_fbo: &mut color_b_fbo,
                color_b_tex: &color_b_tex,

                window_fbo: &mut window_fbo,

                directional_shadow_map: &mut directional_shadow_map,
                point_shadow_map: &mut point_shadow_map,

                projection: projection,
                view: view,
                view_pos: camera.pos,
                time: t,
                skybox: &skybox.texture,
            }.render();
        }

        gl_window.swap_buffers().unwrap();
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
            bitangent: tangent.cross($norm.into()),
        }
    }};
}

fn rect_verticies() -> Vec<Vertex> {
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
