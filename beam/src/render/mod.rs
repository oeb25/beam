pub mod lights;
pub mod mesh;
mod primitives;

use cgmath::{self, InnerSpace, Rad};
use collada;
use gl;
use image;

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

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum MaterialProp<T> {
    Texture(TextureRef),
    Value(T),
}

impl<T> Into<MaterialProp<T>> for TextureRef {
    fn into(self) -> MaterialProp<T> {
        MaterialProp::Texture(self)
    }
}

impl Into<MaterialProp<V3>> for V3 {
    fn into(self) -> MaterialProp<V3> {
        MaterialProp::Value(self)
    }
}

impl Into<MaterialProp<f32>> for f32 {
    fn into(self) -> MaterialProp<f32> {
        MaterialProp::Value(self)
    }
}

impl<'a> Into<MaterialProp<f32>> for &'a f32 {
    fn into(self) -> MaterialProp<f32> {
        MaterialProp::Value(*self)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Material {
    normal_: MaterialProp<V3>,
    albedo_: MaterialProp<V3>,
    metallic_: MaterialProp<f32>,
    roughness_: MaterialProp<f32>,
    ao_: MaterialProp<f32>,
    opacity_: MaterialProp<f32>,
}

macro_rules! setter {
    ($name:ident, $field:ident, $typ:ty) => {
        pub fn $name<T: Into<MaterialProp<$typ>>>(&self, $name: T) -> Material {
            let mut new = self.clone();
            new.$field = $name.into();
            new
        }
    }
}

impl Material {
    pub fn new() -> Material {
        Material {
            normal_: v3(0.5, 0.5, 1.0).into(),
            albedo_: v3(1.0, 1.0, 1.0).into(),
            metallic_: 1.0.into(),
            roughness_: 1.0.into(),
            ao_: 1.0.into(),
            opacity_: 1.0.into(),
        }
    }

    setter!(albedo, albedo_, V3);
    setter!(normal, normal_, V3);
    setter!(metallic, metallic_, f32);
    setter!(roughness, roughness_, f32);
    setter!(ao, ao_, f32);
    setter!(opacity, opacity_, f32);

    pub fn bind<P: ProgramBind>(&self, meshes: &MeshStore, program: &P) {
        macro_rules! prop {
            ($name:ident, $field:ident, $fun:ident) => {{
                match &self.$field {
                    MaterialProp::Texture(texture_ref) => {
                        let texture = meshes.get_texture(&texture_ref);
                        program.bind_bool(concat!("use_mat_", stringify!($name)), false);
                        program.bind_texture(concat!("tex_", stringify!($name)), &texture);
                    }
                    MaterialProp::Value(value) => {
                        program.bind_bool(concat!("use_mat_", stringify!($name)), true);
                        program.$fun(concat!("mat_", stringify!($name)), *value);
                    }
                }
            }};
        }

        prop!(normal, normal_, bind_vec3);
        prop!(albedo, albedo_, bind_vec3);
        prop!(metallic, metallic_, bind_float);
        prop!(roughness, roughness_, bind_float);
        prop!(ao, ao_, bind_float);
        prop!(opacity, opacity_, bind_float);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TextureRef(usize, (u32, u32));

#[derive(Debug)]
pub struct MeshStore {
    pub meshes: Vec<Mesh>,
    pub textures: Vec<Texture>,
    pub materials: Vec<Material>,

    pub fs_textures: HashMap<String, TextureRef>,

    // primitive cache
    pub cube: Option<MeshRef>,
    pub sphere: Option<MeshRef>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshRef(usize);

impl MeshStore {
    pub fn new() -> MeshStore {
        MeshStore {
            meshes: Default::default(),
            textures: Default::default(),
            materials: Default::default(),

            fs_textures: Default::default(),

            // primitive cache
            cube: Default::default(),
            sphere: Default::default(),
        }
    }

    pub fn load_collada(
        &mut self,
        vpin: &mut VertexArrayPin,
        path: impl AsRef<Path>,
    ) -> Result<RenderObject, Error> {
        let path = path.as_ref();

        let src = std::fs::read_to_string(path).context("collada src file not found")?;
        let data: collada::Collada =
            collada::Collada::parse(&src).context("failed to parse collada")?;

        let mut normal_src = None;
        let mut albedo_src = None;
        let mut metallic_src = None;
        let mut roughness_src = None;
        let mut ao_src = None;
        let mut opacity_src = None;

        for image in &data.images {
            let x = |x| Some(path.with_file_name(x));
            match image.source.as_str() {
                "DIFF.png" => albedo_src = x(&image.source),
                "NRM.png" => normal_src = x(&image.source),
                "MET.png" => metallic_src = x(&image.source),
                "ROUGH.png" => roughness_src = x(&image.source),
                "AO.png" => ao_src = x(&image.source),
                "OPAC.png" => opacity_src = x(&image.source),
                _ => {}
            }
        }

        let white3 = v3(1.0, 1.0, 1.0);
        let normal3 = v3(0.5, 0.5, 1.0);

        let custom_material = Material::new()
            .normal::<MaterialProp<_>>(
                normal_src
                    .map(|x| self.load_rgb(x))
                    .transpose()?
                    .map(|x| x.into())
                    .unwrap_or_else(|| normal3.into()),
            )
            .albedo::<MaterialProp<_>>(
                albedo_src
                    .map(|x| self.load_srgb(x))
                    .transpose()?
                    .map(|x| x.into())
                    .unwrap_or_else(|| white3.into()),
            )
            .metallic::<MaterialProp<_>>(
                metallic_src
                    .map(|x| self.load_rgb(x))
                    .transpose()?
                    .map(|x| x.into())
                    .unwrap_or_else(|| 1.0.into()),
            )
            .roughness::<MaterialProp<_>>(
                roughness_src
                    .map(|x| self.load_rgb(x))
                    .transpose()?
                    .map(|x| x.into())
                    .unwrap_or_else(|| 1.0.into()),
            )
            .ao::<MaterialProp<_>>(
                ao_src
                    .map(|x| self.load_rgb(x))
                    .transpose()?
                    .map(|x| x.into())
                    .unwrap_or_else(|| 1.0.into()),
            )
            .opacity::<MaterialProp<_>>(
                opacity_src
                    .map(|x| self.load_rgb(x))
                    .transpose()?
                    .map(|x| x.into())
                    .unwrap_or_else(|| 1.0.into()),
            );

        let mut mesh_ids: HashMap<usize, Vec<MeshRef>> = HashMap::new();

        let mut render_objects = vec![];
        for node in &data.visual_scenes[0].nodes {
            use cgmath::Matrix;
            let mut transform = node.transformations.iter().fold(
                Mat4::from_scale(1.0),
                |acc, t| {
                    let t: Mat4 = match t {
                        collada::Transform::Matrix(a) => unsafe {
                            std::mem::transmute::<[f32; 16], [[f32; 4]; 4]>(*a).into()
                        },
                        _ => unimplemented!(),
                    };

                    t * acc
                },
            );
            transform.swap_rows(1, 2);
            transform = transform.transpose();
            transform.swap_rows(1, 2);
            assert_eq!(node.geometry.len(), 1);
            for geom_instance in &node.geometry {
                let mesh_refs = if let Some(mesh_cache) = mesh_ids.get(&geom_instance.geometry.0) {
                    mesh_cache.clone()
                } else {
                    let geom = &data.geometry[geom_instance.geometry.0];
                    match geom {
                        collada::Geometry::Mesh { triangles } => {
                            let mut meshes = vec![];

                            for triangles in triangles.iter() {
                                let collada::MeshTriangles { vertices, .. } = triangles;

                                let mut verts: Vec<_> = vertices
                                    .iter()
                                    .map(|v| {
                                        let pos = v4(v.pos[0], v.pos[1], v.pos[2], 1.0);
                                        Vertex {
                                            // orient y up instead of z, which is default in blender
                                            pos: v3(pos[0], pos[2], pos[1]),
                                            norm: v3(v.nor[0], v.nor[2], v.nor[1]),
                                            tex: v.tex.into(),
                                            tangent: v3(0.0, 0.0, 0.0),
                                            // bitangent: v3(0.0, 0.0, 0.0),
                                        }
                                    })
                                    .collect();

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

                                // let material =
                                //     self.convert_collada_material(&path, &data, material)?;
                                let mesh = Mesh::new(&verts, vpin);
                                let mesh_ref = self.insert_mesh(mesh);

                                meshes.push(mesh_ref);
                            }
                            mesh_ids.insert(geom_instance.geometry.0, meshes.clone());
                            meshes
                        }
                        _ => unimplemented!(),
                    }
                };

                // let material = geom_instance.material
                //     .map(|material_ref| self.convert_collada_material(&path, &data, material_ref));

                let children = mesh_refs
                    .into_iter()
                    .map(|mesh_ref| {
                        RenderObject::mesh(mesh_ref).with_material(custom_material.clone())
                    })
                    .collect();

                render_objects.push(RenderObject {
                    transform,
                    material: None,
                    child: RenderObjectChild::RenderObjects(children),
                });
            }
        }

        Ok(RenderObject::with_children(render_objects))
    }

    pub fn insert_mesh(&mut self, mesh: Mesh) -> MeshRef {
        let mesh_ref = MeshRef(self.meshes.len());
        self.meshes.push(mesh);
        mesh_ref
    }

    #[allow(unused)]
    pub fn get_mesh(&self, mesh_ref: &MeshRef) -> &Mesh {
        &self.meshes[mesh_ref.0]
    }

    pub fn insert_texture(&mut self, texture: Texture, dimensions: (u32, u32)) -> TextureRef {
        let texture_ref = TextureRef(self.textures.len(), dimensions);
        self.textures.push(texture);
        texture_ref
    }

    pub fn get_texture(&self, texture_ref: &TextureRef) -> &Texture {
        &self.textures[texture_ref.0]
    }

    pub fn get_cube(&mut self, vpin: &mut VertexArrayPin) -> MeshRef {
        if let Some(mesh_ref) = self.cube {
            return mesh_ref;
        }

        let verts = primitives::cube_vertices();
        let mesh = Mesh::new(&verts, vpin);
        let mesh_ref = self.insert_mesh(mesh);
        self.cube = Some(mesh_ref);
        mesh_ref
    }

    pub fn get_sphere(&mut self, vpin: &mut VertexArrayPin) -> MeshRef {
        if let Some(cached) = &self.sphere {
            return *cached;
        }

        let verts = primitives::sphere_verticies(0.5, 24, 16);
        let mesh = Mesh::new(&verts, vpin);
        let sphere_ref = self.insert_mesh(mesh);
        self.sphere = Some(sphere_ref);
        sphere_ref
    }

    pub fn load_srgb(&mut self, path: impl AsRef<Path>) -> Result<TextureRef, Error> {
        self.load(path, TextureInternalFormat::Srgb, TextureFormat::Rgb)
    }
    pub fn load_rgb(&mut self, path: impl AsRef<Path>) -> Result<TextureRef, Error> {
        self.load(path, TextureInternalFormat::Rgb, TextureFormat::Rgb)
    }
    pub fn load_hdr(&mut self, path: impl AsRef<Path>) -> Result<TextureRef, Error> {
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
            Ok((texture, (metadata.width, metadata.height)))
        })
    }
    pub fn load(
        &mut self,
        path: impl AsRef<Path>,
        internal_format: TextureInternalFormat,
        format: TextureFormat,
    ) -> Result<TextureRef, Error> {
        let path = path.as_ref();

        self.cache_or_load(path, move || {
            use image::GenericImage;
            let img = image::open(&path).context(format!("could not load image at {:?}", path))?;
            let dimensions = img.dimensions();
            let texture = Texture::new(TextureKind::Texture2d);
            texture
                .bind()
                .parameter_int(TextureParameter::WrapS, gl::REPEAT as i32)
                .parameter_int(TextureParameter::WrapT, gl::REPEAT as i32)
                .parameter_int(TextureParameter::MinFilter, gl::LINEAR as i32)
                .parameter_int(TextureParameter::MagFilter, gl::LINEAR as i32)
                .load_image(TextureTarget::Texture2d, internal_format, format, &img);
            Ok((texture, dimensions))
        })
    }
    fn cache_or_load(
        &mut self,
        path: impl AsRef<Path>,
        f: impl FnOnce() -> Result<(Texture, (u32, u32)), Error>,
    ) -> Result<TextureRef, Error> {
        let path: &Path = path.as_ref();
        let path_str = path.to_str().unwrap();

        if let Some(t) = self.fs_textures.get(path_str) {
            Ok(*t)
        } else {
            let (texture, dimensions) = f()?;
            let texture_ref = self.insert_texture(texture, dimensions);
            self.fs_textures.insert(path_str.to_owned(), texture_ref);
            Ok(texture_ref)
        }
    }
}

#[derive(Debug)]
pub struct Ibl {
    pub cubemap: Texture,
    pub irradiance_map: Texture,
    pub prefilter_map: Texture,
    pub brdf_lut: Texture,
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

    pub fn combined_transformed_verts(&self, meshes: &MeshStore, transform: &Mat4) -> Vec<V3> {
        let apply = |v: &V3, t: &Mat4| {
            let w = t * v4(v.x, v.y, v.z, 1.0);
            v3(w.x, w.y, w.z)
        };

        let transform = self.transform * transform;
        match &self.child {
            RenderObjectChild::Mesh(mesh_ref) => {
                let mesh = meshes.get_mesh(mesh_ref);
                mesh.simple_verts
                    .iter()
                    .map(|v| apply(v, &transform))
                    .collect()
            }
            RenderObjectChild::RenderObjects(children) => children
                .iter()
                .flat_map(|child| child.combined_transformed_verts(meshes, &transform))
                .collect(),
        }
    }
    pub fn raymarch_many<'a, I>(objs: I, meshes: &MeshStore, p: V3, r: V3) -> (usize, f32)
    where
        I: Iterator<Item = &'a RenderObject>,
    {
        let verts: Vec<(usize, V3)> = objs
            .enumerate()
            .flat_map::<Vec<_>, _>(|(i, obj)| {
                obj.combined_transformed_verts(meshes, &Mat4::from_scale(1.0))
                    .into_iter()
                    .map(|v| (i, v))
                    .collect()
            })
            .collect();

        verts.iter().fold((0, -1.0), |(j, d), (i, v)| {
            let vd = (v - p).normalize().dot(r);
            if vd > d {
                (*i, vd)
            } else {
                (j, d)
            }
        })
    }
}

pub struct GRenderPass {
    pub fbo: Framebuffer,
    pub position: Texture,
    pub normal: Texture,
    pub albedo: Texture,
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

        let (position, normal, albedo, mrao) = {
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
            let mrao = create_texture(
                TextureInternalFormat::Rgba8,
                TextureFormat::Rgba,
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

            (position, normal, albedo, mrao)
        };

        GRenderPass {
            fbo,
            // depth,
            position,
            normal,
            albedo,
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
