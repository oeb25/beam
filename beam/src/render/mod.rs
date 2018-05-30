pub mod lights;
pub mod mesh;
mod primitives;

use cgmath::{self, InnerSpace, Rad};
use collada;
use gl;
use image;

use failure::{Error, ResultExt};

use std::{self, collections::HashMap, path::Path};

use mg::{
    Attachment, Framebuffer, FramebufferBinderBase, FramebufferBinderDraw, FramebufferBinderDrawer,
    FramebufferBinderRead, FramebufferBinderReadDraw, GlError, GlType, Mask, ProgramBind,
    Renderbuffer, Texture, TextureFormat, TextureInternalFormat, TextureKind, TextureParameter,
    TextureTarget,
};

use mesh::{calculate_tangent_and_bitangent, Mesh};
use misc::{v3, v4, Cacher, Mat4, P3, V3, V4, Vertex};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Material {
    pub normal: TextureRef,
    pub albedo: TextureRef,
    pub metallic: TextureRef,
    pub roughness: TextureRef,
    pub ao: TextureRef,
    pub opacity: TextureRef,
}

impl Material {
    pub fn bind<P>(&self, meshes: &MeshStore, program: &P)
    where
        P: ProgramBind,
    {
        program
            .bind_texture("tex_albedo", meshes.get_texture(&self.albedo))
            .bind_texture("tex_metallic", meshes.get_texture(&self.metallic))
            .bind_texture("tex_roughness", meshes.get_texture(&self.roughness))
            .bind_texture("tex_normal", meshes.get_texture(&self.normal))
            .bind_texture("tex_ao", meshes.get_texture(&self.ao))
            .bind_texture("tex_opacity", meshes.get_texture(&self.opacity));
    }

    #[allow(unused)]
    pub fn into_borrowed<'a>(&self, meshes: &'a MeshStore) -> MaterialBorrowed<'a> {
        MaterialBorrowed {
            normal: meshes.get_texture(&self.normal),
            albedo: meshes.get_texture(&self.albedo),
            metallic: meshes.get_texture(&self.metallic),
            roughness: meshes.get_texture(&self.roughness),
            ao: meshes.get_texture(&self.ao),
            opacity: meshes.get_texture(&self.opacity),
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct MaterialBorrowed<'a> {
    pub normal: &'a Texture,
    pub albedo: &'a Texture,
    pub metallic: &'a Texture,
    pub roughness: &'a Texture,
    pub ao: &'a Texture,
    pub opacity: &'a Texture,
}

impl<'a> MaterialBorrowed<'a> {
    #[allow(unused)]
    pub fn bind<P>(&self, program: &P)
    where
        P: ProgramBind,
    {
        program
            .bind_texture("tex_albedo", &self.albedo)
            .bind_texture("tex_metallic", &self.metallic)
            .bind_texture("tex_roughness", &self.roughness)
            .bind_texture("tex_normal", &self.normal)
            .bind_texture("tex_ao", &self.ao)
            .bind_texture("tex_opacity", &self.opacity);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureRef(usize);

#[derive(Default)]
pub struct MeshStore {
    meshes: Vec<Mesh>,
    textures: Vec<Texture>,

    fs_textures: HashMap<String, TextureRef>,

    rgb_textures: Cacher<V3, TextureRef>,
    rgba_textures: Cacher<V4, TextureRef>,

    // primitive cache
    cube: Option<MeshRef>,
    spheres: Cacher<f32, MeshRef>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshRef(usize);

impl MeshStore {
    pub fn load_collada(&mut self, path: impl AsRef<Path>) -> Result<RenderObject, Error> {
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

        for image in data.images.iter() {
            let x = |x| Some(path.with_file_name(x));
            match image.name.as_str() {
                "DIFF" => albedo_src = x(&image.source),
                "ROUGH" => roughness_src = x(&image.source),
                "MET" => metallic_src = x(&image.source),
                "NRM" => normal_src = x(&image.source),
                "AO" => ao_src = x(&image.source),
                "OPAC" => opacity_src = x(&image.source),
                _ => {}
            }
        }

        let white3 = self.rgb_texture(v3(1.0, 1.0, 1.0));

        let custom_material = Material {
            normal: normal_src
                .map(|x| self.load_rgb(x))
                .transpose()?
                .unwrap_or_else(|| self.rgb_texture(v3(0.5, 0.5, 1.0))),
            albedo: albedo_src
                .map(|x| self.load_srgb(x))
                .transpose()?
                .unwrap_or_else(|| white3),
            metallic: metallic_src
                .map(|x| self.load_rgb(x))
                .transpose()?
                .unwrap_or_else(|| white3),
            roughness: roughness_src
                .map(|x| self.load_rgb(x))
                .transpose()?
                .unwrap_or_else(|| white3),
            ao: ao_src
                .map(|x| self.load_rgb(x))
                .transpose()?
                .unwrap_or_else(|| white3),
            opacity: opacity_src
                .map(|x| self.load_rgb(x))
                .transpose()?
                .unwrap_or_else(|| white3),
        };

        let mut mesh_ids: HashMap<usize, Vec<(MeshRef, Material)>> = HashMap::new();

        let mut render_objects = vec![];
        for node in data.visual_scenes[0].nodes.iter() {
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
            for geom_instance in node.geometry.iter() {
                let mesh_refs = if let Some(mesh_cache) = mesh_ids.get(&geom_instance.geometry.0) {
                    mesh_cache.clone()
                } else {
                    let geom = &data.geometry[geom_instance.geometry.0];
                    let meshes = match geom {
                        collada::Geometry::Mesh { triangles } => {
                            let mut meshes = vec![];

                            for triangles in triangles.iter() {
                                let collada::MeshTriangles { vertices, material } = triangles;

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

                                let material =
                                    self.convert_collada_material(&path, &data, material)?;
                                let mesh = Mesh::new(&verts);
                                let mesh_ref = self.insert_mesh(mesh);

                                meshes.push((mesh_ref, material));
                            }
                            mesh_ids.insert(geom_instance.geometry.0, meshes.clone());
                            meshes
                        }
                        _ => unimplemented!(),
                    };
                    meshes
                };

                // let material = geom_instance.material
                //     .map(|material_ref| self.convert_collada_material(&path, &data, material_ref));

                let children = mesh_refs
                    .into_iter()
                    .map(|(mesh_ref, _material)| {
                        RenderObject::mesh(mesh_ref).with_material(custom_material)
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

    fn convert_collada_material(
        &mut self,
        path: &Path,
        data: &collada::Collada,
        material_ref: &collada::MaterialRef,
    ) -> Result<Material, Error> {
        let collada::Material::Effect(effect_ref) = data.materials[material_ref.0];
        let collada::Effect::Phong {
            emission: _,
            ambient: _,
            diffuse,
            specular: _,
            shininess: _,
            index_of_refraction: _,
        } = &data.effects[effect_ref.0];

        let white = self.rgb_texture(v3(1.0, 1.0, 1.0));

        let convert = |c: &collada::PhongProperty| {
            use collada::PhongProperty::*;
            match c {
                Color(color) => Ok(self.rgba_texture((*color).into())),
                Float(f) => Ok(self.rgb_texture(v3(*f, *f, *f))),
                Texture(image_ref) => {
                    let img = &data.images[image_ref.0];
                    // TODO: How do we determin what kind of file it is?
                    self.load_srgb(path.with_file_name(&img.source))
                }
            }
        };

        let albedo = diffuse
            .as_ref()
            .map(convert)
            .transpose()?
            .unwrap_or_else(|| white);

        Ok(Material {
            normal: self.rgb_texture(v3(0.5, 0.5, 1.0)),
            albedo,
            metallic: white,
            roughness: self.rgb_texture(v3(0.5, 0.5, 0.5)),
            ao: white,
            opacity: white,
        })
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

    pub fn get_mesh_mut(&mut self, mesh_ref: &MeshRef) -> &mut Mesh {
        &mut self.meshes[mesh_ref.0]
    }

    pub fn insert_texture(&mut self, texture: Texture) -> TextureRef {
        let texture_ref = TextureRef(self.textures.len());
        self.textures.push(texture);
        texture_ref
    }

    pub fn get_texture(&self, texture_ref: &TextureRef) -> &Texture {
        &self.textures[texture_ref.0]
    }

    fn color_texture(&mut self, color: &[f32]) -> TextureRef {
        let texture = Texture::new(TextureKind::Texture2d);
        let is_rgb = color.len() == 3;
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
                    if is_rgb {
                        TextureInternalFormat::Rgb
                    } else {
                        TextureInternalFormat::Rgba
                    },
                    1,
                    1,
                    if is_rgb {
                        TextureFormat::Rgb
                    } else {
                        TextureFormat::Rgba
                    },
                    color,
                );
        }
        let id = TextureRef(self.textures.len());
        self.textures.push(texture);
        id
    }

    pub fn rgb_texture(&mut self, color: V3) -> TextureRef {
        if let Some(texture_ref) = self.rgb_textures.get(&color) {
            *texture_ref
        } else {
            let id = self.color_texture(&[color.x, color.y, color.z]);
            *self.rgb_textures.insert(color, id)
        }
    }

    pub fn rgba_texture(&mut self, color: V4) -> TextureRef {
        if let Some(texture_ref) = self.rgba_textures.get(&color) {
            *texture_ref
        } else {
            let id = self.color_texture(&[color.x, color.y, color.z, color.w]);
            *self.rgba_textures.insert(color, id)
        }
    }

    pub fn get_cube(&mut self) -> MeshRef {
        if let Some(mesh_ref) = self.cube {
            return mesh_ref;
        }

        let verts = primitives::cube_vertices();
        let mesh = Mesh::new(&verts);
        let mesh_ref = self.insert_mesh(mesh);
        self.cube = Some(mesh_ref);
        mesh_ref
    }

    pub fn get_sphere(&mut self, radius: f32) -> MeshRef {
        if let Some(cached) = self
            .spheres
            .iter()
            .find(|(cache_level, _)| *cache_level == radius)
        {
            return cached.1;
        }

        let verts = primitives::sphere_verticies(radius, 24, 16);
        let mesh = Mesh::new(&verts);
        let mesh_ref = self.insert_mesh(mesh);
        self.spheres.insert(radius, mesh_ref);
        mesh_ref
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
            Ok(texture)
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
            let img = image::open(&path).context(format!("could not load image at {:?}", path))?;
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
        f: impl FnOnce() -> Result<Texture, Error>,
    ) -> Result<TextureRef, Error> {
        let path: &Path = path.as_ref();
        let path_str = path.to_str().unwrap();

        if let Some(t) = self.fs_textures.get(path_str) {
            Ok(*t)
        } else {
            let texture = f()?;
            let texture_ref = self.insert_texture(texture);
            self.fs_textures.insert(path_str.to_owned(), texture_ref);
            Ok(texture_ref)
        }
    }

    pub fn load_pbr_with_default_filenames(
        &mut self,
        path: impl AsRef<Path>,
        extension: &str,
    ) -> Result<Material, Error> {
        let path = path.as_ref();
        let x = |map| path.join(map).with_extension(extension);

        Ok(Material {
            albedo: self
                .load_srgb(x("albedo"))
                .context("failed to load pbr albedo")?,
            metallic: self
                .load_rgb(x("metallic"))
                .context("failed to load pbr metallic")?,
            roughness: self
                .load_rgb(x("roughness"))
                .context("failed to load pbr roughness")?,
            normal: self
                .load_rgb(x("normal"))
                .context("failed to load pbr normal")?,
            ao: self.load_rgb(x("ao")).context("failed to load pbr ao")?,
            opacity: self
                .load_rgb(x("opacity"))
                .context("failed to load pbr opacity")?,
        })
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

    pub fn combined_transformed_verts_within_distance(
        &self,
        meshes: &MeshStore,
        transform: &Mat4,
        p: &V3,
        max_dist: f32,
    ) -> Vec<V3> {
        fn apply(v: &V3, t: &Mat4) -> V3 {
            let w = t * v4(v.x, v.y, v.z, 1.0);
            v3(w.x, w.y, w.z)
        }

        let transform = self.transform * transform;

        match &self.child {
            RenderObjectChild::Mesh(mesh_ref) => {
                let mesh = meshes.get_mesh(mesh_ref);
                mesh.simple_verts
                    .iter()
                    .map(|v| apply(v, &transform))
                    .filter(|v| (v - p).magnitude() < max_dist)
                    .collect()
            }
            RenderObjectChild::RenderObjects(children) => children
                .iter()
                .flat_map(|child| {
                    child
                        .combined_transformed_verts_within_distance(meshes, &transform, p, max_dist)
                })
                .collect(),
        }
    }

    pub fn raymarch(&self, meshes: &MeshStore, p: V3, d: V3) -> f32 {
        let verts = self.combined_transformed_verts_within_distance(
            meshes,
            &Mat4::from_scale(0.0),
            &p,
            6.0,
        );

        let closest_vert = |dp: &V3| {
            verts
                .iter()
                .enumerate()
                .fold((0, std::f32::MAX), |(j, d), (i, v)| {
                    let vd = (dp - v).magnitude();
                    if vd < d {
                        (i, vd)
                    } else {
                        (j, d)
                    }
                })
        };

        const MAX_ITER: usize = 50;
        const EPOSILON: f32 = 1.0;

        let mut dp = p;
        let mut h = closest_vert(&dp);
        for _ in 0..MAX_ITER {
            if h.1 < EPOSILON {
                return h.1;
            }
            dp = dp + h.1 * d;
            h = closest_vert(&dp);
        }
        std::f32::MAX
    }
}

pub struct GRenderPass {
    pub fbo: Framebuffer,
    pub position: Texture,
    pub normal: Texture,
    pub albedo: Texture,
    pub metallic_roughness_ao_opacity: Texture,

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

        let (position, normal, albedo, metallic_roughness_ao_opacity) = {
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
            let metallic_roughness_ao_opacity = create_texture(
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

            (position, normal, albedo, metallic_roughness_ao_opacity)
        };

        GRenderPass {
            fbo,
            // depth,
            position,
            normal,
            albedo,
            metallic_roughness_ao_opacity,

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
pub fn cubemap_from_equirectangular<F, P>(size: u32, program: &P, render_cube: F) -> Texture
where
    F: FnMut(&FramebufferBinderReadDraw, &P),
    P: ProgramBind,
{
    map_cubemap(size, 1, program, render_cube)
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
