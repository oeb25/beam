use failure::{Error, ResultExt};
use gl;
use image;
use mesh::Mesh;
use mg::*;
use render::primitives;
use std::{collections::HashMap, path::Path};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct TextureRef(usize, (u32, u32));

#[derive(Debug)]
pub struct MeshStore {
    pub meshes: Vec<Mesh>,
    pub textures: Vec<Texture>,

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

            fs_textures: Default::default(),

            // primitive cache
            cube: Default::default(),
            sphere: Default::default(),
        }
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
