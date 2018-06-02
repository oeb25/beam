use failure::{Error, ResultExt};
use mg::{
    DrawMode, FramebufferBinderDrawer, FramebufferBinderReadDraw, Mask, Program, ProgramBind,
    ProgramBinding, ProgramPin, TextureSlot, VertexArray, VertexArrayPin, VertexBuffer,
};
use misc::{v3, V3};
use pipeline::Pipeline;
use render::{
    create_irradiance_map, create_prefiler_map, cubemap_from_equirectangular, Ibl, Material,
    MeshRef, MeshStore, RenderObject, RenderTarget, TextureRef,
};
use std::{cell::RefCell, fs, path::Path};

#[derive(Debug)]
struct AssetBuilderPrograms {
    equirectangular: Program,
    convolute_cubemap: Program,
    prefilter_cubemap: Program,
    brdf_lut: Program,
}

impl AssetBuilderPrograms {
    fn preprocess(path: impl AsRef<Path>) -> Result<String, Error> {
        let src = fs::read_to_string(&path)?;
        Ok(format!("#version 410 core\n{}", src))
    }
    fn new() -> Result<AssetBuilderPrograms, Error> {
        macro_rules! shader {
            ($vert:expr, $geom:expr, $frag:expr) => {{
                let vert = AssetBuilderPrograms::preprocess($vert)?;
                let geom = $geom
                    .map(|path| AssetBuilderPrograms::preprocess(path))
                    .transpose()?;
                let frag = AssetBuilderPrograms::preprocess($frag)?;
                Program::new_from_src(
                    &vert,
                    match &geom {
                        Some(x) => Some(x),
                        None => None,
                    },
                    &frag,
                ).expect(&format!("failed to compile {:?}", (vert, geom, frag)))
            }};
            ($vert:expr, $geom:expr, $frag:expr,) => {
                shader!($vert, $geom, $frag)
            };
        }

        macro_rules! fshader {
            ($frag:expr) => {
                shader!("shaders/rect.vert", Some("shaders/rect.geom"), $frag)
            };
        }

        macro_rules! cshader {
            ($frag:expr) => {
                shader!("shaders/rect.vert", Some("shaders/cube.geom"), $frag)
            };
        }

        Ok(AssetBuilderPrograms {
            equirectangular: cshader!("shaders/equirectangular.frag"),
            convolute_cubemap: cshader!("shaders/convolute_cubemap.frag"),
            prefilter_cubemap: cshader!("shaders/prefilter_cubemap.frag"),
            brdf_lut: fshader!("shaders/brdf.frag"),
        })
    }
}

#[derive(Debug)]
pub struct AssetBuilder<'a> {
    ppin: &'a mut ProgramPin,
    vpin: &'a mut VertexArrayPin,
    programs: AssetBuilderPrograms,
    meshes: MeshStore,
}

impl<'a> AssetBuilder<'a> {
    pub fn new(
        ppin: &'a mut ProgramPin,
        vpin: &'a mut VertexArrayPin,
    ) -> Result<AssetBuilder<'a>, Error> {
        let programs = AssetBuilderPrograms::new()?;
        let meshes = MeshStore::new();

        Ok(AssetBuilder {
            ppin,
            vpin,
            programs,
            meshes,
        })
    }
    fn make_rect(vpin: &mut VertexArrayPin) -> VertexArray {
        let rect_vao = VertexArray::new();
        let mut rect_vbo = VertexBuffer::from_data(&[v3(0.0, 0.0, 0.0)]);
        rect_vao.bind(vpin).vbo_attrib(&rect_vbo.bind(), 0, 3, 0);
        rect_vao
    }
    pub fn load_collada(&mut self, path: impl AsRef<Path>) -> Result<RenderObject, Error> {
        self.meshes.load_collada(self.vpin, path)
    }
    pub fn load_pbr_with_default_filenames(
        &mut self,
        path: impl AsRef<Path>,
        extension: &str,
    ) -> Result<Material, Error> {
        let path = path.as_ref();
        let x = |map| path.join(map).with_extension(extension);

        let builder = Material::new()
            .albedo(
                self.meshes.load_srgb(x("albedo"))
                    .context("failed to load pbr albedo")?,
            )
            .metallic(
                self.meshes.load_rgb(x("metallic"))
                    .context("failed to load pbr metallic")?,
            )
            .roughness(
                self.meshes.load_rgb(x("roughness"))
                    .context("failed to load pbr roughness")?,
            )
            .normal(
                self.meshes.load_rgb(x("normal"))
                    .context("failed to load pbr normal")?,
            )
            .ao(self.meshes.load_rgb(x("ao")).context("failed to load pbr ao")?)
            .opacity(
                self.meshes.load_rgb(x("opacity"))
                    .context("failed to load pbr opacity")?,
            );

        Ok(builder)
    }
    pub fn load_ibl(&mut self, path: impl AsRef<Path>) -> Result<Ibl, Error> {
        let vpin = &mut self.vpin;

        let ibl_raw_ref = self.meshes.load_hdr(path)?;
        let ibl_raw = self.meshes.get_texture(&ibl_raw_ref);
        // Borrow checker work around here, we could use the rect vao already define on the pipeline
        // if it were not for the overship issues, but this works, and I don't think it is all that
        // costly.
        let rect = AssetBuilder::make_rect(vpin);

        let mut render_cube = |fbo: &FramebufferBinderReadDraw, program: &ProgramBinding| {
            program.bind_texture_to("equirectangularMap", &ibl_raw, TextureSlot::One);
            rect.bind(vpin)
                .draw_arrays(fbo, program, DrawMode::Points, 0, 1);
        };

        let cubemap = cubemap_from_equirectangular(
            512,
            &self.programs.equirectangular.bind(self.ppin),
            &mut render_cube,
        );
        cubemap.bind().generate_mipmap();

        let mut render_cube = |fbo: &FramebufferBinderReadDraw, program: &ProgramBinding| {
            program.bind_texture_to("equirectangularMap", &cubemap, TextureSlot::One);
            rect.bind(vpin)
                .draw_arrays(fbo, program, DrawMode::Points, 0, 1);
        };
        let irradiance_map = create_irradiance_map(
            32,
            &self.programs.convolute_cubemap.bind(self.ppin),
            &mut render_cube,
        );
        let prefilter_map = create_prefiler_map(
            128,
            &self.programs.prefilter_cubemap.bind(self.ppin),
            &mut render_cube,
        );

        let mut brdf_lut_render_target = RenderTarget::new(512, 512);
        brdf_lut_render_target.set_viewport();
        render_cube(
            &brdf_lut_render_target.bind().clear(Mask::ColorDepth),
            &self.programs.brdf_lut.bind(self.ppin),
        );

        Ok(Ibl {
            cubemap,
            irradiance_map,
            prefilter_map,
            brdf_lut: brdf_lut_render_target.texture,
        })
    }
    // Proxies
    pub fn get_cube(&mut self) -> MeshRef {
        self.meshes.get_cube(self.vpin)
    }

    pub fn get_sphere(&mut self) -> MeshRef {
        self.meshes.get_sphere(self.vpin)
    }

    pub fn to_pipeline(self, w: u32, h: u32, hidpi_factor: f32) -> Pipeline {
        let default_material = Material::new()
            .albedo(v3(0.2, 0.2, 0.2))
            .metallic(0.0)
            .roughness(0.2);

        Pipeline::new(self.vpin, self.meshes, default_material, w, h, hidpi_factor)
    }
}
