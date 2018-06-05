use collada;
use failure::{Error, ResultExt};
use gl;
use mg::{
    DrawMode, FramebufferBinderDrawer, FramebufferBinderReadDraw, Mask, Program, ProgramBind,
    ProgramBinding, ProgramPin, TextureSlot, VertexArray, VertexArrayPin, VertexBuffer,
};
use misc::{v3, v4, Cacher, Mat4, V3, Vertex};
use pipeline::Pipeline;
use render::{
    create_irradiance_map, create_prefiler_map, cubemap_from_equirectangular,
    mesh::calculate_tangent_and_bitangent, mesh::Mesh, store::{MeshRef, MeshStore, TextureRef},
    Ibl, Material, MaterialProp, RenderObject, RenderObjectChild, RenderTarget,
};
use std;
use std::{cell::RefCell, collections::HashMap, fs, path::Path};

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
                    .map(|x| self.meshes.load_rgb(x))
                    .transpose()?
                    .map(|x| x.into())
                    .unwrap_or_else(|| normal3.into()),
            )
            .albedo::<MaterialProp<_>>(
                albedo_src
                    .map(|x| self.meshes.load_srgb(x))
                    .transpose()?
                    .map(|x| x.into())
                    .unwrap_or_else(|| white3.into()),
            )
            .metallic::<MaterialProp<_>>(
                metallic_src
                    .map(|x| self.meshes.load_rgb(x))
                    .transpose()?
                    .map(|x| x.into())
                    .unwrap_or_else(|| 1.0.into()),
            )
            .roughness::<MaterialProp<_>>(
                roughness_src
                    .map(|x| self.meshes.load_rgb(x))
                    .transpose()?
                    .map(|x| x.into())
                    .unwrap_or_else(|| 1.0.into()),
            )
            .ao::<MaterialProp<_>>(
                ao_src
                    .map(|x| self.meshes.load_rgb(x))
                    .transpose()?
                    .map(|x| x.into())
                    .unwrap_or_else(|| 1.0.into()),
            )
            .opacity::<MaterialProp<_>>(
                opacity_src
                    .map(|x| self.meshes.load_rgb(x))
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
                                //     self.meshes.convert_collada_material(&path, &data, material)?;
                                let mesh = Mesh::new(&verts, self.vpin);
                                let mesh_ref = self.meshes.insert_mesh(mesh);

                                meshes.push(mesh_ref);
                            }
                            mesh_ids.insert(geom_instance.geometry.0, meshes.clone());
                            meshes
                        }
                        _ => unimplemented!(),
                    }
                };

                // let material = geom_instance.material
                //     .map(|material_ref| self.meshes.convert_collada_material(&path, &data, material_ref));

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
    pub fn load_pbr_with_default_filenames(
        &mut self,
        path: impl AsRef<Path>,
        extension: &str,
    ) -> Result<Material, Error> {
        let path = path.as_ref();
        let x = |map| path.join(map).with_extension(extension);

        let builder = Material::new()
            .albedo(
                self.meshes
                    .load_srgb(x("albedo"))
                    .context("failed to load pbr albedo")?,
            )
            .metallic(
                self.meshes
                    .load_rgb(x("metallic"))
                    .context("failed to load pbr metallic")?,
            )
            .roughness(
                self.meshes
                    .load_rgb(x("roughness"))
                    .context("failed to load pbr roughness")?,
            )
            .normal(
                self.meshes
                    .load_rgb(x("normal"))
                    .context("failed to load pbr normal")?,
            )
            .ao(self
                .meshes
                .load_rgb(x("ao"))
                .context("failed to load pbr ao")?)
            .opacity(
                self.meshes
                    .load_rgb(x("opacity"))
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
        unsafe {
            gl::Viewport(
                0,
                0,
                brdf_lut_render_target.width as i32,
                brdf_lut_render_target.height as i32,
            );
        }
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
