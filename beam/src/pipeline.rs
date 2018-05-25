use cgmath::Rad;
use gl;
use std::{self, cell::RefMut, path::Path, rc::Rc};

use mg::*;

use hot;
use warmy;

use render::{
    convolute_cubemap, cubemap_from_equirectangular, cubemap_from_importance, v3, Camera,
    DirectionalLight, GRenderPass, Image, ImageKind, Mat4, Mesh, Model, PbrMaterial, PointLight,
    PointShadowMap, RenderObject, RenderObjectKind, RenderTarget, Renderable, ShadowMap,
    TextureCache, V3, V4, Vertex,
};

pub struct RenderProps<'a> {
    pub camera: &'a Camera,
    pub time: f32,

    pub directional_lights: &'a mut [DirectionalLight],
    pub point_lights: &'a mut [PointLight],

    pub skybox_intensity: Option<f32>,
    pub ambient_intensity: Option<f32>,
}

pub struct HotShader(warmy::Res<hot::MyShader>);

impl HotShader {
    pub fn bind<'a>(&'a mut self) -> RefMut<'a, Program> {
        RefMut::map(self.0.borrow_mut(), |a| &mut a.program)
    }
}

#[derive(Default)]
pub struct MeshStore {
    meshes: Vec<Mesh>,
}

pub struct MeshRef(usize);

impl MeshStore {
    pub fn load_collada(
        &mut self,
        texture_cache: &mut TextureCache,
        path: impl AsRef<Path>,
    ) -> Vec<MeshRef> {
        let vertex_groups = Model::new_vertex_data_from_disk_collada(texture_cache, path);

        vertex_groups
            .into_iter()
            .map(|(vertices, materials)| {
                let mesh = Mesh::new(&vertices, materials);
                let id = self.meshes.len();
                self.meshes.push(mesh);
                MeshRef(id)
            })
            .collect()
    }
}

pub struct Pipeline {
    pub warmy_store: warmy::Store<()>,

    pub equirectangular_program: HotShader,
    pub convolute_cubemap_program: HotShader,
    pub prefilter_cubemap_program: HotShader,

    pub pbr_program: HotShader,
    pub directional_shadow_program: HotShader,
    pub point_shadow_program: HotShader,

    pub lighting_pbr_program: HotShader,

    pub skybox_program: HotShader,
    pub blur_program: HotShader,
    pub hdr_program: HotShader,
    pub screen_program: HotShader,

    pub pbr_material: PbrMaterial,
    pub ibl_raw: Rc<Texture>,
    pub ibl_cubemap: Image,
    pub ibl_cubemap_convoluted: Image,
    pub ibl_prefiltered_cubemap: Image,
    pub brdf_lut: Texture,

    pub meshes: MeshStore,

    pub nanosuit: Model,
    pub cyborg: Model,
    pub wall: Model,
    pub cube: Mesh,
    pub rect: VertexArray,

    pub nanosuit_vbo: VertexBuffer<Mat4>,
    pub cyborg_vbo: VertexBuffer<Mat4>,
    pub cube_vbo: VertexBuffer<Mat4>,
    pub wall_vbo: VertexBuffer<Mat4>,

    pub hidpi_factor: f32,

    pub g: GRenderPass,

    pub lighting_target: RenderTarget,

    pub blur_targets: Vec<RenderTarget>,

    pub screen_target: RenderTarget,
    pub window_fbo: Framebuffer,
}

impl Pipeline {
    pub fn new(w: u32, h: u32, hidpi_factor: f32) -> Pipeline {
        unsafe {
            gl::Enable(gl::DEPTH_TEST);
            gl::Enable(gl::CULL_FACE);
            gl::Enable(gl::FRAMEBUFFER_SRGB);
        }

        let mut warmy_store = warmy::Store::new(warmy::StoreOpt::default()).unwrap();

        let ctx = &mut ();

        macro_rules! shader {
            ($vert:expr, $geom:expr, $frag:expr) => {{
                let vert = $vert;
                let geom = $geom;
                let frag = $frag;
                let src = hot::ShaderSrc { vert, geom, frag };
                let src: warmy::LogicalKey = src.into();
                let program: warmy::Res<hot::MyShader> = warmy_store.get(&src, ctx).unwrap();
                // Program::new_from_disk(vert, geom, frag)
                HotShader(program)
            }};
            ($vert:expr, $geom:expr, $frag:expr,) => {
                shader!($vert, $geom, $frag)
            };
        }

        let mut equirectangular_program = shader!(
            "shaders/rect.vert",
            Some("shaders/cube.geom"),
            "shaders/equirectangular.frag"
        );
        let mut convolute_cubemap_program = shader!(
            "shaders/rect.vert",
            Some("shaders/cube.geom"),
            "shaders/convolute_cubemap.frag"
        );
        let mut prefilter_cubemap_program = shader!(
            "shaders/rect.vert",
            Some("shaders/cube.geom"),
            "shaders/prefilter_cubemap.frag"
        );

        let mut brdf_lut_program = shader!(
            "shaders/rect.vert",
            Some("shaders/rect.geom"),
            "shaders/brdf.frag"
        );

        let pbr_program = shader!("shaders/shader.vert", None, "shaders/shader_pbr.frag");
        let skybox_program = shader!("shaders/skybox.vert", None, "shaders/skybox.frag");
        let blur_program = shader!(
            "shaders/rect.vert",
            Some("shaders/rect.geom"),
            "shaders/blur.frag",
        );
        let directional_shadow_program =
            shader!("shaders/shadow.vert", None, "shaders/shadow.frag");
        let point_shadow_program = shader!(
            "shaders/point_shadow.vert",
            Some("shaders/point_shadow.geom"),
            "shaders/point_shadow.frag",
        );
        let lighting_pbr_program = shader!(
            "shaders/rect.vert",
            Some("shaders/rect.geom"),
            "shaders/lighting_pbr.frag"
        );
        let hdr_program = shader!(
            "shaders/rect.vert",
            Some("shaders/rect.geom"),
            "shaders/hdr.frag",
        );
        let screen_program = shader!(
            "shaders/rect.vert",
            Some("shaders/rect.geom"),
            "shaders/screen.frag",
        );

        let mut texture_cache = TextureCache::new();
        let nanosuit = Model::new_from_disk(
            &mut texture_cache,
            "assets/nanosuit_reflection/nanosuit.obj",
        );
        // let nanosuit = Model::new_from_disk(&mut texture_cache, "assets/nice_thing/nice.obj");
        let cyborg = Model::new_from_disk(&mut texture_cache, "assets/cyborg/cyborg.obj");
        // let cyborg = Model::new_from_disk(&mut texture_cache, "assets/Cerberus_LP/Cerberus_LP.obj");
        // let wall = Model::new_from_disk(
        //     &mut texture_cache,
        //     "assets/castle_wall/Aset_castle_wall_M_scDxB_LOD4.obj",
        // );
        // let wall = Model::new_from_disk(&mut texture_cache, "assets/weew/weew.obj");
        let wall = Model::new_from_disk_collada(&mut texture_cache, "../collada/suzanne.dae");

        let tex1 = Image::new_from_disk(
            &mut texture_cache,
            "assets/container2.png",
            ImageKind::Diffuse,
        );
        let tex2 = Image::new_from_disk(
            &mut texture_cache,
            "assets/container2_specular.png",
            ImageKind::Specular,
        );

        // let pbr_material = PbrMaterial::new_with_default_filenames(
        //     &mut texture_cache,
        //     "assets/pbr/gold",
        //     "png",
        // );
        let pbr_material = PbrMaterial::new_with_default_filenames(
            &mut texture_cache,
            "assets/pbr/rusted_iron",
            "png",
        );
        // let pbr_material = PbrMaterial::new_with_default_filenames(
        //     &mut texture_cache,
        //     "assets/castle_wall/materials",
        //     "jpg",
        // );
        // let pbr_material = PbrMaterial::new_with_default_filenames(
        //     &mut texture_cache,
        //     "assets/pbr/stone_wall",
        //     "jpg",
        // );
        // let pbr_material = PbrMaterial::new_with_default_filenames(
        //     &mut texture_cache,
        //     "assets/pbr/metal_bare",
        //     "jpg",
        // );

        let ibl_raw = texture_cache.load_hdr("assets/Newport_Loft/Newport_Loft_Ref.hdr");
        // let ibl_raw = texture_cache.load_hdr("assets/Milkyway/Milkyway_small.hdr");
        let cube_mesh = Mesh::new(&cube_vertices(), vec![tex1, tex2]);
        let mut rect = {
            let mut rect_vao = VertexArray::new();
            let mut rect_vbo: VertexBuffer<V3> = VertexBuffer::new();

            {
                let mut vbo = rect_vbo.bind();
                vbo.buffer_data(&[v3(0.0, 0.0, 0.0)]);
                rect_vao.bind().vbo_attrib(&vbo, 0, 3, 0);
            }

            rect_vao
        };

        let (ibl_cubemap, ibl_cubemap_convoluted, ibl_prefiltered_cubemap, brdf_lut) = {
            let mut render_cube = |fbo: &FramebufferBinderReadDraw, program: &ProgramBinding| {
                program.bind_texture_to("equirectangularMap", &ibl_raw, TextureSlot::One);
                rect.bind()
                    .draw_arrays(fbo, program, DrawMode::Points, 0, 1);
            };

            let ibl_cubemap = cubemap_from_equirectangular(
                512,
                &equirectangular_program.bind().bind(),
                &mut render_cube,
            );

            let mut render_cube = |fbo: &FramebufferBinderReadDraw, program: &ProgramBinding| {
                program.bind_texture_to(
                    "equirectangularMap",
                    &ibl_cubemap.texture,
                    TextureSlot::One,
                );
                rect.bind()
                    .draw_arrays(fbo, program, DrawMode::Points, 0, 1);
            };
            let ibl_cubemap_convoluted = convolute_cubemap(
                32,
                &convolute_cubemap_program.bind().bind(),
                &mut render_cube,
            );
            let ibl_prefiltered_cubemap = cubemap_from_importance(
                128,
                &prefilter_cubemap_program.bind().bind(),
                &mut render_cube,
            );

            let mut brdf_lut_render_target = RenderTarget::new(512, 512);
            brdf_lut_render_target.set_viewport();
            render_cube(
                &brdf_lut_render_target.bind().clear(Mask::ColorDepth),
                &brdf_lut_program.bind().bind(),
            );

            (
                ibl_cubemap,
                ibl_cubemap_convoluted,
                ibl_prefiltered_cubemap,
                brdf_lut_render_target.texture,
            )
        };

        // let (a, b, c, d) = Vertex::soa(&rect_vertices());
        // let rect_mesh = Mesh::new(&a, &b, &c, &d, None, vec![]);

        let screen_target = RenderTarget::new(w, h);
        let window_fbo = unsafe { Framebuffer::window() };

        let g = GRenderPass::new(w, h);

        let lighting_target = RenderTarget::new(w, h);

        let blur_targets = Pipeline::generate_blur_targets(w, h, 6);

        Pipeline {
            warmy_store,

            equirectangular_program,
            convolute_cubemap_program,
            prefilter_cubemap_program,

            pbr_program,
            directional_shadow_program,
            point_shadow_program,

            lighting_pbr_program,

            skybox_program,
            blur_program,
            hdr_program,
            screen_program,

            pbr_material,
            ibl_raw,
            ibl_cubemap,
            ibl_cubemap_convoluted,
            ibl_prefiltered_cubemap,
            brdf_lut,

            nanosuit,
            cyborg,
            wall,
            cube: cube_mesh,
            rect: rect,

            meshes: MeshStore::default(),

            nanosuit_vbo: VertexBuffer::new(),
            cyborg_vbo: VertexBuffer::new(),
            cube_vbo: VertexBuffer::new(),
            wall_vbo: VertexBuffer::new(),

            hidpi_factor,

            g,

            lighting_target,

            // blur_1,
            // blur_2,
            blur_targets,

            screen_target,
            window_fbo,
        }
    }
    fn generate_blur_targets(w: u32, h: u32, n: usize) -> Vec<RenderTarget> {
        let mut blur_targets = Vec::with_capacity(n);

        for i in 0..n {
            let blur_scale = i as u32 + 2;
            let blur = RenderTarget::new(w / blur_scale, h / blur_scale);
            blur_targets.push(blur);
        }

        blur_targets
    }
    pub fn resize(&mut self, w: u32, h: u32) {
        let screen_target = RenderTarget::new(w, h);

        let g = GRenderPass::new(w, h);

        let lighting_target = RenderTarget::new(w, h);

        let blur_targets = Pipeline::generate_blur_targets(w, h, 6);

        self.screen_target = screen_target;
        self.g = g;
        self.lighting_target = lighting_target;
        self.blur_targets = blur_targets;
    }
    pub fn render<T>(&mut self, update_shadows: bool, props: RenderProps, render_objects: T)
    where
        T: Iterator<Item = RenderObject>,
    {
        let view = props.camera.get_view();
        let view_pos = props.camera.pos;
        let projection = props.camera.get_projection();

        let mut nanosuit_transforms = vec![];
        let mut cyborg_transforms = vec![];
        let mut wall_transforms = vec![];
        let mut cube_transforms = vec![];

        for obj in render_objects {
            match obj.kind {
                RenderObjectKind::Cube => cube_transforms.push(obj.transform),
                RenderObjectKind::Nanosuit => nanosuit_transforms.push(obj.transform),
                RenderObjectKind::Cyborg => cyborg_transforms.push(obj.transform),
                RenderObjectKind::Wall => wall_transforms.push(obj.transform),
            }
        }

        self.nanosuit_vbo.bind().buffer_data(&nanosuit_transforms);
        self.cyborg_vbo.bind().buffer_data(&cyborg_transforms);
        self.wall_vbo.bind().buffer_data(&wall_transforms);
        self.cube_vbo.bind().buffer_data(&cube_transforms);

        let pi = std::f32::consts::PI;

        macro_rules! render_scene {
            ($fbo:expr, $program:expr) => {
                render_scene!($fbo, $program, draw_instanced)
            };
            ($fbo:expr, $program:expr, $func:ident) => {{
                {
                    let model = Mat4::from_scale(1.0 / 4.0) * Mat4::from_translation(v3(0.0, 0.0, 4.0));
                    $program.bind_mat4("model", model);
                    self.nanosuit
                        .$func(&$fbo, &$program, &self.nanosuit_vbo.bind());
                }
                if false {
                    let model = Mat4::from_angle_y(Rad(pi));
                    $program.bind_mat4("model", model);
                    self.cyborg.$func(&$fbo, &$program, &self.cyborg_vbo.bind());
                }
                {
                    // let model = Mat4::from_scale(1.0 / 16.0)
                    let model = Mat4::from_scale(4.0 / 1.0)
                        * Mat4::from_translation(v3(0.0, 0.0, 4.0))
                        * Mat4::from_angle_y(Rad(pi + 0.01 * props.time));
                    $program.bind_mat4("model", model);
                    self.wall.$func(&$fbo, &$program, &self.wall_vbo.bind());
                }
                {
                    let model = Mat4::from_scale(1.0);
                    $program.bind_mat4("model", model);
                    self.cube
                        .bind()
                        .$func(&$fbo, &$program, &self.cube_vbo.bind());
                }
            }};
        };

        macro_rules! draw_rect {
            ($fbo:expr, $program:expr) => {
                self.rect
                    .bind()
                    .draw_arrays($fbo, $program, DrawMode::Points, 0, 1);
            };
        }

        // unsafe {
        //     gl::Viewport(0, 0, self.screen_width as i32, self.screen_height as i32);
        // }
        self.screen_target.set_viewport();

        {
            // Render geometry
            let fbo = self.g.fbo.bind();
            fbo.clear(Mask::ColorDepth);
            let mut program_ = self.pbr_program.bind();
            let program = program_.bind();
            program
                .bind_mat4("projection", projection)
                .bind_mat4("view", view)
                .bind_vec3("viewPos", view_pos)
                .bind_float("time", props.time)
                .bind_bool("useMaterial", false)
                .bind_vec3("mat_albedo", v3(0.5, 0.0, 0.0))
                .bind_vec3("mat_metallicRoughnessAo", v3(0.5, 0.5, 0.1));

            self.pbr_material.bind(&program);

            render_scene!(fbo, program);
            // self.render_scene(block, &program, fbo, &mut nanosuit_vbo, &mut cyborg_vbo, &mut cube_vbo);
        }

        if true {
            if true {
                // Render depth map for directional lights
                let mut p_ = self.directional_shadow_program.bind();
                let p = p_.bind();
                unsafe {
                    let (w, h) = ShadowMap::size();
                    gl::Viewport(0, 0, w as i32, h as i32);
                    gl::CullFace(gl::FRONT);
                }
                for light in props.directional_lights.iter_mut() {
                    let (fbo, light_space) = light.bind_shadow_map(props.camera.pos);
                    fbo.clear(Mask::Depth);
                    p.bind_mat4("lightSpace", light_space);
                    render_scene!(fbo, p, draw_geometry_instanced);
                }
                unsafe {
                    gl::CullFace(gl::BACK);
                    // gl::Viewport(0, 0, self.screen_width as i32, self.screen_height as i32);
                }
                self.screen_target.set_viewport();
            }
            if update_shadows {
                // Render depth map for point lights
                unsafe {
                    let (w, h) = PointShadowMap::size();
                    gl::Viewport(0, 0, w as i32, h as i32);
                    gl::CullFace(gl::FRONT);
                }
                let mut p_ = self.point_shadow_program.bind();
                let p = p_.bind();
                for light in props.point_lights.iter_mut() {
                    let far = if let Some(ref shadow_map) = light.shadow_map {
                        shadow_map.far
                    } else {
                        continue;
                    };
                    let position = light.position;
                    light.last_shadow_map_position = position;
                    let (fbo, light_spaces) = light.bind_shadow_map().unwrap();
                    fbo.clear(Mask::Depth);
                    p.bind_mat4s("shadowMatrices", &light_spaces)
                        .bind_vec3("lightPos", position)
                        .bind_float("farPlane", far);
                    render_scene!(fbo, p, draw_geometry_instanced);
                }
                unsafe {
                    gl::CullFace(gl::BACK);
                }
                self.screen_target.set_viewport();
            }
        }

        {
            // Render lighting
            let fbo = self.lighting_target.bind();
            fbo.clear(Mask::ColorDepth);

            let mut g_ = self.lighting_pbr_program.bind();
            let g = g_.bind();

            // macro_rules! uniforms {
            //     ($program:expr, { $($name:ident: $value:expr,)* }) => {
            //         $( $value.bind($program, $name); )*
            //     }
            // }

            // uniforms!(g, {
            //     aPosition: &self.g.position,
            //     aNormal: &self.g.normal,
            //     aAlbedoSpec: &self.g.albedo_spec,
            //     shadowMap: &self.skybox,
            //     skybox: &self.skybox,
            //     viewPos: view_pos,
            // });

            g.bind_texture("aPosition", &self.g.position)
                .bind_texture("aNormal", &self.g.normal)
                .bind_texture("aAlbedo", &self.g.albedo)
                .bind_texture("aAlbedo", &self.g.albedo)
                .bind_texture("aMetallicRoughnessAo", &self.g.metallic_roughness_ao)
                .bind_float(
                    "ambientIntensity",
                    props
                        .ambient_intensity
                        .unwrap_or_else(|| props.skybox_intensity.unwrap_or(0.0)),
                )
                .bind_texture("irradianceMap", &self.ibl_cubemap_convoluted.texture)
                .bind_texture("prefilterMap", &self.ibl_prefiltered_cubemap.texture)
                .bind_texture("brdfLUT", &self.brdf_lut)
                .bind_vec3("viewPos", view_pos);

            let lights: &[_] = &props.directional_lights;

            DirectionalLight::bind_multiple(
                props.camera.pos,
                lights,
                "directionalLights",
                "nrDirLights",
                &g,
            );

            PointLight::bind_multiple(props.point_lights, "pointLights", "nrPointLights", &g);

            draw_rect!(&fbo, &g);
            GlError::check().expect("Lighting pass failed");
        }

        // Skybox Pass
        {
            // Copy z-buffer over from geometry pass
            self.g
                .blit_to(&mut self.lighting_target, Mask::Depth, gl::NEAREST)
        }
        GlError::check().unwrap();
        {
            // Render skybox
            let fbo = self.lighting_target.bind();
            unsafe {
                gl::DepthFunc(gl::LEQUAL);
                gl::Disable(gl::CULL_FACE);
            }
            let mut program_ = self.skybox_program.bind();
            let program = program_.bind();
            let mut view = view.clone();
            view.w = V4::new(0.0, 0.0, 0.0, 0.0);
            program
                .bind_mat4("projection", projection)
                .bind_mat4("view", view)
                .bind_float(
                    "skyboxIntensity",
                    props
                        .skybox_intensity
                        .unwrap_or_else(|| props.ambient_intensity.unwrap_or(0.0)),
                )
                .bind_texture("skybox", &self.ibl_cubemap.texture);
            self.cube.bind().draw(&fbo, &program);
            unsafe {
                gl::DepthFunc(gl::LESS);
                gl::Enable(gl::CULL_FACE);
            }
        }

        // Blur passes
        {
            let passes = self.blur_targets.iter_mut();
            let size = self.lighting_target.width as f32;
            let mut prev = &self.lighting_target;

            for next in passes {
                next.set_viewport();
                let scale = size / next.width as f32;
                {
                    let fbo = next.bind();
                    fbo.clear(Mask::ColorDepth);

                    let mut blur_ = self.blur_program.bind();
                    let blur = blur_.bind();
                    blur.bind_texture("tex", &prev.texture)
                        .bind_float("scale", scale);
                    draw_rect!(&fbo, &blur);
                }

                prev = next;
            }
            self.screen_target.set_viewport();
        }

        // HDR/Screen Pass
        {
            let fbo = self.screen_target.bind();
            fbo.clear(Mask::ColorDepth);

            let mut hdr_ = self.hdr_program.bind();
            let hdr = hdr_.bind();
            hdr.bind_texture("hdrBuffer", &self.lighting_target.texture)
                .bind_float("time", props.time);

            hdr.bind_textures("blur", self.blur_targets.iter().map(|t| &t.texture));

            draw_rect!(&fbo, &hdr);
        }

        // HiDPI Pass
        {
            self.screen_target.set_viewport();
            let fbo = self.window_fbo.bind();
            fbo.clear(Mask::ColorDepth);

            let mut screen_ = self.screen_program.bind();
            let screen = screen_.bind();
            screen.bind_texture("display", &self.screen_target.texture);

            draw_rect!(&fbo, &screen);
        }

        self.warmy_store.sync(&mut ());
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
