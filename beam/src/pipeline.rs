use gl;
use std::{cell::RefMut, path::Path};

use failure::Error;

use mg::{
    DrawMode, Framebuffer, FramebufferBinderDrawer, FramebufferBinderReadDraw, GlError, Mask,
    ProgramBind, ProgramBindingRefMut, Texture, TextureSlot, VertexArray, VertexBuffer,
};

use hot;
use warmy;

use misc::{v3, v4, Cacher, Mat4, V3, V4};

use render::{
    create_irradiance_map, create_prefiler_map, cubemap_from_equirectangular,
    lights::{DirectionalLight, PointLight, SpotLight, PointShadowMap, ShadowMap}, Camera, GRenderPass,
    Material, MeshRef, MeshStore, RenderObject, RenderObjectChild, RenderTarget, Renderable,
};

pub struct RenderProps<'a> {
    pub camera: &'a Camera,
    pub time: f32,

    pub directional_lights: &'a mut [DirectionalLight],
    pub point_lights: &'a mut [PointLight],
    pub spot_lights: &'a mut [SpotLight],

    pub ibl: &'a Ibl,

    pub skybox_intensity: Option<f32>,
    pub ambient_intensity: Option<f32>,
}

pub struct HotShader(warmy::Res<hot::MyShader>);

impl HotShader {
    pub fn bind(&mut self) -> ProgramBindingRefMut {
        ProgramBindingRefMut::new(RefMut::map(self.0.borrow_mut(), |a| &mut a.program))
    }
}

#[derive(Debug)]
pub struct Ibl {
    pub cubemap: Texture,
    pub irradiance_map: Texture,
    pub prefilter_map: Texture,
    pub brdf_lut: Texture,
}

type DrawCalls = Cacher<(MeshRef, Option<Material>), Vec<Mat4>>;

pub struct Pipeline {
    pub warmy_store: warmy::Store<()>,

    pub equirectangular_program: HotShader,
    pub convolute_cubemap_program: HotShader,
    pub prefilter_cubemap_program: HotShader,
    pub brdf_lut_program: HotShader,

    pub pbr_program: HotShader,
    pub directional_shadow_program: HotShader,
    pub point_shadow_program: HotShader,

    pub lighting_pbr_program: HotShader,

    pub skybox_program: HotShader,
    pub blur_program: HotShader,
    pub hdr_program: HotShader,
    pub screen_program: HotShader,

    pub draw_calls: DrawCalls,
    pub meshes: MeshStore,

    pub rect: VertexArray,

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

        let equirectangular_program = shader!(
            "shaders/rect.vert",
            Some("shaders/cube.geom"),
            "shaders/equirectangular.frag",
        );
        let convolute_cubemap_program = shader!(
            "shaders/rect.vert",
            Some("shaders/cube.geom"),
            "shaders/convolute_cubemap.frag",
        );
        let prefilter_cubemap_program = shader!(
            "shaders/rect.vert",
            Some("shaders/cube.geom"),
            "shaders/prefilter_cubemap.frag",
        );

        let brdf_lut_program = shader!(
            "shaders/rect.vert",
            Some("shaders/rect.geom"),
            "shaders/brdf.frag",
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
            "shaders/lighting_pbr.frag",
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

        let meshes = MeshStore::default();

        let rect = {
            let mut rect_vao = VertexArray::new();
            let mut rect_vbo: VertexBuffer<V3> = VertexBuffer::from_data(&[v3(0.0, 0.0, 0.0)]);

            {
                let vbo = rect_vbo.bind();
                rect_vao.bind().vbo_attrib(&vbo, 0, 3, 0);
            }

            rect_vao
        };

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
            brdf_lut_program,

            pbr_program,
            directional_shadow_program,
            point_shadow_program,

            lighting_pbr_program,

            skybox_program,
            blur_program,
            hdr_program,
            screen_program,

            rect,

            meshes,
            draw_calls: DrawCalls::new(),

            hidpi_factor,

            g,

            lighting_target,

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
    pub fn load_ibl(&mut self, path: impl AsRef<Path>) -> Result<Ibl, Error> {
        let ibl_raw_ref = self.meshes.load_hdr(path)?;
        let ibl_raw = self.meshes.get_texture(&ibl_raw_ref);
        // Borrow checker work around here, we could use the rect vao already define on the pipeline
        // if it were not for the overship issues, but this works, and I don't think it is all that
        // costly.
        let mut rect = {
            let mut rect_vao = VertexArray::new();
            let mut rect_vbo = VertexBuffer::from_data(&[v3(0.0, 0.0, 0.0)]);
            rect_vao.bind().vbo_attrib(&rect_vbo.bind(), 0, 3, 0);
            rect_vao
        };

        let mut render_cube = |fbo: &FramebufferBinderReadDraw, program: &ProgramBindingRefMut| {
            program.bind_texture_to("equirectangularMap", &ibl_raw, TextureSlot::One);
            rect.bind()
                .draw_arrays(fbo, program, DrawMode::Points, 0, 1);
        };

        let cubemap = cubemap_from_equirectangular(
            512,
            &self.equirectangular_program.bind(),
            &mut render_cube,
        );
        cubemap.bind().generate_mipmap();

        let mut render_cube = |fbo: &FramebufferBinderReadDraw, program: &ProgramBindingRefMut| {
            program.bind_texture_to("equirectangularMap", &cubemap, TextureSlot::One);
            rect.bind()
                .draw_arrays(fbo, program, DrawMode::Points, 0, 1);
        };
        let irradiance_map =
            create_irradiance_map(32, &self.convolute_cubemap_program.bind(), &mut render_cube);
        let prefilter_map = create_prefiler_map(
            128,
            &self.prefilter_cubemap_program.bind(),
            &mut render_cube,
        );

        let mut brdf_lut_render_target = RenderTarget::new(512, 512);
        brdf_lut_render_target.set_viewport();
        render_cube(
            &brdf_lut_render_target.bind().clear(Mask::ColorDepth),
            &self.brdf_lut_program.bind(),
        );

        Ok(Ibl {
            cubemap,
            irradiance_map,
            prefilter_map,
            brdf_lut: brdf_lut_render_target.texture,
        })
    }
    fn render_object(
        render_object: &RenderObject,
        transform: &Mat4,
        material: Option<Material>,
        calls: &mut DrawCalls,
    ) {
        let transform = transform * render_object.transform;
        // if a material is passed as a prop, use that over the one bound to the mesh
        let material = material.or(render_object.material);

        match &render_object.child {
            RenderObjectChild::Mesh(mesh_ref) => {
                calls.push_into((*mesh_ref, material), transform);
            }
            RenderObjectChild::RenderObjects(render_objects) => {
                for obj in render_objects.iter() {
                    Pipeline::render_object(obj, &transform, material, calls)
                }
            }
        }
    }
    pub fn render<'a, T>(&mut self, update_shadows: bool, props: RenderProps, render_objects: T)
    where
        T: Iterator<Item = &'a RenderObject>,
    {
        let view = props.camera.get_view();
        let view_pos = props.camera.pos;
        let projection = props.camera.get_projection();

        let mut scene_draw_calls_meshes: Vec<_> = {
            self.draw_calls.clear();
            for obj in render_objects {
                Pipeline::render_object(obj, &Mat4::from_scale(1.0), None, &mut self.draw_calls);
            }

            self.draw_calls
                .iter()
                .map(|((mesh_ref, material), transforms)| {
                    (mesh_ref, material, VertexBuffer::from_data(&transforms))
                })
                .collect()
        };

        macro_rules! render_scene {
            ($fbo:expr, $program:expr) => {
                render_scene!($fbo, $program, draw_instanced)
            };
            ($fbo:expr, $program:expr, $func:ident) => {{
                for (mesh_ref, material, transforms) in &mut scene_draw_calls_meshes {
                    let start_slot = $program.next_texture_slot();
                    if let Some(material) = material {
                        material.bind(&self.meshes, &$program);
                    }
                    let mesh = self.meshes.get_mesh_mut(&mesh_ref);
                    mesh.bind()
                        .draw_instanced(&$fbo, &$program, &transforms.bind());
                    &$program.set_next_texture_slot(start_slot);
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

        self.screen_target.set_viewport();

        {
            // Render geometry
            let fbo = self.g.fbo.bind();
            fbo.clear(Mask::ColorDepth);
            let program = self.pbr_program.bind();
            program
                .bind_mat4("projection", projection)
                .bind_mat4("view", view)
                .bind_vec3("viewPos", view_pos)
                .bind_float("time", props.time);

            let white4 = self.meshes.rgba_texture(v4(1.0, 1.0, 1.0, 1.0));
            let white3 = self.meshes.rgb_texture(v3(1.0, 1.0, 1.0));
            let whiteish3 = self.meshes.rgb_texture(v3(0.2, 0.2, 0.2));
            let normal3 = self.meshes.rgb_texture(v3(0.5, 0.5, 1.0));
            let black3 = self.meshes.rgb_texture(v3(0.0, 0.0, 0.0));

            let default_material = Material {
                normal: normal3,
                albedo: whiteish3,
                metallic: black3,
                roughness: whiteish3,
                ao: white3,
                opacity: white3,
            };
            default_material.bind(&self.meshes, &program);

            render_scene!(fbo, program);
        }

        {
            {
                // Render depth map for directional lights
                let p = self.directional_shadow_program.bind();
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
                let p = self.point_shadow_program.bind();
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

            let g = self.lighting_pbr_program.bind();

            g.bind_texture("aPosition", &self.g.position)
                .bind_texture("aNormal", &self.g.normal)
                .bind_texture("aAlbedo", &self.g.albedo)
                .bind_texture("aAlbedo", &self.g.albedo)
                .bind_texture(
                    "aMetallicRoughnessAoOpacity",
                    &self.g.metallic_roughness_ao_opacity,
                )
                .bind_float(
                    "ambientIntensity",
                    props
                        .ambient_intensity
                        .unwrap_or_else(|| props.skybox_intensity.unwrap_or(0.0)),
                )
                .bind_texture("irradianceMap", &props.ibl.irradiance_map)
                .bind_texture("prefilterMap", &props.ibl.prefilter_map)
                .bind_texture("brdfLUT", &props.ibl.brdf_lut)
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
            SpotLight::bind_multiple(props.spot_lights, "spotLights", "nrSpotLights", &g);

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
            let program = self.skybox_program.bind();
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
                .bind_texture("skybox", &props.ibl.cubemap);
            let cube_ref = self.meshes.get_cube();
            self.meshes
                .get_mesh_mut(&cube_ref)
                .bind()
                .draw(&fbo, &program);
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

                    let blur = self.blur_program.bind();
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

            let hdr = self.hdr_program.bind();
            hdr.bind_texture("hdrBuffer", &self.lighting_target.texture)
                .bind_float("time", props.time)
                .bind_textures("blur", self.blur_targets.iter().map(|t| &t.texture));

            draw_rect!(&fbo, &hdr);
        }

        // HiDPI Pass
        {
            self.screen_target.set_viewport();
            let fbo = self.window_fbo.bind();
            fbo.clear(Mask::ColorDepth);

            let screen = self.screen_program.bind();
            screen.bind_texture("display", &self.screen_target.texture);

            draw_rect!(&fbo, &screen);
        }

        self.warmy_store.sync(&mut ());
    }
}
