use gl;
use std::{cell::Ref, path::Path};

use failure::Error;

use mg::{
    DrawMode, Framebuffer, FramebufferBinderDrawer, FramebufferBinderReadDraw, GlError, Mask,
    ProgramBind, ProgramBindingRef, ProgramPin, Texture, TextureSlot, VertexArray, VertexArrayPin,
    VertexBuffer,
};

use hot;
use warmy;

use misc::{v3, v4, Cacher, Mat4, V3, V4};

use render::{
    create_irradiance_map, create_prefiler_map, cubemap_from_equirectangular,
    lights::{DirectionalLight, PointLight, PointShadowMap, ShadowMap, SpotLight},
    store::{MeshRef, MeshStore}, Camera, GRenderPass, Ibl, Material, RenderObject,
    RenderObjectChild, RenderTarget, Renderable,
};

pub struct RenderProps<'a> {
    pub camera: &'a Camera,
    pub time: f32,

    pub directional_lights: &'a mut [DirectionalLight],
    pub point_lights: &'a mut [PointLight],
    pub spot_lights: &'a mut [SpotLight],

    pub default_material: Option<Material>,

    pub ibl: &'a Ibl,

    pub skybox_intensity: Option<f32>,
    pub ambient_intensity: Option<f32>,
}

pub struct HotShader(warmy::Res<hot::MyShader>);

impl HotShader {
    pub fn bind<'a>(&'a self, pin: &'a mut ProgramPin) -> ProgramBindingRef<'a> {
        ProgramBindingRef::new(Ref::map(self.0.borrow(), |a| &a.program), pin)
    }
}

type DrawCalls = Cacher<(MeshRef, Option<Material>), Vec<Mat4>>;

pub struct Pipeline {
    pub warmy_store: warmy::Store<()>,

    pub pbr_program: HotShader,
    pub directional_shadow_program: HotShader,
    pub point_shadow_program: HotShader,

    pub lighting_pbr_program: HotShader,

    pub skybox_program: HotShader,
    pub blur_program: HotShader,
    pub hdr_program: HotShader,
    pub screen_program: HotShader,

    pub default_material: Material,

    pub vbo_cache: Vec<VertexBuffer<Mat4>>,
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
    pub fn new(
        vpin: &mut VertexArrayPin,
        meshes: MeshStore,
        default_material: Material,
        w: u32,
        h: u32,
        hidpi_factor: f32,
    ) -> Pipeline {
        unsafe {
            gl::Enable(gl::DEPTH_TEST);
            gl::Enable(gl::CULL_FACE);
            gl::Enable(gl::FRAMEBUFFER_SRGB);
        }

        let mut warmy_store = warmy::Store::new(warmy::StoreOpt::default()).unwrap();

        let ctx = &mut ();

        macro_rules! shader {
            (x, $vert:expr, $geom:expr, $frag:expr) => {{
                let vert = concat!("shaders/", $vert).into();
                let geom = $geom.map(|x: &str| x.into());
                let frag = concat!("shaders/", $frag).into();
                let src = hot::ShaderSrc { vert, geom, frag };
                let src: warmy::LogicalKey = src.into();
                let program: warmy::Res<hot::MyShader> = warmy_store.get(&src, ctx).unwrap();
                HotShader(program)
            }};
            ($vert:expr, $geom:expr, $frag:expr) => {
                shader!(x, $vert, Some(concat!("shaders/", $geom)), $frag)
            };
            (rect, $frag:expr) => {
                shader!("rect.vert", "rect.geom", $frag)
            };
            ($vert:expr, $frag:expr) => {
                shader!(x, $vert, None, $frag)
            };
        }

        let pbr_program = shader!("shader.vert", "shader_pbr.frag");
        let skybox_program = shader!("skybox.vert", "skybox.frag");
        let blur_program = shader!(rect, "blur.frag");
        let directional_shadow_program = shader!("shadow.vert", "shadow.frag");
        let point_shadow_program = shader!(
            "point_shadow.vert",
            "point_shadow.geom",
            "point_shadow.frag"
        );
        let lighting_pbr_program = shader!(rect, "lighting_pbr.frag");
        let hdr_program = shader!(rect, "hdr.frag");
        let screen_program = shader!(rect, "screen.frag");

        let rect = Pipeline::make_rect(vpin);

        let screen_target = RenderTarget::new(w, h);
        let window_fbo = unsafe { Framebuffer::window() };

        let g = GRenderPass::new(w, h);

        let lighting_target = RenderTarget::new(w, h);

        let blur_targets = Pipeline::generate_blur_targets(w, h, 6);

        Pipeline {
            warmy_store,

            pbr_program,
            directional_shadow_program,
            point_shadow_program,

            lighting_pbr_program,

            skybox_program,
            blur_program,
            hdr_program,
            screen_program,

            default_material,

            rect,

            meshes,

            vbo_cache: vec![],
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
    fn make_rect(vpin: &mut VertexArrayPin) -> VertexArray {
        let rect_vao = VertexArray::new();
        let mut rect_vbo = VertexBuffer::from_data(&[v3(0.0, 0.0, 0.0)]);
        rect_vao.bind(vpin).vbo_attrib(&rect_vbo.bind(), 0, 3, 0);
        rect_vao
    }
    fn render_object(
        render_object: &RenderObject,
        transform: &Mat4,
        material: Option<&Material>,
        calls: &mut DrawCalls,
    ) {
        let transform = transform * render_object.transform;
        // if a material is passed as a prop, use that over the one bound to the mesh
        let material = material.or(render_object.material.as_ref());

        match &render_object.child {
            RenderObjectChild::Mesh(mesh_ref) => {
                calls.push_into((*mesh_ref, material.cloned()), transform);
            }
            RenderObjectChild::RenderObjects(render_objects) => {
                for obj in render_objects.iter() {
                    Pipeline::render_object(obj, &transform, material, calls)
                }
            }
        }
    }
    // fn render_scene<'a, T, F, P>(
    //     &mut self,
    //     scene_draw_calls_meshes: T,
    //     default_material: Material,
    //     fbo: &F,
    //     program: &P,
    // ) where
    //     T: Iterator<Item = &'a (&'a MeshRef, &'a Option<Material>, usize)>,
    //     F: FramebufferBinderDrawer,
    //     P: ProgramBind,
    // {
    //     for (mesh_ref, material, transforms_i) in scene_draw_calls_meshes {
    //         let start_slot = program.next_texture_slot();
    //         if let Some(material) = material {
    //             material.bind(&self.meshes, program);
    //         } else {
    //             default_material.bind(&self.meshes, program);
    //         }
    //         let mesh = self.meshes.get_mesh_mut(&mesh_ref);
    //         let vbo = &mut self.vbo_cache[*transforms_i];
    //         mesh.bind().draw_instanced(fbo, program, &vbo.bind());
    //         program.set_next_texture_slot(start_slot);
    //     }
    // }
    pub fn render<'b, T>(
        &mut self,
        ppin: &mut ProgramPin,
        vpin: &mut VertexArrayPin,
        update_shadows: bool,
        props: RenderProps,
        render_objects: T,
    ) where
        T: Iterator<Item = &'b RenderObject>,
    {
        let view = props.camera.get_view();
        let view_pos = props.camera.pos;
        let projection = props.camera.get_projection();

        let mut scene_draw_calls_meshes: Vec<(&MeshRef, Option<&Material>, usize)> = {
            self.draw_calls.clear();
            for obj in render_objects {
                Pipeline::render_object(obj, &Mat4::from_scale(1.0), None, &mut self.draw_calls);
            }

            self.vbo_cache.reserve(self.draw_calls.len());

            let mut scene_draw_calls_meshes = Vec::with_capacity(self.draw_calls.len());

            for (i, ((mesh_ref, material), transforms)) in self.draw_calls.iter().enumerate() {
                if i >= self.vbo_cache.len() {
                    let vbo = VertexBuffer::new();
                    self.vbo_cache.push(vbo);
                }

                let vbo = &mut self.vbo_cache[i];
                vbo.bind().buffer_data(&transforms);

                scene_draw_calls_meshes.push((mesh_ref, material.as_ref(), i));
            }

            scene_draw_calls_meshes
        };

        let default_material = props
            .default_material
            .as_ref()
            .unwrap_or(&self.default_material);

        macro_rules! render_scene {
            ($fbo:expr, $program:expr) => {
                render_scene!($fbo, $program, draw_instanced)
            };
            ($fbo:expr, $program:expr, $func:ident) => {{
                for (mesh_ref, material, transforms_i) in &mut scene_draw_calls_meshes {
                    let start_slot = $program.next_texture_slot();
                    if let Some(material) = material {
                        material.bind(&self.meshes, &$program);
                    } else {
                        default_material.bind(&self.meshes, &$program);
                    }
                    let mesh = self.meshes.get_mesh(&mesh_ref);
                    let vbo = &mut self.vbo_cache[*transforms_i];
                    mesh.bind(vpin)
                        .draw_instanced(&$fbo, &$program, &vbo.bind());
                    $program.set_next_texture_slot(start_slot);
                }
            }};
        };

        macro_rules! draw_rect {
            ($fbo:expr, $program:expr) => {
                self.rect
                    .bind(vpin)
                    .draw_arrays($fbo, $program, DrawMode::Points, 0, 1);
            };
        }

        self.screen_target.set_viewport();

        {
            // Render geometry
            let fbo = self.g.fbo.bind();
            fbo.clear(Mask::ColorDepth);
            let program = self.pbr_program.bind(ppin);
            program
                .bind_mat4("projection", projection)
                .bind_mat4("view", view)
                .bind_vec3("viewPos", view_pos)
                .bind_float("time", props.time);

            default_material.bind(&self.meshes, &program);

            render_scene!(fbo, program);
        }

        {
            {
                // Render depth map for directional lights
                let p = self.directional_shadow_program.bind(ppin);
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
                let p = self.point_shadow_program.bind(ppin);
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

            let g = self.lighting_pbr_program.bind(ppin);

            g.bind_texture("aPosition", &self.g.position)
                .bind_texture("aNormal", &self.g.normal)
                .bind_texture("aAlbedo", &self.g.albedo)
                .bind_texture("aAlbedo", &self.g.albedo)
                .bind_texture("aMrao", &self.g.mrao)
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
            let program = self.skybox_program.bind(ppin);
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
            let cube_ref = self.meshes.get_cube(vpin);
            self.meshes
                .get_mesh(&cube_ref)
                .bind(vpin)
                .draw(&fbo, &program);
            unsafe {
                gl::DepthFunc(gl::LESS);
                gl::Enable(gl::CULL_FACE);
            }
        }

        // Blur passes
        if false {
            let passes = self.blur_targets.iter_mut();
            let size = self.lighting_target.width as f32;
            let mut prev = &self.lighting_target;

            for next in passes {
                next.set_viewport();
                let scale = size / next.width as f32;
                {
                    let fbo = next.bind();
                    fbo.clear(Mask::ColorDepth);

                    let blur = self.blur_program.bind(ppin);
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

            let hdr = self.hdr_program.bind(ppin);
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

            let screen = self.screen_program.bind(ppin);
            screen.bind_texture("display", &self.screen_target.texture);

            draw_rect!(&fbo, &screen);
        }

        self.warmy_store.sync(&mut ());
    }
}
