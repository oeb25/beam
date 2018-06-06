use gl;
use std::{self, cell::Ref, path::Path};

use failure::Error;

use mg::{
    DrawMode, Framebuffer, FramebufferBinderDrawer, FramebufferBinderReadDraw, GlError, Mask,
    Program, ProgramBind, ProgramBindingRef, ProgramPin, Texture, TextureSlot, UniformLocation,
    VertexArray, VertexArrayPin, VertexBuffer,
};

use hot;
use warmy;

use misc::{v3, v4, Cacher, Mat4, V3, V4};

use render::{
    create_irradiance_map, create_prefiler_map, cubemap_from_equirectangular,
    dsl::{DrawCall, FramebufferCall, GlCall, Mat4g, ProgramCall, ProgramLike, UniformValue},
    lights::{DirectionalLight, PointLight, PointShadowMap, ShadowMap, SpotLight}, mesh::Mesh,
    store::{MeshRef, MeshStore}, Camera, GRenderPass, Ibl, Material, RenderObject,
    RenderObjectChild, RenderTarget, Renderable,
};

pub struct RenderProps<'a> {
    pub camera: &'a Camera,
    pub time: f32,

    pub directional_lights: &'a mut [DirectionalLight],
    pub point_lights: &'a mut [PointLight],
    pub spot_lights: &'a mut [SpotLight],

    pub cube_mesh: &'a Mesh,

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

impl std::cmp::PartialEq for HotShader {
    fn eq(&self, rhs: &HotShader) -> bool {
        (*self.0.borrow()) == (*rhs.0.borrow())
    }
}

impl std::fmt::Debug for HotShader {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(fmt, "{:?}", self.0.borrow())
    }
}

impl ProgramLike for HotShader {
    fn id(&self) -> u32 {
        self.0.borrow().program.id
    }
    fn get_uniform_location(&self, name: &str) -> UniformLocation {
        self.0.borrow().program.get_uniform_location(name)
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

    pub vbo_cache: Vec<VertexBuffer<Mat4g>>,
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
            (x, $name:expr, $vert:expr, $geom:expr, $frag:expr) => {{
                let vert = concat!("shaders/", $vert).into();
                let geom = $geom.map(|x: &str| x.into());
                let frag = concat!("shaders/", $frag).into();
                let src = hot::ShaderSrc {
                    name: $name.to_owned(),
                    vert,
                    geom,
                    frag,
                };
                let src: warmy::LogicalKey = src.into();
                let program: warmy::Res<hot::MyShader> = warmy_store.get(&src, ctx).unwrap();
                HotShader(program)
            }};
            ($name:expr, $vert:expr, $geom:expr, $frag:expr) => {
                shader!(x, $name, $vert, Some(concat!("shaders/", $geom)), $frag)
            };
            ($name:expr,rect, $frag:expr) => {{
                shader!($name, "rect.vert", "rect.geom", $frag)
            }};
            ($name:expr, $vert:expr, $frag:expr) => {
                shader!(x, $name, $vert, None, $frag)
            };
        }

        let pbr_program = shader!("shader pbr", "shader.vert", "shader_pbr.frag");
        let skybox_program = shader!("skybox", "skybox.vert", "skybox.frag");
        let blur_program = shader!("blur", rect, "blur.frag");
        let directional_shadow_program = shader!("shadow", "shadow.vert", "shadow.frag");
        let point_shadow_program = shader!(
            "point shadow",
            "point_shadow.vert",
            "point_shadow.geom",
            "point_shadow.frag"
        );
        let lighting_pbr_program = shader!("lighting pbr", rect, "lighting_pbr.frag");
        let hdr_program = shader!("hdr", rect, "hdr.frag");
        let screen_program = shader!("screen", rect, "screen.frag");

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
    pub fn new_render<'b, 'a: 'b, T>(
        &'a mut self,
        update_shadows: bool,
        props: RenderProps<'a>,
        render_objects: T,
    ) -> Vec<GlCall<'a, HotShader>>
    where
        T: Iterator<Item = &'b RenderObject>,
    {
        let mut calls = Vec::with_capacity(451);

        macro_rules! marker {
            ($m:expr) => {
                calls.push(GlCall::Marker($m))
            };
        }

        macro_rules! uniforms {
            ($($name:ident: $value:expr,)*) => {
                vec![
                    $((std::borrow::Cow::from(stringify!($name)), $value.into()),)*
                ]
            }
        }

        marker!("start");

        calls.push(GlCall::SaveTextureSlot);

        let view = props.camera.get_view();
        let view_pos = props.camera.pos;
        let projection = props.camera.get_projection();

        marker!("prepare scene_draw_calls");
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
        marker!("more prep");

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
                    calls.push(GlCall::SaveTextureSlot);
                    let uniforms = if let Some(material) = material {
                        material.bind_new(&self.meshes)
                    } else {
                        default_material.bind_new(&self.meshes)
                    };
                    calls.push(GlCall::Program(&$program, ProgramCall::Uniforms(uniforms)));
                    let mesh = self.meshes.get_mesh(&mesh_ref);
                    let vbo = &self.vbo_cache[*transforms_i];
                    calls.push(GlCall::Draw(&$program, &$fbo, mesh.draw_instanced_new(vbo)));
                    calls.push(GlCall::RestoreTextureSlot);
                }
            }};
        };

        macro_rules! draw_rect {
            ($program:expr, $fbo:expr) => {
                let d = DrawCall::Arrays(&self.rect, DrawMode::Points, 0, 1);
                calls.push(GlCall::Draw($program, $fbo, d));
            };
        }

        calls.push(self.screen_target.set_viewport());

        {
            marker!("render geometry to g buffer");
            // Render geometry
            calls.push(GlCall::Framebuffer(
                &self.g.fbo,
                FramebufferCall::Clear(Mask::ColorDepth),
            ));

            let mut uniforms = uniforms!(
                projection: projection,
                view: view,
                viewPos: view_pos,
                time: props.time,
            );

            uniforms.append(&mut default_material.bind_new(&self.meshes));

            calls.push(GlCall::Program(
                &self.pbr_program,
                ProgramCall::Uniforms(uniforms),
            ));

            render_scene!(&self.g.fbo, &self.pbr_program);
        }

        {
            {
                marker!("shadow directional lights");
                // Render depth map for directional lights
                let (w, h) = ShadowMap::size();
                calls.push(GlCall::Viewport(0, 0, w, h));
                calls.push(GlCall::CullFace(gl::FRONT));

                for light in props.directional_lights.iter() {
                    calls.push(GlCall::SaveTextureSlot);
                    let light_space = light.space(props.camera.pos);
                    calls.push(GlCall::Framebuffer(
                        &light.shadow_map.fbo,
                        FramebufferCall::Clear(Mask::ColorDepth),
                    ));
                    calls.push(GlCall::Program(
                        &self.directional_shadow_program,
                        uniforms!(lightSpace: light_space,).into(),
                    ));
                    render_scene!(
                        light.shadow_map.fbo,
                        &self.directional_shadow_program,
                        draw_geometry_instanced
                    );
                    calls.push(GlCall::RestoreTextureSlot);
                }

                calls.push(GlCall::CullFace(gl::BACK));

                calls.push(self.screen_target.set_viewport());
            }
            marker!("shadow point lights");
            if update_shadows {
                // Render depth map for point lights
                let (w, h) = PointShadowMap::size();
                calls.push(GlCall::Viewport(0, 0, w, h));
                calls.push(GlCall::CullFace(gl::FRONT));

                for light in props.point_lights.iter_mut() {
                    if light.shadow_map.is_some() {
                        let position = light.position;
                        light.last_shadow_map_position = position;
                    }
                }
                for light in props.point_lights.iter() {
                    let (far, fbo) = if let Some(ref shadow_map) = light.shadow_map {
                        (shadow_map.far, &shadow_map.fbo)
                    } else {
                        continue;
                    };
                    let position = light.position;
                    let light_spaces = light.space().unwrap();
                    calls.push(GlCall::Framebuffer(
                        &fbo,
                        FramebufferCall::Clear(Mask::ColorDepth),
                    ));
                    let uniforms = uniforms!(
                        shadowMatrices: light_spaces.to_vec(),
                        lightPos: position,
                        farPlane: far,
                    );
                    calls.push(GlCall::Program(&self.point_shadow_program, uniforms.into()));
                    render_scene!(fbo, &self.point_shadow_program, draw_geometry_instanced);
                }

                calls.push(GlCall::CullFace(gl::BACK));

                calls.push(self.screen_target.set_viewport());
            }
        }

        {
            marker!("render lighting");
            // Render lighting
            calls.push(GlCall::SaveTextureSlot);
            calls.push(GlCall::Framebuffer(
                &self.lighting_target.framebuffer,
                FramebufferCall::Clear(Mask::ColorDepth),
            ));

            let mut uniforms: Vec<(std::borrow::Cow<_>, _)> = uniforms!(
                aPosition: &self.g.position,
                aNormal: &self.g.normal,
                aAlbedo: &self.g.albedo,
                aEmission: &self.g.emission,
                aMrao: &self.g.mrao,
                ambientIntensity:
                    props
                        .ambient_intensity
                        .or(props.skybox_intensity)
                        .unwrap_or(0.0),
                irradianceMap: &props.ibl.irradiance_map,
                prefilterMap: &props.ibl.prefilter_map,
                brdfLUT: &props.ibl.brdf_lut,
                viewPos: view_pos,
            );

            let mut a = DirectionalLight::bind_multiple_new(
                props.camera.pos,
                props.directional_lights,
                "directionalLights",
                "nrDirLights",
            );

            let mut b =
                PointLight::bind_multiple_new(props.point_lights, "pointLights", "nrPointLights");
            let mut c =
                SpotLight::bind_multiple_new(props.spot_lights, "spotLights", "nrSpotLights");

            uniforms.append(&mut a);
            uniforms.append(&mut b);
            uniforms.append(&mut c);

            calls.push(GlCall::Program(&self.lighting_pbr_program, uniforms.into()));

            draw_rect!(
                &self.lighting_pbr_program,
                &self.lighting_target.framebuffer
            );
            calls.push(GlCall::RestoreTextureSlot);
        }

        // Skybox Pass
        {
            marker!("skybox");
            // Copy z-buffer over from geometry pass
            calls.push(GlCall::Framebuffer(
                &self.g.fbo,
                FramebufferCall::BlitTo(
                    &self.lighting_target.framebuffer,
                    (0, 0, self.g.width, self.g.height),
                    (
                        0,
                        0,
                        self.lighting_target.width,
                        self.lighting_target.height,
                    ),
                    Mask::Depth,
                    gl::NEAREST,
                ),
            ));
        }

        {
            calls.push(GlCall::SaveTextureSlot);
            // Render skybox
            calls.push(GlCall::DepthFunc(gl::LEQUAL));
            calls.push(GlCall::Disable(gl::CULL_FACE));

            let mut view = view.clone();
            view.w = V4::new(0.0, 0.0, 0.0, 0.0);
            let uniforms = uniforms!(
                projection: projection,
                view: view,
                skyboxIntensity:
                    props
                        .skybox_intensity
                        .or(props.ambient_intensity)
                        .unwrap_or(0.0),
                skybox: &props.ibl.cubemap,
            );
            calls.push(GlCall::Program(&self.skybox_program, uniforms.into()));

            calls.push(GlCall::Draw(
                &self.skybox_program,
                &self.lighting_target.framebuffer,
                props.cube_mesh.draw_new(),
            ));

            calls.push(GlCall::DepthFunc(gl::LESS));
            calls.push(GlCall::Enable(gl::CULL_FACE));
            calls.push(GlCall::RestoreTextureSlot);
        }

        // Blur passes
        if false {
            marker!("blur");
            calls.push(GlCall::SaveTextureSlot);
            let passes = self.blur_targets.iter();
            let size = self.lighting_target.width as f32;
            let mut prev = &self.lighting_target;

            for next in passes {
                calls.push(next.set_viewport());
                let scale = size / next.width as f32;
                {
                    calls.push(GlCall::Framebuffer(
                        &next.framebuffer,
                        FramebufferCall::Clear(Mask::ColorDepth),
                    ));

                    let uniforms = uniforms!(
                        tex: &prev.texture,
                        scale: scale,
                    );
                    calls.push(GlCall::Program(&self.blur_program, uniforms.into()));

                    draw_rect!(&self.blur_program, &next.framebuffer);
                }

                prev = next;
            }
            calls.push(self.screen_target.set_viewport());
            calls.push(GlCall::RestoreTextureSlot);
        }

        // HDR/Screen Pass
        {
            marker!("hdr");
            calls.push(GlCall::SaveTextureSlot);
            calls.push(GlCall::Framebuffer(
                &self.screen_target.framebuffer,
                FramebufferCall::Clear(Mask::ColorDepth),
            ));

            let uniforms = uniforms!(
                hdrBuffer: &self.lighting_target.texture,
                time: props.time,
                blur: UniformValue::Textures(
                    self.blur_targets
                        .iter()
                        .map(|t| &t.texture)
                        .collect::<Vec<_>>(),
                ),
            );
            calls.push(GlCall::Program(&self.hdr_program, uniforms.into()));

            draw_rect!(&self.hdr_program, &self.screen_target.framebuffer);
            calls.push(GlCall::RestoreTextureSlot);
        }

        // HiDPI Pass
        {
            marker!("screen");
            calls.push(GlCall::SaveTextureSlot);
            calls.push(self.screen_target.set_viewport());

            calls.push(GlCall::Framebuffer(
                &self.window_fbo,
                FramebufferCall::Clear(Mask::ColorDepth),
            ));

            let uniforms = uniforms!(
                display: &self.screen_target.texture,
            );
            calls.push(GlCall::Program(&self.screen_program, uniforms.into()));

            draw_rect!(&self.screen_program, &self.window_fbo);
            calls.push(GlCall::RestoreTextureSlot);
        }

        self.warmy_store.sync(&mut ());

        calls.push(GlCall::RestoreTextureSlot);

        marker!("done!");

        println!("number of calls: {:?}", calls.len());

        calls
    }
}
