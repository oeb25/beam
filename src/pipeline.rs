use cgmath::Rad;
use gl;
use std::{self, rc::Rc, cell::RefMut};

use mg::*;

use hot;
use warmy;

use render::{
    v3, Camera, CubeMapBuilder, DirectionalLight, GRenderPass, Image, ImageKind, Mat4, Mesh, Model,
    Object, ObjectKind, PointLight, PointShadowMap, RenderTarget, ShadowMap, TextureCache, V3, V4,
    Vertex,
};

pub struct RenderProps<'a> {
    pub objects: &'a [Object],
    pub camera: &'a Camera,
    pub time: f32,

    pub directional_lights: &'a mut [DirectionalLight],
    pub point_lights: &'a mut [PointLight],
}

pub struct HotShader(warmy::Res<hot::MyShader>);

impl HotShader {
    pub fn bind<'a>(&'a mut self) -> RefMut<'a, Program> {
        RefMut::map(self.0.borrow_mut(), |a| &mut a.program)
    }
}

pub struct Pipeline {
    pub warmy_store: warmy::Store<()>,

    pub vertex_program: HotShader,
    pub directional_shadow_program: HotShader,
    pub point_shadow_program: HotShader,

    pub directional_lighting_program: HotShader,
    pub point_lighting_program: HotShader,

    pub lighting_program: HotShader,

    pub skybox_program: HotShader,
    pub blur_program: HotShader,
    pub hdr_program: HotShader,
    pub screen_program: HotShader,

    pub nanosuit: Model<Vertex>,
    pub cyborg: Model<Vertex>,
    pub cube: Mesh<Vertex>,
    pub rect: VertexArray,

    pub nanosuit_vbo: VertexBuffer<Mat4>,
    pub cyborg_vbo: VertexBuffer<Mat4>,
    pub cube_vbo: VertexBuffer<Mat4>,

    pub screen_width: u32,
    pub screen_height: u32,
    pub hidpi_factor: f32,

    pub g: GRenderPass,

    pub color_a: RenderTarget,
    pub color_b: RenderTarget,

    // pub blur_1: RenderTarget,
    // pub blur_2: RenderTarget,
    pub blur_targets: Vec<RenderTarget>,

    pub final_target: RenderTarget,
    pub window_fbo: Framebuffer,

    pub skybox: Rc<Texture>,
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

        let vertex_program = shader!("shaders/shader.vert", None, "shaders/shader.frag");
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
        let directional_lighting_program = shader!(
            "shaders/rect.vert",
            Some("shaders/rect.geom"),
            "shaders/directional_lighting.frag",
        );
        let point_lighting_program = shader!(
            "shaders/rect.vert",
            Some("shaders/rect.geom"),
            "shaders/point_lighting.frag"
        );
        let lighting_program = shader!(
            "shaders/rect.vert",
            Some("shaders/rect.geom"),
            "shaders/lighting.frag"
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

        // let skybox = CubeMapBuilder {
        //     back: "assets/skybox/back.jpg",
        //     front: "assets/skybox/front.jpg",
        //     right: "assets/skybox/right.jpg",
        //     bottom: "assets/skybox/bottom.jpg",
        //     left: "assets/skybox/left.jpg",
        //     top: "assets/skybox/top.jpg",
        // }.build();
        let skybox = CubeMapBuilder {
            back: "assets/darkcity/darkcity_lf.tga",
            front: "assets/darkcity/darkcity_rt.tga",
            right: "assets/darkcity/darkcity_ft.tga",
            bottom: "assets/darkcity/darkcity_dn.tga",
            left: "assets/darkcity/darkcity_bk.tga",
            top: "assets/darkcity/darkcity_up.tga",
        }.build();
        let mut texture_cache = TextureCache::new();
        let nanosuit = Model::<Vertex>::new_from_disk(
            &mut texture_cache,
            "assets/nanosuit_reflection/nanosuit.obj",
        );
        // let nanosuit = Model::new_from_disk(&mut texture_cache, "assets/nice_thing/nice.obj");
        let cyborg = Model::<Vertex>::new_from_disk(&mut texture_cache, "assets/cyborg/cyborg.obj");

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

        let cube_mesh = Mesh::new(&cube_vertices(), vec![tex1, tex2]);
        // let (a, b, c, d) = Vertex::soa(&rect_vertices());
        // let rect_mesh = Mesh::new(&a, &b, &c, &d, None, vec![]);
        let rect = {
            let mut rect_vao = VertexArray::new();
            let mut rect_vbo: VertexBuffer<V3> = VertexBuffer::new();

            {
                let mut vbo = rect_vbo.bind();
                vbo.buffer_data(&[v3(0.0, 0.0, 0.0)]);
                rect_vao.bind().vbo_attrib(&vbo, 0, 3, 0);
            }

            rect_vao
        };

        let final_target = RenderTarget::new(w, h);
        let window_fbo = unsafe { Framebuffer::window() };

        let g = GRenderPass::new(w, h);

        let color_a = RenderTarget::new(w, h);
        let color_b = RenderTarget::new(w, h);

        let blur_scale = 2;
        let blur_1 = RenderTarget::new(w / blur_scale, h / blur_scale);
        let blur_scale = 3;
        let blur_2 = RenderTarget::new(w / blur_scale, h / blur_scale);
        let blur_scale = 4;
        let blur_3 = RenderTarget::new(w / blur_scale, h / blur_scale);
        let blur_scale = 5;
        let blur_4 = RenderTarget::new(w / blur_scale, h / blur_scale);
        let blur_scale = 6;
        let blur_5 = RenderTarget::new(w / blur_scale, h / blur_scale);
        let blur_scale = 7;
        let blur_6 = RenderTarget::new(w / blur_scale, h / blur_scale);

        let blur_targets = vec![blur_1, blur_2, blur_3, blur_4, blur_5, blur_6];

        Pipeline {
            warmy_store,

            vertex_program,
            directional_shadow_program,
            point_shadow_program,
            directional_lighting_program,
            point_lighting_program,

            lighting_program,

            skybox_program,
            blur_program,
            hdr_program,
            screen_program,

            nanosuit,
            cyborg,
            cube: cube_mesh,
            rect: rect,

            nanosuit_vbo: VertexBuffer::new(),
            cyborg_vbo: VertexBuffer::new(),
            cube_vbo: VertexBuffer::new(),

            screen_width: w,
            screen_height: h,
            hidpi_factor,

            g,

            color_a,
            color_b,

            // blur_1,
            // blur_2,
            blur_targets,

            final_target,
            window_fbo,

            skybox: skybox.texture,
        }
    }
    pub fn resize(&mut self, w: u32, h: u32) {
        let final_target = RenderTarget::new(w, h);

        let g = GRenderPass::new(w, h);

        let color_a = RenderTarget::new(w, h);
        let color_b = RenderTarget::new(w, h);

        let blur_scale = 2;
        let blur_1 = RenderTarget::new(w / blur_scale, h / blur_scale);
        let blur_scale = 3;
        let blur_2 = RenderTarget::new(w / blur_scale, h / blur_scale);
        let blur_scale = 4;
        let blur_3 = RenderTarget::new(w / blur_scale, h / blur_scale);
        let blur_scale = 5;
        let blur_4 = RenderTarget::new(w / blur_scale, h / blur_scale);
        let blur_scale = 6;
        let blur_5 = RenderTarget::new(w / blur_scale, h / blur_scale);
        let blur_scale = 7;
        let blur_6 = RenderTarget::new(w / blur_scale, h / blur_scale);

        let blur_targets = vec![blur_1, blur_2, blur_3, blur_4, blur_5, blur_6];

        self.final_target = final_target;
        self.g = g;
        self.color_a = color_a;
        self.color_b = color_b;
        self.blur_targets = blur_targets;

        self.screen_width = w;
        self.screen_height = h;
    }
    pub fn render(&mut self, update_shadows: bool, props: RenderProps) {
        let view = props.camera.get_view();
        let view_pos = props.camera.pos;
        let projection = props.camera.get_projection();

        let mut nanosuit_transforms = vec![];
        let mut cyborg_transforms = vec![];
        let mut cube_transforms = vec![];

        for obj in props.objects.iter() {
            match obj.kind {
                ObjectKind::Cube => cube_transforms.push(obj.transform),
                ObjectKind::Nanosuit => nanosuit_transforms.push(obj.transform),
                ObjectKind::Cyborg => cyborg_transforms.push(obj.transform),
            }
        }

        self.nanosuit_vbo.bind().buffer_data(&nanosuit_transforms);
        self.cyborg_vbo.bind().buffer_data(&cyborg_transforms);
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
                {
                    let model = Mat4::from_angle_y(Rad(pi));
                    $program.bind_mat4("model", model);
                    self.cyborg.$func(&$fbo, &$program, &self.cyborg_vbo.bind());
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

        unsafe {
            gl::Viewport(0, 0, self.screen_width as i32, self.screen_height as i32);
        }

        {
            // Render geometry
            let fbo = self.g.fbo.bind();
            fbo.clear(Mask::ColorDepth);
            let mut program_ = self.vertex_program.bind();
            let program = program_.bind();
            program
                .bind_mat4("projection", projection)
                .bind_mat4("view", view)
                .bind_vec3("viewPos", view_pos)
                .bind_float("time", props.time)
                .bind_texture("shadowMap", &self.skybox)
                .bind_texture("skybox", &self.skybox);

            render_scene!(fbo, program);
            // self.render_scene(block, &program, fbo, &mut nanosuit_vbo, &mut cyborg_vbo, &mut cube_vbo);
        }

        // Clear backbuffer
        self.color_b.bind().clear(Mask::ColorDepth);
        if true {
            if update_shadows {
                // Render depth map for directional lights
                let mut p_ = self.directional_shadow_program.bind();
                let p = p_.bind();
                unsafe {
                    let (w, h) = ShadowMap::size();
                    gl::Viewport(0, 0, w as i32, h as i32);
                }
                unsafe {
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
                    gl::Viewport(0, 0, self.screen_width as i32, self.screen_height as i32);
                }
            }
            if update_shadows {
                // Render depth map for point lights
                let (w, h) = PointShadowMap::size();
                unsafe {
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
                    gl::Viewport(0, 0, self.screen_width as i32, self.screen_height as i32);
                }
            }
        }

        {
            // Render lighting
            {
                let fbo = self.color_b.bind();
                fbo.clear(Mask::ColorDepth);

                let mut g_ = self.lighting_program.bind();
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
                    .bind_texture("aAlbedoSpec", &self.g.albedo_spec)
                    .bind_vec3("viewPos", view_pos);

                let lights: &[_] = &props.directional_lights;

                DirectionalLight::bind_multiple(
                    props.camera.pos,
                    lights,
                    TextureSlot::Four,
                    "directionalLights",
                    "nrDirLights",
                    &g,
                );

                PointLight::bind_multiple(
                    props.point_lights,
                    TextureSlot::Five,
                    "pointLights",
                    "nrPointLights",
                    &g,
                );

                GlError::check().expect("Failed prior to draw directional ligts rect");
                draw_rect!(&fbo, &g);
                GlError::check().expect("Failed to draw directional ligts rect");
            }
        }

        // Skybox Pass
        {
            // Copy z-buffer over from geometry pass
            let g_binding = self.g.fbo.read();
            let hdr_binding = self.color_b.draw();
            let (w, h) = (self.screen_width as i32, self.screen_height as i32);
            hdr_binding.blit_framebuffer(
                &g_binding,
                (0, 0, w, h),
                (0, 0, w, h),
                Mask::Depth,
                gl::NEAREST,
            );
        }
        GlError::check().unwrap();
        {
            // Render skybox
            let fbo = self.color_b.bind();
            {
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
                    .bind_texture("skybox", &self.skybox);
                self.cube.bind().draw(&fbo, &program);
                unsafe {
                    gl::DepthFunc(gl::LESS);
                    gl::Enable(gl::CULL_FACE);
                }
            }
        }

        // Blur passes
        {
            // let passes = vec![&mut self.blur_1, &mut self.blur_2];
            let passes = self.blur_targets.iter_mut();
            {
                let size = self.color_b.size.0 as f32;
                let mut prev = &self.color_b;

                for mut next in passes {
                    unsafe {
                        gl::Viewport(0, 0, next.size.0 as i32, next.size.1 as i32);
                    }
                    let scale = size / next.size.0 as f32;
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
            }
            unsafe {
                gl::Viewport(0, 0, self.screen_width as i32, self.screen_height as i32);
            }
        }

        // HDR/Screen Pass
        {
            let fbo = self.final_target.bind();
            fbo.clear(Mask::ColorDepth);

            let mut hdr_ = self.hdr_program.bind();
            let hdr = hdr_.bind();
            hdr.bind_texture("hdrBuffer", &self.color_b.texture)
                // .bind_texture("blur1", &self.blur_1.texture, TextureSlot::One)
                // .bind_texture("blur2", &self.blur_2.texture, TextureSlot::Two)
                .bind_float("time", props.time);

            // for (i, blur) in self.blur_targets.iter().enumerate() {
            //     let n = i;
            //     hdr.bind_texture(&format!("blur[{}]", n), &blur.texture);
            // }
            hdr.bind_textures("blur", self.blur_targets.iter().map(|t| &t.texture));

            draw_rect!(&fbo, &hdr);
        }

        // HiDPI Pass
        {
            unsafe {
                gl::Viewport(
                    0,
                    0,
                    (self.screen_width as f32 * self.hidpi_factor) as i32,
                    (self.screen_height as f32 * self.hidpi_factor) as i32,
                );
            }
            let fbo = self.window_fbo.bind();
            fbo.clear(Mask::ColorDepth);

            let mut screen_ = self.screen_program.bind();
            let screen = screen_.bind();
            screen.bind_texture("display", &self.final_target.texture);

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
