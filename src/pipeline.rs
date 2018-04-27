use std::{self, rc::Rc};
use gl;
use cgmath::Rad;
use timing::{Timer, NoopTimer};
use mg::*;
use flame;
use render::{Camera, DirectionalLight, PointLight, Model, Mesh, GRenderPass,
             CubeMapBuilder, Object, ObjectKind, Vertex, v3, Image, ImageKind,
             Mat4, V4, ShadowMap, PointShadowMap};

pub struct RenderProps<'a> {
    pub objects: &'a [Object],
    pub camera: &'a Camera,
    pub time: f32,

    pub directional_lights: &'a mut [DirectionalLight],
    pub point_lights: &'a mut [PointLight],
}

pub struct Pipeline {
    pub vertex_program: Program,
    pub directional_shadow_program: Program,
    pub point_shadow_program: Program,
    pub directional_lighting_program: Program,
    pub point_lighting_program: Program,
    pub skybox_program: Program,
    pub hdr_program: Program,

    pub nanosuit: Model,
    pub cyborg: Model,
    pub cube: Mesh,
    pub rect: Mesh,

    pub nanosuit_vbo: VertexBuffer<Mat4>,
    pub cyborg_vbo: VertexBuffer<Mat4>,
    pub cube_vbo: VertexBuffer<Mat4>,

    pub screen_width: u32,
    pub screen_height: u32,

    pub g: GRenderPass,
    pub color_a_fbo: Framebuffer,
    pub color_a_tex: Texture,
    pub color_b_fbo: Framebuffer,
    pub color_b_tex: Texture,

    pub window_fbo: Framebuffer,

    pub skybox: Texture,
}

impl Pipeline {
    fn load_vertex_program() -> Program {
        Program::new_from_disk("shaders/shader.vert", None, "shaders/shader.frag").unwrap()
    }

    pub fn new(w: u32, h: u32) -> Pipeline {
        unsafe {
            gl::Enable(gl::DEPTH_TEST);
            gl::Enable(gl::CULL_FACE);
            gl::Enable(gl::FRAMEBUFFER_SRGB);
        }

        let vertex_program = Pipeline::load_vertex_program();
        let skybox_program =
            Program::new_from_disk("shaders/skybox.vert", None, "shaders/skybox.frag").unwrap();
        let hdr_program =
            Program::new_from_disk("shaders/hdr.vert", None, "shaders/hdr.frag").unwrap();
        let directional_shadow_program =
            Program::new_from_disk("shaders/shadow.vert", None, "shaders/shadow.frag").unwrap();
        let point_shadow_program = Program::new_from_disk(
            "shaders/point_shadow.vert",
            Some("shaders/point_shadow.geom"),
            "shaders/point_shadow.frag",
        ).unwrap();
        let directional_lighting_program = Program::new_from_disk(
            "shaders/lighting.vert",
            None,
            "shaders/directional_lighting.frag",
        ).unwrap();
        let point_lighting_program =
            Program::new_from_disk("shaders/lighting.vert", None, "shaders/point_lighting.frag")
                .unwrap();

        let skybox = CubeMapBuilder {
            back: "assets/skybox/back.jpg",
            front: "assets/skybox/front.jpg",
            right: "assets/skybox/right.jpg",
            bottom: "assets/skybox/bottom.jpg",
            left: "assets/skybox/left.jpg",
            top: "assets/skybox/top.jpg",
        }.build();
        let nanosuit = Model::new_from_disk("assets/nanosuit_reflection/nanosuit.obj");
        let cyborg = Model::new_from_disk("assets/cyborg/cyborg.obj");

        let tex1 = Image::new_from_disk("assets/container2.png", ImageKind::Diffuse);
        let tex2 = Image::new_from_disk("assets/container2_specular.png", ImageKind::Specular);
        let (tex1, tex2) = (Rc::new(tex1), Rc::new(tex2));

        let (a, b, c, d) = Vertex::soa(&cube_vertices());
        let cube_mesh = Mesh::new(&a, &b, &c, &d, None, vec![tex1, tex2]);
        let (a, b, c, d) = Vertex::soa(&rect_vertices());
        let rect_mesh = Mesh::new(&a, &b, &c, &d, None, vec![]);

        let window_fbo = unsafe { Framebuffer::window() };

        let g = GRenderPass::new(w, h);

        let mut color_a_fbo = Framebuffer::new();
        let mut color_a_depth = Renderbuffer::new();
        color_a_depth
            .bind()
            .storage(TextureInternalFormat::DepthComponent, w, h);
        let color_a_tex = Texture::new(TextureKind::Texture2d);
        color_a_tex
            .bind()
            .empty(
                TextureTarget::Texture2d,
                0,
                TextureInternalFormat::Rgba16f,
                w,
                h,
                TextureFormat::Rgba,
                GlType::Float,
            )
            .parameter_int(TextureParameter::MinFilter, gl::LINEAR as i32)
            .parameter_int(TextureParameter::MagFilter, gl::LINEAR as i32);
        color_a_fbo
            .bind()
            .texture_2d(
                Attachment::Color0,
                TextureTarget::Texture2d,
                &color_a_tex,
                0,
            )
            .renderbuffer(Attachment::Depth, &color_a_depth)
            .check_status()
            .expect("framebuffer not complete");

        let mut color_b_fbo = Framebuffer::new();
        let mut color_b_depth = Renderbuffer::new();
        color_b_depth
            .bind()
            .storage(TextureInternalFormat::DepthComponent, w, h);
        let color_b_tex = Texture::new(TextureKind::Texture2d);
        color_b_tex
            .bind()
            .empty(
                TextureTarget::Texture2d,
                0,
                TextureInternalFormat::Rgba16f,
                w,
                h,
                TextureFormat::Rgba,
                GlType::Float,
            )
            .parameter_int(TextureParameter::MinFilter, gl::LINEAR as i32)
            .parameter_int(TextureParameter::MagFilter, gl::LINEAR as i32);
        color_b_fbo
            .bind()
            .texture_2d(
                Attachment::Color0,
                TextureTarget::Texture2d,
                &color_b_tex,
                0,
            )
            .renderbuffer(Attachment::Depth, &color_a_depth)
            .check_status()
            .expect("framebuffer not complete");

        Pipeline {
            vertex_program,
            directional_shadow_program,
            point_shadow_program,
            directional_lighting_program,
            point_lighting_program,
            skybox_program,
            hdr_program,

            nanosuit,
            cyborg,
            cube: cube_mesh,
            rect: rect_mesh,

            nanosuit_vbo: VertexBuffer::new(),
            cyborg_vbo: VertexBuffer::new(),
            cube_vbo: VertexBuffer::new(),

            screen_width: w,
            screen_height: h,

            g,
            color_a_fbo,
            color_a_tex,
            color_b_fbo,
            color_b_tex,

            window_fbo,

            skybox: skybox.texture,
        }
    }
    #[flame]
    pub fn render<'a, T: Timer<'a>>(&mut self, timings: &mut T, update_shadows: bool, props: RenderProps) {
        macro_rules! begin {
            ($name:expr) => (
                timings.time($name)
            )
        }

        begin!("reload vertex shader");
        let hot_reload = false;
        if hot_reload {
            self.vertex_program = Pipeline::load_vertex_program();
        }

        begin!("setup");

        let view = props.camera.get_view();
        let view_pos = props.camera.pos;
        let projection = props.camera.get_projection();

        let mut nanosuit_transforms = vec![];
        let mut cyborg_transforms = vec![];
        let mut cube_transforms = vec![];

        begin!("categorize objects");

        for obj in props.objects.iter() {
            match obj.kind {
                ObjectKind::Cube => cube_transforms.push(obj.transform),
                ObjectKind::Nanosuit => nanosuit_transforms.push(obj.transform),
                ObjectKind::Cyborg => cyborg_transforms.push(obj.transform),
            }
        }
        begin!("load instances to gpu");

        self.nanosuit_vbo.bind().buffer_data(&nanosuit_transforms);
        self.cyborg_vbo.bind().buffer_data(&cyborg_transforms);
        self.cube_vbo.bind().buffer_data(&cube_transforms);


        let pi = std::f32::consts::PI;

        macro_rules! render_scene {
            ($pass:expr, $fbo:expr, $program:expr) => {{
                let _guard = flame::start_guard("render scene");
                if false {
                    $pass.time("render nanosuit");
                    let model = Mat4::from_scale(1.0 / 4.0) * Mat4::from_translation(v3(0.0, 0.0, 4.0));
                    $program.bind_mat4("model", model);
                    // let block = $pass.block("draw nanosuit instanced");
                    let block = &mut NoopTimer;
                    self.nanosuit
                        .draw_instanced(block, &$fbo, &$program, &self.nanosuit_vbo.bind());
                    block.end_block();
                }
                {
                    $pass.time("render cyborg");
                    let model = Mat4::from_angle_y(Rad(pi));
                    $program.bind_mat4("model", model);
                    // let block = $pass.block("draw cyborg instanced");
                    let block = &mut NoopTimer;
                    self.cyborg
                        .draw_instanced(block, &$fbo, &$program, &self.cyborg_vbo.bind());
                    block.end_block();
                }
                {
                    $pass.time("render cube");
                    let model = Mat4::from_scale(1.0);
                    $program.bind_mat4("model", model);
                    self.cube
                        .bind()
                        .draw_instanced(&mut NoopTimer, &$fbo, &$program, &self.cube_vbo.bind());
                }
            }};
        };

        {
            let _guard = flame::start_guard("render geometry");
            begin!("prepare render geometry");
            // Render geometry
            let fbo = self.g.fbo.bind();
            fbo.clear(Mask::ColorDepth);
            let program = self.vertex_program.bind();
            program
                .bind_mat4("projection", projection)
                .bind_mat4("view", view)
                .bind_vec3("viewPos", view_pos)
                .bind_float("time", props.time)
                .bind_texture("shadowMap", &self.skybox, TextureSlot::Six)
                .bind_texture("skybox", &self.skybox, TextureSlot::Six);

            let block = timings.block("render geometry");
            render_scene!(block, fbo, program);
            block.end_block();
            // self.render_scene(block, &program, fbo, &mut nanosuit_vbo, &mut cyborg_vbo, &mut cube_vbo);
        }

        // Clear backbuffer
        self.color_b_fbo.bind().clear(Mask::ColorDepth);
        if update_shadows || true {
            {
                let _guard = flame::start_guard("render directional shadows");
                begin!("update directional shadowmap");
                // Render depth map for directional lights
                let p = self.directional_shadow_program.bind();
                unsafe {
                    let (w, h) = ShadowMap::size();
                    gl::Viewport(
                        0,
                        0,
                        w as i32,
                        h as i32,
                    );
                    gl::CullFace(gl::FRONT);
                }
                for light in props.directional_lights.iter_mut() {
                    let (fbo, light_space) = light.bind_shadow_map();
                    fbo.clear(Mask::Depth);
                    p.bind_mat4("lightSpace", light_space);
                    let mut block = timings.block("dir");
                    render_scene!(block, fbo, p);
                    block.end_block();
                }
            }
            {
                let _guard = flame::start_guard("render point shadows");
                begin!("update point shadowmap");
                // Render depth map for point lights
                let (w, h) = PointShadowMap::size();
                unsafe {
                    gl::Viewport(
                        0,
                        0,
                        w as i32,
                        h as i32,
                    );
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
                    let mut block = timings.block("point");
                    render_scene!(block, fbo, p);
                    block.end_block();
                }
                unsafe {
                    gl::CullFace(gl::BACK);
                    gl::Viewport(
                        0,
                        0,
                        self.screen_width as i32,
                        self.screen_height as i32,
                    );
                }
            }
        }

        {
            // Render lighting
            let _guard = flame::start_guard("render lighting");

            begin!("render lighting, directional pass");
            {
                let fbo = self.color_a_fbo.bind();
                fbo.clear(Mask::ColorDepth);

                let g = self.directional_lighting_program.bind();
                g.bind_texture("aPosition", &self.g.position, TextureSlot::Zero)
                    .bind_texture("aNormal", &self.g.normal, TextureSlot::One)
                    .bind_texture("aAlbedoSpec", &self.g.albedo_spec, TextureSlot::Two)
                    .bind_texture("shadowMap", &self.skybox, TextureSlot::Three)
                    .bind_texture("skybox", &self.skybox, TextureSlot::Three)
                    .bind_vec3("viewPos", view_pos);

                let lights: &[_] = &props.directional_lights;

                DirectionalLight::bind_multiple(
                    lights,
                    TextureSlot::Four,
                    "lights",
                    "nrLights",
                    &g,
                );

                let mut c = VertexBuffer::from_data(&vec![Mat4::from_scale(1.0)]);
                self.rect.bind().draw_instanced(&mut NoopTimer, &fbo, &g, &c.bind());
            }

            begin!("render lighting, point light pass");
            {
                let fbo = self.color_b_fbo.bind();
                fbo.clear(Mask::ColorDepth);

                let g = self.point_lighting_program.bind();
                g.bind_texture("aPosition", &self.g.position, TextureSlot::Zero)
                    .bind_texture("aNormal", &self.g.normal, TextureSlot::One)
                    .bind_texture("aAlbedoSpec", &self.g.albedo_spec, TextureSlot::Two)
                    .bind_texture("skybox", &self.skybox, TextureSlot::Three)
                    .bind_texture("shadowMap", &self.skybox, TextureSlot::Three)
                    .bind_texture("accumulator", &self.color_a_tex, TextureSlot::Four)
                    .bind_vec3("viewPos", view_pos);

                PointLight::bind_multiple(
                    props.point_lights,
                    TextureSlot::Five,
                    "lights",
                    "nrLights",
                    &g,
                );
                GlError::check().unwrap();

                let mut c = VertexBuffer::from_data(&vec![Mat4::from_scale(1.0)]);
                GlError::check().expect("pre-render");
                let mut bind = self.rect.bind();
                GlError::check().expect("post rect bind");
                let c = c.bind();
                GlError::check().expect("post c bind");
                bind.draw_instanced(&mut NoopTimer, &fbo, &g, &c);
                GlError::check().expect("post-render");
            }
        }

        begin!("render lighting, skybox pass");

        let color_fbo = &mut self.color_b_fbo;
        let color_tex = &self.color_b_tex;

        // Skybox Pass
        {
            // Copy z-buffer over from geometry pass
            let g_binding = self.g.fbo.read();
            let hdr_binding = color_fbo.draw();
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
            let fbo = color_fbo.bind();
            {
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
                    .bind_texture("skybox", &self.skybox, TextureSlot::Ten);
                self.cube.bind().draw(&fbo, &program);
                unsafe {
                    gl::DepthFunc(gl::LESS);
                    gl::Enable(gl::CULL_FACE);
                }
            }
        }

        begin!("screen pass pre");
        // HDR/Screen Pass
        {
            let fbo = self.window_fbo.bind();
            fbo.clear(Mask::ColorDepth);

            let hdr = self.hdr_program.bind();
            hdr.bind_texture("hdrBuffer", &color_tex, TextureSlot::Zero)
                .bind_float("time", props.time);
            let mut c = VertexBuffer::from_data(&vec![Mat4::from_scale(1.0)]);

            begin!("actual screen pass");
            self.rect.bind().draw_instanced(&mut NoopTimer, &fbo, &hdr, &c.bind());


            // #[repr(C)]
            // struct Ve3(f32, f32, f32);
            // let verticies = [
            //     Ve3( 0.5,  0.5, 0.0),  // top right
            //     Ve3( 0.5, -0.5, 0.0),  // bottom right
            //     Ve3(-0.5, -0.5, 0.0),  // bottom left
            //     Ve3(-0.5,  0.5, 0.0)   // top left
            // ];
            // #[repr(C)]
            // struct Ve2(f32, f32);
            // let instances = [
            //     Ve2( 0.5,  0.5),  // top right
            //     Ve2( 0.5, -0.5),  // bottom right
            //     Ve2(-0.5, -0.5),  // bottom left
            //     Ve2(-0.5,  0.5)   // top left
            // ];
            // let indecies: [u32; 6] = [
            //     0, 3, 1,
            //     1, 3, 2
            // ];

            // let mut vao = VertexArray::new();
            // let mut vbo = VertexBuffer::from_data(&verticies);
            // let mut vbo_instances = VertexBuffer::from_data(&instances);
            // let mut ebo = ElementBuffer::from_data(&indecies);
            // {
            //     let mut vao = vao.bind();
            //     {
            //         let vbo = vbo.bind();
            //         vao.vbo_attrib(&vbo, 0, 3, 0);
            //     }
            //     {
            //         let vbo_instances = vbo_instances.bind();
            //         vao.vbo_attrib(&vbo_instances, 1, 2, 2 * std::mem::size_of::<f32>());
            //         vao.attrib_divisor(1, 1);
            //     let ebo = ebo.bind();
            //     vao.draw_elements_instanced(&fbo, DrawMode::Triangles, &ebo, vbo_instances.len());
            //     }
            // }
        }
    }
}

macro_rules! v {
    ($pos:expr, $norm:expr, $tex:expr, $tangent:expr) => {{
        let tangent = $tangent.into();
        Vertex {
            pos: $pos.into(),
            tex: $tex.into(),
            norm: $norm.into(),
            tangent: tangent,
        }
    }};
}

fn rect_vertices() -> Vec<Vertex> {
    vec![
        v!(
            [-1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [-1.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [-1.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [1.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0, 0.0]
        ),
        v!(
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0],
            [0.0, 1.0, 0.0]
        ),
    ]
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
