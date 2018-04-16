#![feature(fs_read_write, stmt_expr_attributes, transpose_result)]

extern crate cgmath;
extern crate genmesh;
extern crate gl;
extern crate glutin;
extern crate image;
extern crate obj;
extern crate time;

use cgmath::{InnerSpace, Rad};

use glutin::GlContext;

use time::{Duration, PreciseTime};

use std::rc::Rc;

mod mg;
mod render;

use mg::*;
use render::*;

struct Camera {
    pos: V3,
    fov: Rad<f32>,
    aspect: f32,
    yaw: f32,
    pitch: f32,
}

impl Camera {
    fn new(pos: V3, fov: Rad<f32>, aspect: f32) -> Camera {
        Camera {
            pos,
            fov,
            aspect,
            yaw: 0.0,
            pitch: 0.0,
        }
    }

    fn up(&self) -> V3 {
        v3(0.0, 1.0, 0.0)
    }
    fn front(&self) -> V3 {
        let (ps, pc) = self.pitch.sin_cos();
        let (ys, yc) = self.yaw.sin_cos();

        v3(pc * yc, ps, pc * ys).normalize()
    }
    #[allow(unused)]
    fn front_look_at(&self, target: &V3) -> V3 {
        (target - self.pos).normalize()
    }
    fn get_view(&self) -> Mat4 {
        let origo = cgmath::Point3::new(0.0, 0.0, 0.0);
        Mat4::look_at(origo + self.pos, origo + self.pos + self.front(), self.up())
    }
    #[allow(unused)]
    fn get_view_look_at(&self, target: &V3) -> Mat4 {
        let origo = cgmath::Point3::new(0.0, 0.0, 0.0);
        Mat4::look_at(
            origo + self.pos,
            origo + self.pos + self.front_look_at(target),
            self.up(),
        )
    }
    fn get_projection(&self) -> Mat4 {
        cgmath::PerspectiveFov {
            fovy: self.fov,
            aspect: self.aspect,
            near: 0.01,
            far: 100.0,
        }.into()
    }
}

struct CubeMapBuilder<T> {
    back: T,
    front: T,
    right: T,
    bottom: T,
    left: T,
    top: T,
}

impl<'a> CubeMapBuilder<&'a str> {
    fn build(self) -> Image {
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

        Image {
            texture,
            path: sum_path,
            kind: ImageKind::CubeMap,
        }
    }
}

#[derive(Default)]
struct Input {
    w: f32,
    a: f32,
    s: f32,
    d: f32,

    up: f32,
    down: f32,
    left: f32,
    right: f32,

    space: f32,
    shift: f32,
}

fn main() {
    let (screen_width, screen_height) = (800, 600);
    let mut events_loop = glutin::EventsLoop::new();
    let window = glutin::WindowBuilder::new()
        .with_title("Hello, world!")
        .with_dimensions(screen_width, screen_height);
    let context = glutin::ContextBuilder::new().with_vsync(true);
    let gl_window = glutin::GlWindow::new(window, context, &events_loop).unwrap();

    unsafe {
        gl_window.make_current().unwrap();
    }

    gl::load_with(|symbol| gl_window.get_proc_address(symbol) as *const _);

    let mut program =
        Program::new_from_disk("shaders/shader.vs", None, "shaders/shader.fs").unwrap();
    let mut skybox_program =
        Program::new_from_disk("shaders/skybox.vs", None, "shaders/skybox.fs").unwrap();
    let mut hdr_program = Program::new_from_disk("shaders/hdr.vs", None, "shaders/hdr.fs").unwrap();
    let mut directional_shadow_program =
        Program::new_from_disk("shaders/shadow.vs", None, "shaders/shadow.fs").unwrap();
    let mut point_shadow_program = Program::new_from_disk(
        "shaders/point_shadow.vs",
        Some("shaders/point_shadow.gs"),
        "shaders/point_shadow.fs",
    ).unwrap();
    let mut directional_lighting_program = Program::new_from_disk(
        "shaders/lighting.vs",
        None,
        "shaders/directional_lighting.fs",
    ).unwrap();
    let mut point_lighting_program =
        Program::new_from_disk("shaders/lighting.vs", None, "shaders/point_lighting.fs").unwrap();

    let skybox = CubeMapBuilder {
        back: "assets/skybox/back.jpg",
        front: "assets/skybox/front.jpg",
        right: "assets/skybox/right.jpg",
        bottom: "assets/skybox/bottom.jpg",
        left: "assets/skybox/left.jpg",
        top: "assets/skybox/top.jpg",
    }.build();
    let mut nanosuit = Model::new_from_disk("assets/nanosuit_reflection/nanosuit.obj");
    let mut cyborg = Model::new_from_disk("assets/cyborg/cyborg.obj");

    let tex1 = Image::new_from_disk("assets/container2.png", ImageKind::Diffuse);
    let tex2 = Image::new_from_disk("assets/container2_specular.png", ImageKind::Specular);
    let (tex1, tex2) = (Rc::new(tex1), Rc::new(tex2));

    let mut cube_mesh = Mesh::new(cube_vertices(), None, vec![tex1, tex2]);
    let mut rect_mesh = Mesh::new(rect_verticies(), None, vec![]);

    let mut t: f32 = 0.0;

    unsafe {
        gl::Enable(gl::DEPTH_TEST);
        gl::Enable(gl::CULL_FACE);
        gl::Enable(gl::FRAMEBUFFER_SRGB);
    }

    let mut camera = Camera::new(
        v3(t.sin(), (t / 10.0).sin(), 1.0),
        Rad(std::f32::consts::PI / 2.0),
        (screen_width as f32) / (screen_height as f32),
    );

    let mut inputs = Input::default();

    let mut running = true;
    let mut last_pos = None;
    let mut is = vec![];
    for i in 0..2 {
        for n in 0..i {
            let x = i as f32 / 2.0;
            let v = v3(n as f32 - x, -i as f32 - 5.0, i as f32 / 2.0) * 2.0;
            let v = Mat4::from_translation(v) * Mat4::from_angle_y(Rad(i as f32 - 1.0));
            let obj = Object {
                kind: ObjectKind::Nanosuit,
                transform: v,
            };
            is.push(obj);
            let obj = Object {
                kind: ObjectKind::Cyborg,
                transform: v,
            };
            is.push(obj);
        }
    }
    println!("drawing {} nanosuits", is.len());

    let (w, h) = gl_window.get_inner_size().unwrap();
    let (w, h) = (w * 2, h * 2);

    let mut window_fbo = unsafe { Framebuffer::window() };

    let mut g = GRenderPass::new(w, h);

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

    let light_pos1 = v3(
        1.5 + -20.0 * (t / 10.0).sin(),
        1.0,
        -20.0 * (t / 20.0).sin(),
    );
    let light_pos2 = v3(
        1.5 + -10.0 * ((t + 23.0) / 14.0).sin(),
        2.0,
        -10.0 * (t / 90.0).sin(),
    );
    // let light_pos2 = camera.pos + v3(
    //     (t / 10.0).sin() * 2.0,
    //     3.0,
    //     (t / 10.0).cos() * 2.0,
    // );

    let one = v3(1.0, 1.0, 1.0);

    let mut sun = DirectionalLight {
        diffuse: v3(0.8, 0.8, 0.8) * 1.0,
        ambient: one * 0.1,
        specular: v3(0.8, 0.8, 0.8) * 0.2,

        direction: v3(1.5, 1.0, 0.0).normalize(),

        shadow_map: ShadowMap::new(),
    };

    let mut point_light_1 = PointLight {
        diffuse: v3(0.4, 0.3, 0.3),
        ambient: one * 0.0,
        specular: one * 0.2,

        position: light_pos1,
        last_shadow_map_position: light_pos1,

        constant: 1.0,
        linear: 0.07,
        quadratic: 0.017,

        shadow_map: PointShadowMap::new(),
    };
    let mut point_light_2 = PointLight {
        diffuse: v3(0.2, 0.2, 0.2),
        ambient: one * 0.0,
        specular: one * 0.2,

        position: light_pos2,
        last_shadow_map_position: light_pos2,

        constant: 1.0,
        linear: 0.07,
        quadratic: 0.007,

        shadow_map: PointShadowMap::new(),
    };
    let mut point_light_3 = PointLight {
        diffuse: v3(0.2, 0.2, 0.2),
        ambient: one * 0.0,
        specular: one * 0.2,

        position: light_pos1 + light_pos2,
        last_shadow_map_position: light_pos1 + light_pos2,

        constant: 1.0,
        linear: 0.07,
        quadratic: 0.007,

        shadow_map: PointShadowMap::new(),
    };
    let mut point_light_4 = PointLight {
        diffuse: v3(0.2, 0.2, 0.8),
        ambient: one * 0.0,
        specular: one * 0.2,

        position: v3(
            light_pos1.x * light_pos2.x,
            light_pos1.y * light_pos2.y,
            light_pos1.z * light_pos2.z,
        ),
        last_shadow_map_position: one,

        constant: 1.0,
        linear: 0.07,
        quadratic: 0.007,

        shadow_map: PointShadowMap::new(),
    };

    let mut fps_last_time = PreciseTime::now();
    let fps_step = Duration::seconds(1);
    let mut fps_number_of_frames = 0;

    let mut shadows_last_time = PreciseTime::now();
    let shadows_step = Duration::milliseconds(16 * 8);

    let mut update_shadows;

    while running {
        let mut mouse_delta = (0.0, 0.0);

        let now = PreciseTime::now();
        {
            fps_number_of_frames += 1;
            let delta = fps_last_time.to(now);
            if delta > fps_step {
                fps_last_time = now;
                gl_window.set_title(&format!("FPS: {}", fps_number_of_frames));
                fps_number_of_frames = 0;
                hdr_program =
                    Program::new_from_disk("shaders/hdr.vs", None, "shaders/hdr.fs").unwrap();
            }
        }
        {
            update_shadows = false;
            let delta = shadows_last_time.to(now);
            if delta > shadows_step {
                shadows_last_time = now;
                update_shadows = true;
            }
        }

        events_loop.poll_events(|event| match event {
            glutin::Event::WindowEvent { event, .. } => match event {
                glutin::WindowEvent::Closed => running = false,
                glutin::WindowEvent::Resized(w, h) => gl_window.resize(w, h),
                glutin::WindowEvent::CursorMoved { position, .. } => {
                    match last_pos {
                        None => {
                            last_pos = Some(position);
                        }
                        Some(lp) => {
                            last_pos = Some(position);
                            mouse_delta = (
                                position.0 as f32 - lp.0 as f32,
                                position.1 as f32 - lp.1 as f32,
                            );
                        }
                    }
                    // let (w, h) = gl_window.get_outer_size().unwrap();
                    // let (x, y) = gl_window.get_position().unwrap();
                    // ignore_next_mouse_move = true;
                    // gl_window.set_cursor_position(x + w as i32 / 2, y + h as i32 / 2).unwrap();
                }
                glutin::WindowEvent::KeyboardInput { input, .. } => {
                    if let Some(keycode) = input.virtual_keycode {
                        use glutin::VirtualKeyCode as Kc;

                        let value = if input.state == glutin::ElementState::Pressed {
                            1.0
                        } else {
                            0.0
                        };

                        match keycode {
                            Kc::Escape => running = false,
                            Kc::W => inputs.w = value,
                            Kc::A => inputs.a = value,
                            Kc::S => inputs.s = value,
                            Kc::D => inputs.d = value,
                            Kc::Up => inputs.up = value,
                            Kc::Down => inputs.down = value,
                            Kc::Left => inputs.left = value,
                            Kc::Right => inputs.right = value,
                            Kc::Space => inputs.space = value,
                            Kc::LShift => inputs.shift = value,
                            _ => {}
                        }
                    }
                }
                _ => {}
            },
            _ => (),
        });

        t += 1.0;

        point_light_1.position = v3(
            1.5 + -20.0 * (t / 10.0).sin(),
            1.0,
            -20.0 * (t / 20.0).sin(),
        );
        point_light_2.position = v3(
            1.5 + -10.0 * ((t + 23.0) / 14.0).sin(),
            2.0,
            -10.0 * (t / 90.0).sin(),
        );
        point_light_3.position = point_light_1.position + point_light_2.position;
        point_light_4.position = v3(
            point_light_1.position.x * point_light_2.position.x,
            point_light_1.position.y * point_light_2.position.y,
            point_light_1.position.z * point_light_2.position.z,
        );

        let pi = std::f32::consts::PI;

        let up = camera.up();
        let right = camera.front().cross(up).normalize();
        let front = up.cross(right).normalize();

        let walk_speed = 0.1;
        let sensitivity = 0.005;

        camera.pos += walk_speed
            * (front * (inputs.w - inputs.s) + right * (inputs.d - inputs.a)
                + up * (inputs.space - inputs.shift));
        camera.yaw += sensitivity * mouse_delta.0;
        camera.pitch = (camera.pitch - sensitivity * mouse_delta.1)
            .max(-pi / 2.001)
            .min(pi / 2.001);
        // camera.yaw += sensitivity * (inputs.right - inputs.left);
        // camera.pitch = (camera.pitch + sensitivity * (inputs.up - inputs.down))
        //     .max(-pi / 2.001)
        //     .min(pi / 2.001);

        let view = camera.get_view();
        let projection = camera.get_projection();

        // Begin rendering!
        {
            let mut objects = vec![
                Object {
                    kind: ObjectKind::Cube,
                    transform: Mat4::from_nonuniform_scale(100.0, 0.1, 100.0)
                        * Mat4::from_translation(v3(0.0, -200.0, 0.0)),
                },
                Object {
                    kind: ObjectKind::Cube,
                    transform: Mat4::from_translation(point_light_1.position.clone()),
                },
                Object {
                    kind: ObjectKind::Cube,
                    transform: Mat4::from_translation(point_light_2.position.clone()),
                },
                Object {
                    kind: ObjectKind::Cube,
                    transform: Mat4::from_translation(point_light_3.position.clone()),
                },
                Object {
                    kind: ObjectKind::Cube,
                    transform: Mat4::from_translation(point_light_4.position.clone()),
                },
            ];

            objects.append(&mut is.clone());

            Pipeline {
                vertex_program: &mut program,
                directional_shadow_program: &mut directional_shadow_program,
                point_shadow_program: &mut point_shadow_program,
                directional_lighting_program: &mut directional_lighting_program,
                point_lighting_program: &mut point_lighting_program,
                skybox_program: &mut skybox_program,
                hdr_program: &mut hdr_program,

                nanosuit: &mut nanosuit,
                cyborg: &mut cyborg,
                cube: &mut cube_mesh,
                rect: &mut rect_mesh,

                objects: objects,
                directional_lights: vec![&mut sun],
                point_lights: vec![
                    &mut point_light_1,
                    &mut point_light_2,
                    &mut point_light_3,
                    &mut point_light_4,
                ],

                screen_width: screen_width,
                screen_height: screen_height,

                g: &mut g,
                color_a_fbo: &mut color_a_fbo,
                color_a_tex: &color_a_tex,
                color_b_fbo: &mut color_b_fbo,
                color_b_tex: &color_b_tex,

                window_fbo: &mut window_fbo,

                projection: projection,
                view: view,
                view_pos: camera.pos,
                time: t,
                skybox: &skybox.texture,
            }.render(update_shadows);
        }

        gl_window.swap_buffers().unwrap();
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

fn rect_verticies() -> Vec<Vertex> {
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
