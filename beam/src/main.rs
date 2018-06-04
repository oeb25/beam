#![feature(fs_read_write, stmt_expr_attributes, transpose_result, box_syntax, box_patterns)]
#![feature(custom_attribute, nll, iterator_flatten, concat_idents)]
#![allow(unused_imports)]
#[macro_use]
extern crate failure;
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate glsl_layout;
extern crate cgmath;
extern crate collada;
extern crate gl;
extern crate glutin;
extern crate image;
extern crate mg;
extern crate time;
extern crate warmy;

use std::collections::VecDeque;

use failure::Error;

use cgmath::{InnerSpace, Rad};

use glutin::GlContext;

use time::{Duration, PreciseTime};

pub mod assets;
pub mod hot;
pub mod logic;
pub mod misc;
pub mod pipeline;
pub mod render;

use misc::{v3, Mat4, P3};
use pipeline::{Pipeline, RenderProps};
use render::{hex, hsv, mesh, rgb, Camera, Material, RenderObject};

use render::lights::{DirectionalLight, PointLight, PointShadowMap, ShadowMap, SpotLight};

#[derive(Default)]
struct Input {
    mouse_delta: (f32, f32),

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
    ctrl: f32,
}

struct Scene {
    camera: Camera,
    game: logic::Game,
    directional_lights: Vec<DirectionalLight>,
    point_lights: Vec<PointLight>,
    spot_lights: Vec<SpotLight>,
}

impl Scene {
    fn new(screen_width: u32, screen_height: u32) -> Scene {
        let game = logic::Game::new();

        let one = v3(1.0, 1.0, 1.0);

        let sun = DirectionalLight {
            color: hsv(0.1, 0.5, 1.0) * 0.05,
            direction: v3(0.0, 1.0, -1.5).normalize(),

            shadow_map: ShadowMap::new(),
        };

        let directional_lights = vec![sun];

        let point_lights = vec![
            PointLight {
                color: rgb(255, 25, 25) * 3.0,
                position: v3(-10.0, 4.0, 10.0),
                last_shadow_map_position: v3(-10.0, 4.0, 10.0),
                shadow_map: Some(PointShadowMap::new()),
            },
            PointLight {
                color: hex(0x0050ff) * 1.0,
                position: v3(10.0, 4.0, 10.0),
                last_shadow_map_position: one,
                shadow_map: None,
            },
            PointLight {
                color: hex(0x00ff2e) * 1.0,
                position: v3(10.0, 4.0, -10.0),
                last_shadow_map_position: one,
                shadow_map: None,
            },
            PointLight {
                color: hex(0xffc700) * 1.0,
                position: v3(-10.0, 4.0, -10.0),
                last_shadow_map_position: one,
                shadow_map: None,
            },
        ];

        let spot_lights = vec![SpotLight {
            color: rgb(25, 25, 255) * 1.9,
            position: v3(0.0, 25.0, 0.0),
            direction: v3(0.0, -1.0, 0.0),
            cut_off: Rad(0.1),
            outer_cut_off: Rad(0.4),
        }];

        let camera = Camera::new(
            v3(-15.0, 5.0, 0.0),
            Rad(std::f32::consts::PI / 2.0),
            (screen_width as f32) / (screen_height as f32),
        );

        Scene {
            camera,
            game,
            directional_lights,
            point_lights,
            spot_lights,
        }
    }
    fn resize(&mut self, screen_width: u32, screen_height: u32) {
        self.camera = Camera::new(
            v3(-15.0, 5.0, 0.0),
            Rad(std::f32::consts::PI / 2.0),
            (screen_width as f32) / (screen_height as f32),
        );
    }
    fn tick(&mut self, _t: f32, dt: f32, inputs: &Input) {
        let pi = std::f32::consts::PI;

        let up = self.camera.up();
        let right = self.camera.front().cross(up).normalize();
        let front = up.cross(right).normalize();

        let walk_speed = (0.1 + 0.2 * inputs.shift) * (dt.max(0.01) / 0.016);
        let sensitivity = 0.005;

        self.camera.pos += walk_speed
            * (front * (inputs.w - inputs.s)
                + right * (inputs.d - inputs.a)
                + up * (inputs.space - inputs.ctrl));
        self.camera.yaw += sensitivity * inputs.mouse_delta.0;
        self.camera.pitch = (self.camera.pitch - sensitivity * inputs.mouse_delta.1)
            .max(-pi / 2.001)
            .min(pi / 2.001);

        // self.camera.yaw += sensitivity * (inputs.right - inputs.left);
        // self.camera.pitch = (self.camera.pitch + sensitivity * (inputs.up - inputs.down))
        //     .max(-pi / 2.001)
        //     .min(pi / 2.001);

        // self.spot_lights[0].position = self.camera.pos;
        // self.spot_lights[0].direction = self.camera.front();
    }
}

fn main() -> Result<(), Error> {
    let (screen_width, screen_height) = (1024, 768);
    let mut events_loop = glutin::EventsLoop::new();
    let window = glutin::WindowBuilder::new()
        .with_title("Hello, world!")
        .with_dimensions(screen_width, screen_height);
    let context = glutin::ContextBuilder::new()
        .with_vsync(true)
        .with_gl_profile(glutin::GlProfile::Core)
        .with_srgb(true);
    let gl_window = glutin::GlWindow::new(window, context, &events_loop).unwrap();
    gl_window
        .window()
        .set_cursor_state(glutin::CursorState::Grab)
        .unwrap();

    let hidpi_factor = gl_window.window().hidpi_factor();

    unsafe {
        gl_window.make_current().unwrap();
    }

    gl::load_with(|symbol| gl_window.get_proc_address(symbol) as *const _);

    let mut ppin = unsafe { mg::Program::get_pin() };
    let mut vpin = unsafe { mg::VertexArray::get_pin() };

    let mut t: f32 = 0.0;

    let (w, h) = gl_window.get_inner_size().unwrap();

    let mut scene = Scene::new(w, h);

    let mut inputs = Input::default();

    let mut running = true;
    // let mut last_pos = None;

    let mut assets = assets::AssetBuilder::new(&mut ppin, &mut vpin)?;
    let room_ibl = assets.load_ibl("assets/Newport_Loft/Newport_Loft_Ref.hdr")?;
    // let rust_material = assets.load_pbr_with_default_filenames("assets/pbr/rusted_iron", "png")?;
    let plastic_material = assets.load_pbr_with_default_filenames("assets/pbr/plastic", "png")?;
    // let gold_material = assets.load_pbr_with_default_filenames("assets/pbr/gold", "png")?;

    let suzanne = assets
        .load_collada("assets/suzanne/suzanne.dae")?
        .scale(1.0 / 2.0)
        .translate(v3(0.0, 20.0, 0.0));

    let owl = assets
        .load_collada("assets/owl/owl.dae")?
        .translate(v3(0.0, 0.0, 0.0));

    let mut is = vec![];

    let cube_mesh = RenderObject::mesh(assets.get_cube());
    let sphere_mesh = RenderObject::mesh(assets.get_sphere());

    let mut gradient_textures = vec![];

    for i in 0..5 {
        let val = i as f32 / 4.0;
        let texture = val;
        gradient_textures.push(texture);
    }

    let sphere_material = Material::new().albedo(v3(0.0, 0.1, 1.0));

    for (i, rough) in gradient_textures.iter().enumerate() {
        for (n, met) in gradient_textures.iter().enumerate() {
            let v = v3(i as f32 * 2.0, 10.0 - n as f32 * 2.0, -13.0);
            let obj = sphere_mesh
                .translate(v)
                .with_material(sphere_material.metallic(met).roughness(rough));
            is.push(obj);
        }
    }

    let mut fps_last_time = PreciseTime::now();
    let fps_step = Duration::seconds(1);
    let mut fps_number_of_frames = 0;

    let mut queued_actions = VecDeque::new();

    let mut last_time = PreciseTime::now();
    let frame_time = 0.5;
    let mut timer = 0.0;

    let logic_render_props = logic::RenderProps {
        owl_mesh: &owl,
        cube_mesh: &cube_mesh,
        plastic_material,
    };

    let light_material = Material::new().metallic(0.0).roughness(0.0);

    let mut render_objects = vec![];

    let mut update_shadows = false;

    let mut pipeline = assets.to_pipeline(w, h, hidpi_factor);

    while running {
        update_shadows = !update_shadows;

        let now = PreciseTime::now();
        let dt = last_time.to(now).num_milliseconds() as f32 / 1000.0;
        last_time = now;

        {
            fps_number_of_frames += 1;
            let delta = fps_last_time.to(now);
            if delta > fps_step {
                fps_last_time = now;
                gl_window.set_title(&format!("FPS: {}", fps_number_of_frames));
                fps_number_of_frames = 0;
            }
        }

        inputs.mouse_delta = (0.0, 0.0);

        events_loop.poll_events(|event| match event {
            glutin::Event::WindowEvent { event, .. } => match event {
                glutin::WindowEvent::CloseRequested => running = false,
                glutin::WindowEvent::Resized(w, h) => {
                    scene.resize(w, h);
                    gl_window.resize(w, h);
                    pipeline.resize(w, h);
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
                            Kc::Escape => {
                                println!("quitting...");
                                running = false;
                            }
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
                            Kc::LControl => inputs.ctrl = value,
                            _ => {}
                        }
                        if input.state == glutin::ElementState::Pressed {
                            use logic::{Action, Direction};
                            let use_wasd = false;
                            // Determin forward according to camera direction
                            let forward = {
                                let cdir = scene.camera.front();
                                let dotx = cdir.dot(v3(1.0, 0.0, 0.0));
                                let doty = cdir.dot(v3(0.0, 0.0, 1.0));
                                if dotx.abs() > 0.5 {
                                    if dotx > 0.0 {
                                        Direction::Left
                                    } else {
                                        Direction::Right
                                    }
                                } else {
                                    if doty > 0.0 {
                                        Direction::Up
                                    } else {
                                        Direction::Down
                                    }
                                }
                            };
                            let action = match (keycode, use_wasd) {
                                (Kc::W, true) | (Kc::Up, _) => Some(Action::Move(forward)),
                                (Kc::D, true) | (Kc::Right, _) => {
                                    Some(Action::Move(forward.clockwise()))
                                }
                                (Kc::S, true) | (Kc::Down, _) => {
                                    Some(Action::Move(forward.opposite()))
                                }
                                (Kc::A, true) | (Kc::Left, _) => {
                                    Some(Action::Move(forward.couter_clockwise()))
                                }
                                (Kc::Z, _) => Some(Action::Undo),
                                _ => None,
                            };
                            if let Some(action) = action {
                                queued_actions.push_back(action);
                            }
                        }
                    }
                }
                // x => println!("{:?}", x),
                _ => {}
            },
            glutin::Event::DeviceEvent { event, .. } => {
                if let glutin::DeviceEvent::MouseMotion { delta } = event {
                    inputs.mouse_delta = (delta.0 as f32, delta.1 as f32);
                }
            }
            // x => println!("{:?}", x),
            _ => (),
        });

        t += 1.0;

        timer += dt;

        if timer > frame_time {
            if let Some(action) = queued_actions.pop_front() {
                println!("{:?}", action);
                timer = 0.0;
                scene.game = scene.game.action(action);
            }
        }

        scene.tick(t, dt, &inputs);

        // Begin rendering!
        {
            render_objects.clear();

            for light in &scene.point_lights {
                let mesh = sphere_mesh
                    .transform(Mat4::from_translation(light.position) * Mat4::from_scale(0.8))
                    .with_material(light_material.albedo(light.color).emission(light.color * 0.8));
                render_objects.push(mesh);
            }

            for light in &scene.spot_lights {
                let mesh = suzanne
                    .transform(Mat4::from_translation(light.position) * Mat4::from_scale(0.8))
                    .with_material(light_material.albedo(light.color).emission(light.color * 0.8));
                render_objects.push(mesh);
            }

            render_objects.append(&mut is.clone());

            // let trace_scene = false;
            // if trace_scene {
            //     let ray = (scene.camera.pos, scene.camera.front());
            //     let res = RenderObject::raymarch_many(
            //         render_objects.iter(),
            //         &pipeline.meshes,
            //         ray.0,
            //         ray.1,
            //     );
            //     if res.1 > 0.99 {
            //         render_objects[res.0] =
            //             render_objects[res.0].with_material(gold_material.clone());
            //     }
            // }

            let game_calls = scene.game.render(&logic_render_props, timer / frame_time);

            let game_call = RenderObject::with_children(game_calls)
                .translate(v3(-5.0, 0.0, -5.0))
                .scale(2.0);

            render_objects.push(game_call);

            pipeline.render(
                &mut ppin,
                &mut vpin,
                update_shadows,
                RenderProps {
                    camera: &scene.camera,
                    directional_lights: &mut scene.directional_lights,
                    point_lights: &mut scene.point_lights,
                    spot_lights: &mut scene.spot_lights,
                    time: t,

                    default_material: None,

                    ibl: &room_ibl,

                    ambient_intensity: Some(1.0),
                    skybox_intensity: Some(1.0),
                },
                render_objects.iter(),
            );
        }

        gl_window.swap_buffers().unwrap();
    }

    println!("Complete");

    Ok(())
}
