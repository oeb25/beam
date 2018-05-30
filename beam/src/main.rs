#![feature(fs_read_write, stmt_expr_attributes, transpose_result, box_syntax, box_patterns)]
#![feature(custom_attribute, nll, iterator_flatten)]
#![allow(unused_imports)]
#[macro_use]
extern crate failure;
extern crate cgmath;
extern crate collada;
extern crate gl;
extern crate glutin;
extern crate image;
extern crate mg;
extern crate time;
extern crate warmy;

use failure::Error;

use cgmath::{InnerSpace, Rad};

use glutin::GlContext;

use time::{Duration, PreciseTime};

mod hot;
mod logic;
mod misc;
mod pipeline;
mod render;

use misc::{v3, Mat4, P3};
use pipeline::{Pipeline, RenderProps};
use render::{hex, hsv, mesh, rgb, Camera, Material, RenderObject};
use std::collections::VecDeque;

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
                color: rgb(255, 25, 25) * 1.0,
                position: v3(-10.0, 2.0, 10.0),
                last_shadow_map_position: v3(-10.0, 2.0, 10.0),
                shadow_map: Some(PointShadowMap::new()),
            },
            PointLight {
                color: hex(0x0050ff) * 1.0,
                position: v3(10.0, 2.0, 10.0),
                last_shadow_map_position: one,
                shadow_map: None,
            },
            PointLight {
                color: hex(0x00ff2e) * 1.0,
                position: v3(10.0, 2.0, -10.0),
                last_shadow_map_position: one,
                shadow_map: None,
            },
            PointLight {
                color: hex(0xffc700) * 1.0,
                position: v3(-10.0, 2.0, -10.0),
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
    fn tick(&mut self, _t: f32, _dt: f32, inputs: &Input) {
        let pi = std::f32::consts::PI;

        let up = self.camera.up();
        let right = self.camera.front().cross(up).normalize();
        let front = up.cross(right).normalize();

        let walk_speed = 0.1;
        let sensitivity = 0.005;

        self.camera.pos += walk_speed
            * (front * (inputs.w - inputs.s)
                + right * (inputs.d - inputs.a)
                + up * (inputs.space - inputs.shift));
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
        .with_vsync(false)
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

    let mut t: f32 = 0.0;

    let (w, h) = gl_window.get_inner_size().unwrap();

    let mut scene = Scene::new(w, h);

    let mut inputs = Input::default();

    let mut running = true;
    let mut last_pos = None;

    let mut pipeline = Pipeline::new(w, h, hidpi_factor);

    let room_ibl = pipeline.load_ibl("assets/Newport_Loft/Newport_Loft_Ref.hdr")?;
    let _rust_material = pipeline
        .meshes
        .load_pbr_with_default_filenames("assets/pbr/rusted_iron", "png")?;
    let _plastic_material = pipeline
        .meshes
        .load_pbr_with_default_filenames("assets/pbr/plastic", "png")?;
    let gold_material = pipeline
        .meshes
        .load_pbr_with_default_filenames("assets/pbr/gold", "png")?;

    let white3 = pipeline.meshes.rgb_texture(v3(1.0, 1.0, 1.0));
    let black3 = pipeline.meshes.rgb_texture(v3(0.0, 0.0, 0.0));
    let normal3 = pipeline.meshes.rgb_texture(v3(0.5, 0.5, 1.0));

    let suzanne = pipeline
        .meshes
        .load_collada("assets/suzanne/suzanne.dae")?
        .scale(1.0 / 2.0)
        .translate(v3(0.0, 20.0, 0.0));

    let owl = pipeline
        .meshes
        .load_collada("assets/owl/owl.dae")?
        .translate(v3(0.0, 0.0, 0.0));

    let mut is = vec![];

    for i in 1..5 {
        for n in 0..i {
            let x = i as f32 / 2.0;
            let v = v3(i as f32 / 2.0, -i as f32 - 5.0, n as f32 - x) * 2.0;
            let v = Mat4::from_translation(v) * Mat4::from_angle_y(Rad(i as f32 - 1.0));
            let obj = suzanne.transform(v); // .with_material(pbr_materials[m % nr_pbr_materials]);
            is.push(obj);
        }
    }
    let v = Mat4::from_angle_y(Rad(-1.5));
    is.push(owl.transform(v));

    println!("drawing {} nanosuits", is.len());

    let cube_mesh = RenderObject::mesh(pipeline.meshes.get_cube());
    let sphere_mesh = RenderObject::mesh(pipeline.meshes.get_sphere(0.5));

    let mut fps_last_time = PreciseTime::now();
    let fps_step = Duration::seconds(1);
    let mut fps_number_of_frames = 0;

    let mut shadows_last_time = PreciseTime::now();
    let shadows_step = Duration::milliseconds(16 * 1);

    let mut update_shadows = false;

    let mut queued_actions = VecDeque::new();

    while running {
        update_shadows = !update_shadows;

        let now = PreciseTime::now();
        {
            fps_number_of_frames += 1;
            let delta = fps_last_time.to(now);
            if delta > fps_step {
                fps_last_time = now;
                gl_window.set_title(&format!("FPS: {}", fps_number_of_frames));
                fps_number_of_frames = 0;
            }
        }
        {
            // update_shadows = false;
            let delta = shadows_last_time.to(now);
            if delta > shadows_step {
                shadows_last_time = now;
                // update_shadows = true;
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
                glutin::WindowEvent::CursorMoved { position, .. } => {
                    match last_pos {
                        None => {
                            last_pos = Some(position);
                        }
                        Some(lp) => {
                            last_pos = Some(position);
                            inputs.mouse_delta = (
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
                        if input.state == glutin::ElementState::Pressed {
                            let use_wasd = false;
                            let action = match keycode {
                                Kc::W if use_wasd => Some(logic::Action::Up),
                                Kc::Up => Some(logic::Action::Up),
                                Kc::D if use_wasd => Some(logic::Action::Right),
                                Kc::Right => Some(logic::Action::Right),
                                Kc::S if use_wasd => Some(logic::Action::Down),
                                Kc::Down => Some(logic::Action::Down),
                                Kc::A if use_wasd => Some(logic::Action::Left),
                                Kc::Left => Some(logic::Action::Left),
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

        if let Some(action) = queued_actions.pop_front() {
            scene.game = scene.game.action(action);
        }

        scene.tick(t, t, &inputs);

        // Begin rendering!
        {
            let mut objects: Vec<_> = [
                (v3(0.0, -0.0, 0.0), v3(20.0, 0.1, 20.0)),
                // (v3(10.0, -10.0, 0.0), v3(0.1, 20.0, 20.0)),
                // (v3(-10.0, -10.0, 0.0), v3(0.1, 20.0, 20.0)),
                // (v3(0.0, -10.0, -10.0), v3(20.0, 20.0, 0.1)),
                // (v3(0.0, -10.0, 10.0), v3(20.0, 20.0, 0.1)),
            ].into_iter()
                .map(|(p, s)| {
                    cube_mesh.transform(
                        Mat4::from_translation(*p) * Mat4::from_nonuniform_scale(s.x, s.y, s.z),
                    )
                })
                .collect();

            for light in &scene.point_lights {
                let mesh = sphere_mesh
                    .transform(Mat4::from_translation(light.position) * Mat4::from_scale(0.8))
                    .with_material(Material {
                        albedo: pipeline.meshes.rgb_texture(light.color),
                        normal: normal3,
                        metallic: black3,
                        roughness: white3,
                        ao: white3,
                        opacity: white3,
                    });
                objects.push(mesh);
            }

            for light in &scene.spot_lights {
                let mesh = suzanne
                    .transform(Mat4::from_translation(light.position) * Mat4::from_scale(0.8))
                    .with_material(Material {
                        albedo: pipeline.meshes.rgb_texture(light.color),
                        normal: normal3,
                        metallic: black3,
                        roughness: black3,
                        ao: white3,
                        opacity: white3,
                    });
                objects.push(mesh);
            }

            objects.append(&mut is.clone());

            let trace_scene = false;
            if trace_scene {
                let ray = (scene.camera.pos, scene.camera.front());
                let res =
                    RenderObject::raymarch_many(objects.iter(), &pipeline.meshes, ray.0, ray.1);
                if res.1 > 0.99 {
                    objects[res.0] = objects[res.0].with_material(gold_material);
                }
            }

            let game_calls = scene.game.render(&owl, &mut pipeline.meshes);

            let game_call = RenderObject::with_children(game_calls)
                .translate(v3(-5.0, 0.0, -5.0))
                .scale(2.0);

            objects.push(game_call);

            pipeline.render(
                update_shadows,
                RenderProps {
                    camera: &scene.camera,
                    directional_lights: &mut scene.directional_lights,
                    point_lights: &mut scene.point_lights,
                    spot_lights: &mut scene.spot_lights,
                    time: t,

                    ibl: &room_ibl,

                    ambient_intensity: Some(0.1),
                    skybox_intensity: Some(0.1),
                },
                objects.iter(),
            );
        }

        gl_window.swap_buffers().unwrap();

        // let report = timings.end();
        // report_cache.push_front(report);
        // if report_cache.len() > 60 {
        //     report_cache.truncate(30);
        // }
        // Report::averange(report_cache.iter()).print();
    }

    // flame::dump_html(&mut std::fs::File::create("flame-graph.html").unwrap()).unwrap();

    Ok(())
}
