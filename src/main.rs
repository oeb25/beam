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

mod mg;
mod render;

use mg::*;
use render::*;

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

    let mut t: f32 = 0.0;

    let mut camera = Camera::new(
        v3(t.sin(), (t / 10.0).sin(), 1.0),
        Rad(std::f32::consts::PI / 2.0),
        (screen_width as f32) / (screen_height as f32),
    );

    let mut inputs = Input::default();

    let mut running = true;
    let mut last_pos = None;
    let mut is = vec![];
    for i in 0..5 {
        for n in 0..i {
            let x = i as f32 / 2.0;
            let v = v3(n as f32 - x, -i as f32 - 5.0, i as f32 / 2.0) * 2.0;
            let v = Mat4::from_translation(v) * Mat4::from_angle_y(Rad(i as f32 - 1.0));
            if false {
                let obj = Object {
                    kind: ObjectKind::Nanosuit,
                    transform: v,
                };
                is.push(obj);
            }
            let obj = Object {
                kind: ObjectKind::Cyborg,
                transform: v,
            };
            is.push(obj);
        }
    }
    println!("drawing {} nanosuits", is.len());

    let (w, h) = gl_window.get_inner_size().unwrap();

    let mut pipeline = Pipeline::new(w * 2, h * 2);

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

    let one = v3(1.0, 1.0, 1.0);

    let mut sun = DirectionalLight {
        diffuse: v3(0.8, 0.8, 0.8) * 1.0,
        ambient: one * 0.01,
        specular: v3(0.8, 0.8, 0.8) * 0.2,

        direction: v3(1.5, 1.0, 0.0).normalize(),

        shadow_map: ShadowMap::new(),
    };

    let mut point_lights = [
        PointLight {
            diffuse: v3(0.4, 0.3, 0.3),
            ambient: one * 0.0,
            specular: one * 0.2,

            position: light_pos1,
            last_shadow_map_position: light_pos1,

            constant: 1.0,
            linear: 0.07,
            quadratic: 0.017,

            shadow_map: PointShadowMap::new(),
        },
        PointLight {
            diffuse: v3(0.2, 0.2, 0.2),
            ambient: one * 0.0,
            specular: one * 0.2,

            position: light_pos2,
            last_shadow_map_position: light_pos2,

            constant: 1.0,
            linear: 0.07,
            quadratic: 0.007,

            shadow_map: PointShadowMap::new(),
        },
        PointLight {
            diffuse: v3(0.2, 0.2, 0.2),
            ambient: one * 0.0,
            specular: one * 0.2,

            position: light_pos1 + light_pos2,
            last_shadow_map_position: light_pos1 + light_pos2,

            constant: 1.0,
            linear: 0.07,
            quadratic: 0.007,

            shadow_map: PointShadowMap::new(),
        },
        PointLight {
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
        },
    ];

    let mut fps_last_time = PreciseTime::now();
    let fps_step = Duration::seconds(1);
    let mut fps_number_of_frames = 0;

    let mut shadows_last_time = PreciseTime::now();
    let shadows_step = Duration::milliseconds(16 * 1);

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

        point_lights[0].position = v3(
            1.5 + -20.0 * (t / 10.0).sin(),
            1.0,
            -20.0 * (t / 20.0).sin(),
        );
        point_lights[1].position = v3(
            1.5 + -10.0 * ((t + 23.0) / 14.0).sin(),
            2.0,
            -10.0 * (t / 90.0).sin(),
        );
        point_lights[2].position = point_lights[0].position + point_lights[1].position;
        point_lights[3].position = v3(
            point_lights[0].position.x * point_lights[1].position.x,
            point_lights[0].position.y * point_lights[1].position.y,
            point_lights[0].position.z * point_lights[1].position.z,
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

        // Begin rendering!
        {
            let mut objects = vec![
                Object {
                    kind: ObjectKind::Cube,
                    transform: Mat4::from_nonuniform_scale(100.0, 0.1, 100.0)
                        * Mat4::from_translation(v3(0.0, -200.0, 0.0)),
                },
            ];

            for light in point_lights.iter() {
                objects.push(Object {
                    kind: ObjectKind::Cube,
                    transform: Mat4::from_translation(light.position.clone()),
                });
            }

            objects.append(&mut is.clone());

            pipeline.render(
                update_shadows,
                RenderProps {
                    camera: &camera,
                    objects: &objects,
                    directional_lights: &mut [&mut sun],
                    point_lights: &mut point_lights,
                    time: t,
                },
            );
        }

        gl_window.swap_buffers().unwrap();
    }
}
