#![feature(fs_read_write, stmt_expr_attributes, transpose_result, box_syntax, box_patterns)]
#![feature(custom_attribute, nll, iterator_flatten)]

extern crate cgmath;
extern crate collada;
extern crate genmesh;
extern crate gl;
extern crate glutin;
extern crate image;
extern crate obj;
extern crate time;
extern crate mg;
extern crate warmy;

use cgmath::{InnerSpace, Rad};

use glutin::GlContext;

use time::{Duration, PreciseTime};

mod hot;
mod pipeline;
mod render;
// mod logic;

use pipeline::*;
use render::*;

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
    point_lights: Vec<PointLight>,
    directional_lights: Vec<DirectionalLight>,
}

impl Scene {
    fn new(screen_width: u32, screen_height: u32) -> Scene {
        // let game = logic::Game::new((8,8)); 

        let light_pos1 = v3(1.5, 1.0, 0.0);
        let light_pos2 = v3(1.5 + -10.0 * (23.0 / 14.0 as f32).sin(), 2.0, 0.0);

        let one = v3(1.0, 1.0, 1.0);

        let sun = DirectionalLight {
            color: rgb(255,0,255) * 0.2,
            direction: v3(0.0, 1.0, -1.5).normalize(),

            shadow_map: ShadowMap::new(),
        };

        let point_lights = vec![
            PointLight {
                color: v3(0.7, 0.4, 0.2) * 0.0,

                position: light_pos2,
                last_shadow_map_position: light_pos2,

                shadow_map: Some(PointShadowMap::new()),
            },
            PointLight {
                color: v3(1.0, 0.0, 0.2) * 0.0,

                position: light_pos1 + light_pos2,
                last_shadow_map_position: light_pos1 + light_pos2,

                shadow_map: None,
            },
            PointLight {
                color: v3(0.2, 0.2, 0.8) * 0.0,

                position: v3(
                    light_pos1.x * light_pos2.x,
                    1.0,
                    light_pos1.z * light_pos2.z,
                ),
                last_shadow_map_position: one,

                shadow_map: None,
            },
        ];

        let directional_lights = vec![sun];

        let camera = Camera::new(
            v3(-15.0, 5.0, 0.0),
            Rad(std::f32::consts::PI / 2.0),
            (screen_width as f32) / (screen_height as f32),
        );

        Scene {
            camera,
            point_lights,
            directional_lights,
        }
    }
    fn tick(&mut self, _t: f32, _dt: f32, inputs: &Input) {
        let pi = std::f32::consts::PI;

        let up = self.camera.up();
        let right = self.camera.front().cross(up).normalize();
        let front = up.cross(right).normalize();

        let walk_speed = 0.1;
        let sensitivity = 0.005;

        self.camera.pos += walk_speed
            * (front * (inputs.w - inputs.s) + right * (inputs.d - inputs.a)
                + up * (inputs.space - inputs.shift));
        self.camera.yaw += sensitivity * inputs.mouse_delta.0;
        self.camera.pitch = (self.camera.pitch - sensitivity * inputs.mouse_delta.1)
            .max(-pi / 2.001)
            .min(pi / 2.001);
        // self.camera.yaw += sensitivity * (inputs.right - inputs.left);
        // self.camera.pitch = (self.camera.pitch + sensitivity * (inputs.up - inputs.down))
        //     .max(-pi / 2.001)
        //     .min(pi / 2.001);
    }
}

fn main() {
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
    // gl_window.window().set_cursor_state(glutin::CursorState::Grab).unwrap();

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

    let room_ibl = pipeline.load_ibl("assets/Newport_Loft/Newport_Loft_Ref.hdr");
    let rust_material = pipeline.meshes.load_pbr_with_default_filenames(
        "assets/pbr/rusted_iron",
        "png",
    );
    let plastic_material = pipeline.meshes.load_pbr_with_default_filenames(
        "assets/pbr/plastic",
        "png",
    );
    let gold_material = pipeline.meshes.load_pbr_with_default_filenames(
        "assets/pbr/gold",
        "png",
    );

    let white4 = pipeline.meshes.rgba_texture(v4(1.0, 1.0, 1.0, 1.0));
    let white3 = pipeline.meshes.rgb_texture(v3(1.0, 1.0, 1.0));
    let gray3 = pipeline.meshes.rgb_texture(v3(0.5, 0.5, 0.5));
    let blueish4 = pipeline.meshes.rgba_texture(v4(0.2, 0.5, 1.0, 1.0));
    let black3 = pipeline.meshes.rgb_texture(v3(0.0, 0.0, 0.0));
    let normal3 = pipeline.meshes.rgb_texture(v3(0.5, 0.5, 1.0));

    let suzanne = pipeline.meshes
        .load_collada("assets/suzanne/suzanne.dae")
        .scale(1.0 / 2.0)
        .translate(v3(0.0, 20.0, 0.0));

    let mut is = vec![];

    let mut m = 0;

    let pbr_materials = vec![rust_material, plastic_material, gold_material];
    let nr_pbr_materials = pbr_materials.len();

    for i in 1..5 {
        for n in 0..i {
            let x = i as f32 / 2.0;
            let v = v3(i as f32 / 2.0, -i as f32 - 5.0, n as f32 - x) * 2.0;
            let v = Mat4::from_translation(v) * Mat4::from_angle_y(Rad(i as f32 - 1.0));
            let obj = suzanne.transform(v);// .with_material(pbr_materials[m % nr_pbr_materials]);
            m += 1;
            is.push(obj);
        }
    }

    println!("drawing {} nanosuits", is.len());

    let cube_mesh = RenderObject::mesh(pipeline.meshes.get_cube());
    let sphere_mesh = RenderObject::mesh(pipeline.meshes.get_sphere(0.5));

    let mut fps_last_time = PreciseTime::now();
    let fps_step = Duration::seconds(1);
    let mut fps_number_of_frames = 0;

    let mut shadows_last_time = PreciseTime::now();
    let shadows_step = Duration::milliseconds(16 * 1);

    let mut update_shadows = false;

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
                    gl_window.resize(w, h);
                    pipeline.resize(w, h);
                },
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
                    }
                }
                // x => println!("{:?}", x),
                _ => {},
            },
            glutin::Event::DeviceEvent { event, .. } => {
                match event {
                    glutin::DeviceEvent::MouseMotion { delta } => {
                        inputs.mouse_delta = (delta.0 as f32, delta.1 as f32);
                    }
                    _ => {},
                }
            },
            // x => println!("{:?}", x),
            _ => (),
        });

        t += 1.0;

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
                .map(|(p, s)|
                    cube_mesh.transform(
                        Mat4::from_translation(*p)
                        * Mat4::from_nonuniform_scale(s.x, s.y, s.z)
                    )
                )
                .collect();

            for light in scene.point_lights.iter() {
                let mesh = sphere_mesh
                    .transform(
                        Mat4::from_translation(light.position.clone()) *
                        Mat4::from_scale(0.8)
                    )
                    .with_material(Material {
                        albedo: pipeline.meshes.rgb_texture(light.color),
                        normal: normal3,
                        metallic: black3,
                        roughness: white3,
                        ao: white3,
                    });
                objects.push(mesh);
            }

            objects.append(&mut is.clone());

            pipeline.render(
                update_shadows,
                RenderProps {
                    camera: &scene.camera,
                    directional_lights: &mut scene.directional_lights,
                    point_lights: &mut scene.point_lights,
                    time: t,

                    ibl: &room_ibl,

                    ambient_intensity: Some(0.1),
                    skybox_intensity: Some(0.1),
                },
                &objects,
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
}
