use misc::{Vertex, V3, v2, v3};
use render::mesh::calculate_tangent_and_bitangent;

macro_rules! v {
    ($pos:expr, $norm:expr, $tex:expr, $tangent:expr) => {{
        let tangent = $tangent.into();
        let norm = $norm.into();
        Vertex {
            pos: $pos.into(),
            tex: $tex.into(),
            norm: norm,
            tangent: tangent,
            // bitangent: tangent.cross(norm),
        }
    }};
}

pub fn cube_vertices() -> Vec<Vertex> {
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
pub fn sphere_verticies(radius: f32, nb_long: usize, nb_lat: usize) -> Vec<Vertex> {
    use std::f32::consts::PI;

    let mut verts: Vec<Vertex> = vec![Vertex::default(); (nb_long + 1) * nb_lat + 2];

    let up = v3(0.0, 1.0, 0.0) * radius;
    verts[0] = Vertex {
        pos: up,
        norm: up,
        tex: v2(0.0, 0.0),
        tangent: v3(0.0, 0.0, 0.0),
        // bitangent: v3(0.0, 0.0, 0.0),
    };
    for lat in 0..nb_lat {
        let a1 = PI * (lat as f32 + 1.0) / (nb_lat as f32 + 1.0);
        let (sin1, cos1) = a1.sin_cos();

        for lon in 0..=nb_long {
            let a2 = PI * 2.0 * (if lon == nb_long { 0.0 } else { lon as f32 }) / nb_long as f32;
            let (sin2, cos2) = a2.sin_cos();

            let pos = v3(
                sin1 * cos2,
                cos1,
                sin1 * sin2,
            );
            let norm = pos;
            let tex = v2(
                lon as f32 / nb_long as f32,
                1.0 - (lat as f32 + 1.0) / (nb_lat as f32 + 1.0)
            );

            verts[lon + lat * (nb_long + 1) + 1] = Vertex {
                pos: pos * radius,
                norm,
                tex,
                tangent: v3(0.0, 0.0, 0.0),
                // bitangent: v3(0.0, 0.0, 0.0),
            };
        }
    }
    let len = verts.len();
    verts[len - 1] = Vertex {
        pos: -up,
        norm: -up,
        tex: v2(0.0, 0.0),
        tangent: v3(0.0, 0.0, 0.0),
        // bitangent: v3(0.0, 0.0, 0.0),
    };

    let nb_faces = verts.len();
    let nb_triangles = nb_faces * 2;
    let nb_indices = nb_triangles * 3;

    let mut new_verts: Vec<Vertex> = Vec::with_capacity(nb_indices);

    let mut v = |i: usize| new_verts.push(verts[i].clone());

    for lon in 0..nb_long {
        v(lon + 2);
        v(lon + 1);
        v(0);
    }

    for lat in 0..(nb_lat - 1) {
        for lon in 0..nb_long {
            let current = lon + lat * (nb_long + 1) + 1;
            let next = current + nb_long + 1;

            v(current);
            v(current + 1);
            v(next + 1);

            v(current);
            v(next + 1);
            v(next);
        }
    }

    for lon in 0..nb_long {
        v(len - 1);
        v(len - (lon + 2) - 1);
        v(len - (lon + 1) - 1);
    }

    for vs in new_verts.chunks_mut(3) {
        if vs.len() < 3 {
            continue;
        }

        let x: *mut _ = &mut vs[0];
        let y: *mut _ = &mut vs[1];
        let z: *mut _ = &mut vs[2];
        unsafe {
            calculate_tangent_and_bitangent(&mut *x, &mut *y, &mut *z);
        }
    }
    new_verts
}
