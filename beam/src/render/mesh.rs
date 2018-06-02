use std::mem;

use mg::{
    DrawMode, FramebufferBinderDrawer, ProgramBind, VertexArray, VertexArrayBinder, VertexArrayPin,
    VertexBuffer, VertexBufferBinder,
};
use misc::{Mat4, V3, Vertex};

macro_rules! offset_of {
    ($ty:ty, $field:ident) => {
        #[allow(unused_unsafe)]
        unsafe {
            &(*(0 as *const $ty)).$field as *const _ as usize
        }
    };
}

macro_rules! size_of {
    ($ty:ty, $field:ident) => {
        #[allow(unused_unsafe)]
        unsafe {
            mem::size_of_val(&(*(0 as *const $ty)).$field)
        }
    };
}

macro_rules! implement_vertex {
    ($s:ty, [ $($field:ident,)* ]) => {
        #[allow(unused)]
        impl $s {
            fn vertex_attribs(vao: &mut VertexArrayBinder, vbo: &mut VertexBufferBinder<$s>) {
                let float_size = mem::size_of::<f32>();

                let mut i = 0;
                $(
                    vao.vbo_attrib(
                        &vbo,
                        i,
                        size_of!($s, $field) / float_size,
                        offset_of!($s, $field),
                    );
                    i += 1;
                )*
            }
        }
    }
}

implement_vertex!(Vertex, [pos, norm, tex, tangent,]);

#[derive(Debug)]
pub struct Mesh {
    vcount: usize,
    vao: VertexArray,
    pub simple_verts: Vec<V3>,
}

impl Mesh {
    pub fn new(vertices: &[Vertex], vpin: &mut VertexArrayPin) -> Mesh {
        let simple_verts = vertices.iter().map(|v| v.pos).collect();

        let vao = VertexArray::new();
        let mut vbo = VertexBuffer::from_data(vertices);

        {
            let float_size = mem::size_of::<f32>();
            let vao_binder = vao.bind(vpin);
            let vbo_binder = vbo.bind();

            macro_rules! x {
                ($i:expr, $e:ident) => (
                    vao_binder.vbo_attrib(
                        &vbo_binder,
                        $i,
                        size_of!(Vertex, $e) / float_size,
                        offset_of!(Vertex, $e)
                    )
                )
            }

            x!(0, pos);
            x!(1, norm);
            x!(2, tex);
            x!(3, tangent);
            // x!(4, bitangent);
        }

        Mesh {
            vcount: vertices.len(),
            vao,
            simple_verts,
        }
    }
    pub fn bind<'a>(&'a self, vpin: &'a mut VertexArrayPin) -> MeshBinding<'a> {
        MeshBinding {
            vao: self.vao.bind(vpin),
            vcount: self.vcount,
        }
    }
}

pub struct MeshBinding<'a> {
    vao: VertexArrayBinder<'a>,
    vcount: usize,
}
impl<'a> MeshBinding<'a> {
    pub fn draw<F, P>(&self, fbo: &F, program: &P)
    where
        F: FramebufferBinderDrawer,
        P: ProgramBind,
    {
        // self.bind_textures(program);

        self.vao
            .draw_arrays(fbo, program, DrawMode::Triangles, 0, self.vcount);
    }
    pub fn draw_geometry_instanced<F, P>(
        &self,
        fbo: &F,
        _program: &P,
        transforms: &VertexBufferBinder<Mat4>,
    ) where
        F: FramebufferBinderDrawer,
        P: ProgramBind,
    {
        let offset = 5;
        let width = 4;
        for i in 0..width {
            let index = i + offset;
            self.vao
                .vbo_attrib(&transforms, index, width, width * i * mem::size_of::<f32>())
                .attrib_divisor(index, 1);
        }

        self.vao
            .draw_arrays_instanced(fbo, DrawMode::Triangles, 0, self.vcount, transforms.len());
    }
    pub fn draw_instanced<F, P>(&self, fbo: &F, program: &P, transforms: &VertexBufferBinder<Mat4>)
    where
        F: FramebufferBinderDrawer,
        P: ProgramBind,
    {
        // self.bind_textures(program);
        self.draw_geometry_instanced(fbo, program, transforms);
    }
}

pub fn calculate_tangent_and_bitangent(va: &mut Vertex, vb: &mut Vertex, vc: &mut Vertex) {
    let v1 = va.pos;
    let v2 = vb.pos;
    let v3 = vc.pos;

    let w1 = va.tex;
    let w2 = vb.tex;
    let w3 = vc.tex;

    let d1 = v2 - v1;
    let d2 = v3 - v1;
    let t1 = w2 - w1;
    let t2 = w3 - w1;

    let r = 1.0 / (t1.x * t2.y - t2.x * t1.y);
    let sdir = r * (t2.y * d1 - t1.y * d2);
    // let tdir = r * (t1.x * d2 - t2.x * d1);

    va.tangent = sdir;
    // va.bitangent = tdir;
    vb.tangent = sdir;
    // vb.bitangent = tdir;
    vc.tangent = sdir;
    // vc.bitangent = tdir;
}
