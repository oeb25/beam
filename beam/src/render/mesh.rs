use std::mem;

use mg::{
    DrawMode, FramebufferBinderDrawer, ProgramBinding, VertexArray, VertexBuffer,
    VertexBufferBinder,
};
use misc::{Mat4, Vertex};

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

pub struct Mesh {
    vcount: usize,
    vao: VertexArray,
}

impl Mesh {
    pub fn new(vertices: &[Vertex]) -> Mesh {
        let mut vao = VertexArray::new();
        let mut vbo = VertexBuffer::from_data(vertices);

        {
            let float_size = mem::size_of::<f32>();
            let vao_binder = vao.bind();
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
            vao: vao,
        }
    }
    pub fn bind(&mut self) -> MeshBinding {
        MeshBinding(self)
    }
}

pub struct MeshBinding<'a>(&'a mut Mesh);
impl<'a> MeshBinding<'a> {
    pub fn draw<F>(&mut self, fbo: &F, program: &ProgramBinding)
    where
        F: FramebufferBinderDrawer,
    {
        // self.bind_textures(program);

        self.0
            .vao
            .bind()
            .draw_arrays(fbo, program, DrawMode::Triangles, 0, self.0.vcount);
    }
    pub fn draw_geometry_instanced<F>(
        &mut self,
        fbo: &F,
        _program: &ProgramBinding,
        transforms: &VertexBufferBinder<Mat4>,
    ) where
        F: FramebufferBinderDrawer,
    {
        let mut vao = self.0.vao.bind();
        let offset = 5;
        let width = 4;
        for i in 0..width {
            let index = i + offset;
            vao.vbo_attrib(&transforms, index, width, width * i * mem::size_of::<f32>())
                .attrib_divisor(index, 1);
        }

        vao.draw_arrays_instanced(fbo, DrawMode::Triangles, 0, self.0.vcount, transforms.len());
    }
    pub fn draw_instanced<F>(
        &mut self,
        fbo: &F,
        program: &ProgramBinding,
        transforms: &VertexBufferBinder<Mat4>,
    ) where
        F: FramebufferBinderDrawer,
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
