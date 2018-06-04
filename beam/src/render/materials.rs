use mg::{ProgramBind, Texture, UniformBuffer, GlError};
use misc::{v3, V3};
use render::store::{MeshStore, TextureRef};

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum MaterialProp<T> {
    Texture(TextureRef),
    Value(T),
}

impl<T> Into<MaterialProp<T>> for TextureRef {
    fn into(self) -> MaterialProp<T> {
        MaterialProp::Texture(self)
    }
}

impl Into<MaterialProp<V3>> for V3 {
    fn into(self) -> MaterialProp<V3> {
        MaterialProp::Value(self)
    }
}

impl Into<MaterialProp<f32>> for f32 {
    fn into(self) -> MaterialProp<f32> {
        MaterialProp::Value(self)
    }
}

impl<'a> Into<MaterialProp<f32>> for &'a f32 {
    fn into(self) -> MaterialProp<f32> {
        MaterialProp::Value(*self)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Material {
    normal_: MaterialProp<V3>,
    albedo_: MaterialProp<V3>,
    metallic_: MaterialProp<f32>,
    roughness_: MaterialProp<f32>,
    ao_: MaterialProp<f32>,
    opacity_: MaterialProp<f32>,
}

macro_rules! setter {
    ($name:ident, $field:ident, $typ:ty) => {
        pub fn $name<T: Into<MaterialProp<$typ>>>(&self, $name: T) -> Material {
            let mut new = self.clone();
            new.$field = $name.into();
            new
        }
    }
}

impl Material {
    pub fn new() -> Material {
        Material {
            normal_: v3(0.5, 0.5, 1.0).into(),
            albedo_: v3(1.0, 1.0, 1.0).into(),
            metallic_: 1.0.into(),
            roughness_: 1.0.into(),
            ao_: 1.0.into(),
            opacity_: 1.0.into(),
        }
    }

    setter!(albedo, albedo_, V3);
    setter!(normal, normal_, V3);
    setter!(metallic, metallic_, f32);
    setter!(roughness, roughness_, f32);
    setter!(ao, ao_, f32);
    setter!(opacity, opacity_, f32);

    pub fn bind<P: ProgramBind>(&self, meshes: &MeshStore, program: &P) {
        macro_rules! prop {
            ($name:ident, $field:ident, $use:ident, $mat:ident, $fun:ident) => {{
                match &self.$field {
                    MaterialProp::Texture(texture_ref) => {
                        let texture = meshes.get_texture(&texture_ref);
                        program.bind_bool(concat!("use_mat_", stringify!($name)), false);
                        program.bind_texture(concat!("tex_", stringify!($name)), &texture);
                    }
                    MaterialProp::Value(value) => {
                        program.bind_bool(concat!("use_mat_", stringify!($name)), true);
                        program.$fun(concat!("mat_", stringify!($name)), *value);
                    }
                }
            }};
        }

        prop!(normal, normal_, use_mat_normal, mat_normal, bind_vec3);
        prop!(albedo, albedo_, use_mat_albedo, mat_albedo, bind_vec3);
        prop!(metallic, metallic_, use_mat_metallic, mat_metallic, bind_float);
        prop!(roughness, roughness_, use_mat_roughness, mat_roughness, bind_float);
        prop!(ao, ao_, use_mat_ao, mat_ao, bind_float);
        prop!(opacity, opacity_, use_mat_opacity, mat_opacity, bind_float);
    }
}

#[derive(Debug)]
pub struct Ibl {
    pub cubemap: Texture,
    pub irradiance_map: Texture,
    pub prefilter_map: Texture,
    pub brdf_lut: Texture,
}
