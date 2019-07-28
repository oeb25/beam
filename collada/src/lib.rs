#![feature(custom_attribute)]

mod raw;

use failure::{Error, Fail};

#[derive(Debug)]
pub struct Image {
    pub name: String,
    pub source: String,
}

#[derive(Debug, Clone, Copy)]
pub struct ImageRef(pub usize);

#[derive(Debug)]
pub enum PhongProperty {
    Color([f32; 4]),
    Texture(ImageRef),
    Float(f32),
}

#[derive(Debug)]
pub enum Effect {
    Phong {
        emission: Option<PhongProperty>,
        ambient: Option<PhongProperty>,
        diffuse: Option<PhongProperty>,
        specular: Option<PhongProperty>,
        shininess: Option<PhongProperty>,
        index_of_refraction: Option<PhongProperty>,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct EffectRef(pub usize);

#[derive(Debug)]
pub enum Material {
    Effect(EffectRef),
}

#[repr(C)]
#[derive(Debug)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub nor: [f32; 3],
    pub tex: [f32; 2],
}

#[derive(Debug, Clone, Copy)]
pub struct MaterialRef(pub usize);

#[derive(Debug)]
pub struct MeshTriangles {
    pub vertices: Vec<Vertex>,
    pub material: MaterialRef,
}

#[derive(Debug)]
pub enum Geometry {
    ConvexMesh,
    Mesh { triangles: Vec<MeshTriangles> },
    Spline,
}

#[derive(Debug, Clone, Copy)]
pub struct GeometryRef(pub usize);

#[derive(Debug)]
pub struct Controller {}

#[derive(Debug)]
pub enum Transform {
    LookAt,
    Matrix([f32; 16]),
    Rotate,
    Scale,
    Skew,
    Translate,
}

#[derive(Debug)]
pub struct InstanceGeometry {
    pub geometry: GeometryRef,
    pub material: Option<Vec<MaterialRef>>,
}

#[derive(Debug)]
pub struct Node {
    pub transformations: Vec<Transform>,
    pub geometry: Vec<InstanceGeometry>,
}

#[derive(Debug)]
struct Asset;

#[derive(Debug)]
pub struct VisualScene {
    asset: Asset,
    pub nodes: Vec<Node>,
    evaluate_scene: (),
}
#[derive(Debug, Clone, Copy)]
pub struct VisualSceneRef(usize);

#[derive(Debug)]
pub struct Scene {
    physics: Vec<()>,
    visual: Option<VisualSceneRef>,
}

#[derive(Debug)]
#[allow(unused)]
pub struct Collada {
    pub images: Vec<Image>,
    pub effects: Vec<Effect>,
    pub materials: Vec<Material>,
    pub geometry: Vec<Geometry>,
    pub controllers: Vec<Controller>,
    pub visual_scenes: Vec<VisualScene>,
    pub scene: Scene,
}

#[derive(Debug, Fail)]
enum ColladaError {
    #[fail(display = "Technique should be the last element in effect")]
    LibrayTechniqueNotLast,
    #[fail(display = "No technique was found in effect")]
    LibrayTechniqueNotFound,
    #[fail(display = "Texture {:?} was not found", 0)]
    TextureNotFound(String),
    #[fail(display = "Sampler2D {:?} was not found", 0)]
    Sampler2DNotFound(String),
    #[fail(display = "Surface {:?} was not found", 0)]
    SurfaceNotFound(String),
    #[fail(display = "Visual scene {:?} was not found", 0)]
    VisualSceneNotFound(String),
}
use ColladaError::*;

impl Collada {
    pub fn parse(src: &str) -> Result<Collada, Error> {
        let data: raw::ColladaRaw = serde_xml_rs::deserialize(src.as_bytes())?;
        data.into()
    }
}

impl From<raw::ColladaRaw> for Result<Collada, Error> {
    #[allow(unused)]
    fn from(data: raw::ColladaRaw) -> Result<Collada, Error> {
        let raw::ColladaRaw {
            asset,
            library_images,
            library_effects,
            library_materials,
            library_geometries,
            library_controllers,
            library_visual_scenes,
            scene,
        } = data;
        let mut images = vec![];
        let mut image_ids = vec![];
        for (i, image) in library_images.image.into_iter().enumerate() {
            let img = Image {
                name: image.name.clone(),
                source: image.init_from,
            };
            images.push(img);
            image_ids.push((image.name, ImageRef(i)));
        }
        let mut effects = vec![];
        let mut effect_ids = vec![];
        for (i, effect) in library_effects.effect.into_iter().enumerate() {
            let id = effect.id;
            let effect = effect.profile_common;
            let mut technique = None;
            let mut newparams = vec![];
            for element in effect.elements.into_iter() {
                match element {
                    raw::LibraryEffectProfileCommonElement::NewParam(newparam) => {
                        if technique.is_some() {
                            return Err(LibrayTechniqueNotLast)?;
                        }
                        newparams.push(newparam);
                    }
                    raw::LibraryEffectProfileCommonElement::Technique(new_technique) => {
                        technique = Some(new_technique);
                    }
                }
            }
            // let technique = if let Some(technique) = technique {
            //     technique
            // } else {
            //     Err(LibrayTechniqueNotFound)?
            // };
            let technique = technique.ok_or(LibrayTechniqueNotFound)?;
            let convert = |x: &raw::TechniqueValue| match x {
                raw::TechniqueValue::Color { value, .. } => {
                    let mut result = [0.0; 4];
                    for (i, n) in value
                        .split_whitespace()
                        .map(|x| x.parse().unwrap())
                        .enumerate()
                    {
                        result[i] = n;
                    }
                    Ok(PhongProperty::Color(result))
                }
                raw::TechniqueValue::Texture { texture } => {
                    let id = newparams
                        .iter()
                        .find(|x| &x.sid == texture)
                        .ok_or(TextureNotFound(texture.clone()))?;
                    let id = match &id.kind {
                        raw::NewParamKind::Sampler2D { source } => {
                            newparams.iter().find(|x| &x.sid == source).ok_or_else(|| {
                                Sampler2DNotFound(source.clone()).context("second indirection")
                            })?
                        }
                        _ => unimplemented!("non sampler on second indirect"),
                    };
                    let id = match &id.kind {
                        raw::NewParamKind::Surface { init_from, .. } => {
                            image_ids
                                .iter()
                                .find(|x| &x.0 == init_from)
                                .ok_or(
                                    SurfaceNotFound(init_from.clone()).context("third indirection"),
                                )?
                                .1
                        }
                        _ => unimplemented!("non surface on third indirect"),
                    };
                    Ok(PhongProperty::Texture(id))
                }
                raw::TechniqueValue::Float { value, .. } => {
                    let res: Result<_, Error> = Ok(PhongProperty::Float(value.parse()?));
                    res
                }
            };
            let new_effect = match &technique {
                raw::Technique::Phong {
                    emission,
                    ambient,
                    diffuse,
                    specular,
                    shininess,
                    index_of_refraction,
                    ..
                } => Effect::Phong {
                    emission: emission.as_ref().map(convert).transpose()?,
                    ambient: ambient.as_ref().map(convert).transpose()?,
                    diffuse: diffuse.as_ref().map(convert).transpose()?,
                    specular: specular.as_ref().map(convert).transpose()?,
                    shininess: shininess.as_ref().map(convert).transpose()?,
                    index_of_refraction: index_of_refraction.as_ref().map(convert).transpose()?,
                },
            };

            effects.push(new_effect);
            effect_ids.push((id, EffectRef(i)));
        }

        let mut materials = vec![];
        let mut material_ids = vec![];
        for (i, material) in library_materials.material.into_iter().enumerate() {
            let id = effect_ids
                .iter()
                .find(|x| x.0 == &material.instance_effect.url[1..])
                .unwrap()
                .1;
            let new_material = Material::Effect(id);
            materials.push(new_material);
            material_ids.push((material.id, MaterialRef(i)));
        }

        let mut geometry = vec![];
        let mut geometry_ids = vec![];
        for (i, geom) in library_geometries.geometry.into_iter().enumerate() {
            let mesh = geom.mesh;
            let triangles = mesh
                .triangles
                .iter()
                .map(|triangles| {
                    let input = &triangles.input;
                    let v_input_id = &mesh.vertices.input;
                    let n_input_id = &input[1];
                    let t_input_id = &input[2];
                    let find = |s: &str| {
                        &mesh
                            .source
                            .iter()
                            .find(|x| x.id == s[1..])
                            .unwrap()
                            .float_array
                            .data
                    };
                    let vd = find(&v_input_id.source);
                    let nd = find(&n_input_id.source);
                    let td = find(&t_input_id.source);
                    let vertices = triangles
                        .p
                        .chunks(3)
                        .map(|x| (x[0] * 3, x[1] * 3, x[2] * 2))
                        .map(|(v, n, t)| {
                            (
                                [vd[v], vd[v + 1], vd[v + 2]],
                                [nd[n], nd[n + 1], nd[n + 2]],
                                [td[t], td[t + 1]],
                            )
                        })
                        .map(|(pos, nor, tex)| Vertex { pos, nor, tex })
                        .collect();

                    let material = material_ids
                        .iter()
                        .find(|x| x.0 == triangles.material)
                        .expect("could not find material for mesh")
                        .1;

                    MeshTriangles { vertices, material }
                })
                .collect();

            let new_geometry = Geometry::Mesh { triangles };

            geometry.push(new_geometry);
            geometry_ids.push((geom.id, GeometryRef(i)));
        }

        let controllers = vec![];

        let mut visual_scenes = vec![];
        let mut visual_scene_ids = vec![];
        for (i, scene) in library_visual_scenes.visual_scene.into_iter().enumerate() {
            let nodes = scene
                .nodes
                .into_iter()
                .map(|node| {
                    let (transformations, instance_geometry): (Vec<_>, Vec<_>) =
                        node.elements.into_iter().partition(|e| {
                            use raw::SceneNodeElement::*;
                            match e {
                                LookAt | Matrix(_) | Rotate | Scale | Skew | Translate => true,
                                InstanceGeometry(_) => false,
                            }
                        });

                    let transformations = transformations.into_iter().map(|x| match x {
                        raw::SceneNodeElement::LookAt => raw::Transformation::LookAt,
                        raw::SceneNodeElement::Matrix(a) => raw::Transformation::Matrix(a),
                        raw::SceneNodeElement::Rotate => raw::Transformation::Rotate,
                        raw::SceneNodeElement::Scale => raw::Transformation::Scale,
                        raw::SceneNodeElement::Skew => raw::Transformation::Skew,
                        raw::SceneNodeElement::Translate => raw::Transformation::Translate,
                        raw::SceneNodeElement::InstanceGeometry(_) => unreachable!(),
                    });

                    let instance_geometry = instance_geometry.into_iter().map(|x| match x {
                        raw::SceneNodeElement::InstanceGeometry(g) => g,
                        _ => unreachable!(),
                    });

                    let transformations = transformations
                        .map(|transform| match transform {
                            raw::Transformation::Matrix(data) => {
                                let mut matrix = [0.0; 16];
                                for (i, v) in data.into_iter().enumerate() {
                                    matrix[i] = v;
                                }
                                Transform::Matrix(matrix)
                            }
                            x => unimplemented!("{:?}", x),
                        })
                        .collect();
                    Node {
                        transformations,
                        geometry: instance_geometry
                            .into_iter()
                            .map(|geom| InstanceGeometry {
                                geometry: geometry_ids
                                    .iter()
                                    .find(|x| x.0 == &geom.url[1..])
                                    .expect("could not find geometry for node")
                                    .1,
                                material: geom.bind_material.map(|mat| {
                                    mat.technique_common
                                        .materials
                                        .into_iter()
                                        .map(|t| match t {
                                            raw::BindTechniqueCommon::InstanceMaterial {
                                                target,
                                                ..
                                            } => {
                                                material_ids
                                                    .iter()
                                                    .find(|x| x.0 == &target[1..])
                                                    .expect("could not find material for node")
                                                    .1
                                            }
                                        })
                                        .collect()
                                }),
                            })
                            .collect(),
                    }
                })
                .collect();

            let new_scene = VisualScene {
                asset: Asset,
                nodes,
                evaluate_scene: (),
            };
            visual_scenes.push(new_scene);
            visual_scene_ids.push((scene.id, VisualSceneRef(i)));
        }

        let scene = Scene {
            physics: vec![],
            visual: Some(
                visual_scene_ids
                    .iter()
                    .find(|x| x.0 == &scene.instance_visual_scene.url[1..])
                    .ok_or_else(|| {
                        VisualSceneNotFound(scene.instance_visual_scene.url.clone())
                            .context("from main scene")
                    })?
                    .1,
            ),
        };

        let out = Collada {
            images,
            effects,
            materials,
            geometry,
            controllers,
            visual_scenes,
            scene,
        };

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std;

    fn it_works_run() -> Result<(), Error> {
        let srcs = ["./suzanne.dae", "./owl.dae"];
        for src in &srcs {
            println!("parsing {:?}", src);
            let src = std::fs::read_to_string(src)?;
            let data: raw::ColladaRaw = serde_xml_rs::deserialize(src.as_bytes())?;
            let _dae: Result<Collada, _> = data.into();
        }

        Ok(())
    }

    #[test]
    fn it_works() {
        it_works_run().unwrap()
    }
}
